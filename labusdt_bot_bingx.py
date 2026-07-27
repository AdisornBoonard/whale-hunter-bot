"""
confluence_3cond_bot.py
------------------------
Single-file drop-in bot for BingX Futures — trades the "High-Probability
Confluence + Distance Filter Dashboard (3 Conditions)" Pine Script: Order
Block + RSI Divergence, gated by 3 INDEPENDENT EMA trend filters. Each
condition manages its own single open ticket (never re-enters itself while
its own trade is open) but the 3 conditions never block each other — they
can all be in a trade at the same time.

Same deploy pattern as before: one file, one process, PORT from the
environment, a background thread doing the work while a web server keeps
the service alive/health-checkable online.

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python confluence_3cond_bot.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT confluence_3cond_bot:app

⚠️ MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders
firing on the same signal.

⚠️ REAL-EXCHANGE CAVEAT (read this): all 3 conditions trade the SAME symbol.
If two conditions are LONG at once, BingX merges them into ONE real LONG
position (it has no concept of "condition"). Each condition still gets its
own REAL take-profit/stop-loss bracket orders sized to its own ticket, so
real protection exists — but the dashboard's per-condition win/loss stats
are tracked in SOFTWARE (checking each condition's own TP/SL against the
latest closed candle, exactly like the Pine Script does with high/low).
In the rare case two same-side tickets from different conditions are open
together, attribution between them can be imprecise at the edges — the
money-management (bracket orders) is still real and correct either way.
"""

import os
import json
import threading
import uuid
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Response

load_dotenv()
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

# ============================================================================
# 1) INDICATOR - Python port of the Pine Script (Order Block + RSI Divergence)
# ============================================================================

def find_pivot_highs_lows(high: np.ndarray, low: np.ndarray, left: int, right: int):
    n = len(high)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for j in range(left, n - right):
        wh = high[j - left: j + right + 1]
        wl = low[j - left: j + right + 1]
        if np.all(np.isnan(wh)) or np.all(np.isnan(wl)):
            continue
        if high[j] == np.nanmax(wh):
            is_high[j] = True
        if low[j] == np.nanmin(wl):
            is_low[j] = True
    return is_high, is_low


def shift_by(arr: np.ndarray, right: int, fill=np.nan):
    """result[i] = arr[i-right], mirrors Pine's `[right]` lookback."""
    n = len(arr)
    if arr.dtype == bool:
        out = np.zeros(n, dtype=bool)
        if right < n:
            out[right:] = arr[: n - right]
        return out
    out = np.full(n, fill)
    if right < n:
        out[right:] = arr[: n - right]
    return out


def value_when(cond: np.ndarray, source: np.ndarray, occurrence: int):
    """Python port of ta.valuewhen(cond, source, occurrence)."""
    n = len(cond)
    out = np.full(n, np.nan)
    hist = []
    for i in range(n):
        if cond[i]:
            hist.append(source[i])
        if len(hist) > occurrence:
            out[i] = hist[-1 - occurrence]
    return out


def rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder's smoothing (what Pine's ta.rsi uses internally)."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < length:
        return out
    alpha = 1.0 / length
    out[length - 1] = np.nanmean(values[:length])
    for i in range(length, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi_series(close: np.ndarray, length: int) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up[np.isnan(delta)] = np.nan
    down[np.isnan(delta)] = np.nan
    roll_up = rma(np.nan_to_num(up, nan=0.0), length)
    roll_down = rma(np.nan_to_num(down, nan=0.0), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = roll_up / roll_down
        rsi = 100 - 100 / (1 + rs)
        rsi[roll_down == 0] = 100.0
    return rsi


def ema_series(close: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(close).ewm(span=length, adjust=False).mean().values


def compute_shared_indicator(df: pd.DataFrame, cfg: dict) -> dict:
    """Order Block (swing) detection + RSI divergence — identical for all
    3 conditions in the Pine Script, so computed once here."""
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n = len(df)
    length = 5

    is_high, is_low = find_pivot_highs_lows(high, low, length, length)
    ph_confirmed = shift_by(is_high, length, fill=False)
    pl_confirmed = shift_by(is_low, length, fill=False)
    high_shift = shift_by(high, length)
    low_shift = shift_by(low, length)

    last_bull_ob = np.full(n, np.nan)
    last_bear_ob = np.full(n, np.nan)
    lb, ls = np.nan, np.nan
    for i in range(n):
        if pl_confirmed[i]:
            lb = low_shift[i]
        if ph_confirmed[i]:
            ls = high_shift[i]
        last_bull_ob[i] = lb
        last_bear_ob[i] = ls

    near_bull_ob = (~np.isnan(last_bull_ob)) & (np.abs(close - last_bull_ob) / np.where(last_bull_ob == 0, np.nan, last_bull_ob) < 0.015)
    near_bear_ob = (~np.isnan(last_bear_ob)) & (np.abs(close - last_bear_ob) / np.where(last_bear_ob == 0, np.nan, last_bear_ob) < 0.015)

    rsi = rsi_series(close, cfg["rsi_len"])
    rsi_shift = shift_by(rsi, length)

    crossover_os = np.zeros(n, dtype=bool)
    crossunder_ob = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(rsi[i]) and not np.isnan(rsi[i - 1]):
            if rsi[i - 1] <= cfg["rsi_os"] and rsi[i] > cfg["rsi_os"]:
                crossover_os[i] = True
            if rsi[i - 1] >= cfg["rsi_ob"] and rsi[i] < cfg["rsi_ob"]:
                crossunder_ob[i] = True

    if cfg["use_rsi_div"]:
        price_ll = low < value_when(pl_confirmed, low_shift, 1)
        rsi_hl = rsi > value_when(pl_confirmed, rsi_shift, 1)
        bullish_div = price_ll & rsi_hl & (rsi < 45)

        price_hh = high > value_when(ph_confirmed, high_shift, 1)
        # NOTE: ported literally from the original Pine — it compares rsi to
        # a PRICE reference (high[len]) here, not to rsi[len]. Kept as-is for
        # fidelity to the source script.
        rsi_lh = rsi < value_when(ph_confirmed, high_shift, 1)
        bearish_div = price_hh & rsi_lh & (rsi > 55)
    else:
        bullish_div = crossover_os
        bearish_div = crossunder_ob

    return {
        "near_bull_ob": near_bull_ob, "near_bear_ob": near_bear_ob,
        "last_bull_ob": last_bull_ob, "last_bear_ob": last_bear_ob,
        "rsi": rsi, "bullish_div": np.nan_to_num(bullish_div, nan=0).astype(bool),
        "bearish_div": np.nan_to_num(bearish_div, nan=0).astype(bool),
    }


def condition_signal(ind: dict, close: np.ndarray, idx: int, cfg: dict, use_ema: bool, ema_len: int):
    """Raw long/short for ONE condition at bar `idx` (the last closed candle)."""
    ema = ema_series(close, ema_len)
    uptrend = (not use_ema) or (close[idx] > ema[idx])
    downtrend = (not use_ema) or (close[idx] < ema[idx])
    raw_long = bool(uptrend and ind["near_bull_ob"][idx] and
                    (ind["bullish_div"][idx] or ind["rsi"][idx] < cfg["rsi_os"]))
    raw_short = bool(downtrend and ind["near_bear_ob"][idx] and
                     (ind["bearish_div"][idx] or ind["rsi"][idx] > cfg["rsi_ob"]))
    return raw_long, raw_short


def condition1_signal(ind: dict, close: np.ndarray, idx: int, cfg: dict):
    """เงื่อนไข 1: เหมือน condition_signal ทุกอย่าง แต่ถ้า 'Reversal จากระยะห่าง
    EMA' เปิดอยู่ และราคาห่างจาก EMA เส้นที่สอง (c1_rev_ema_len) เกิน
    c1_rev_pct% ให้สลับสัญญาณ Long<->Short (เทรดสวนกลับ)."""
    base_long, base_short = condition_signal(ind, close, idx, cfg, cfg["c1_use_ema"], cfg["c1_ema_len"])

    is_far = False
    if cfg.get("c1_use_rev_dist"):
        rev_ema = ema_series(close, cfg["c1_rev_ema_len"])
        ref = rev_ema[idx]
        dist = abs(close[idx] - ref) / ref if ref else 0.0
        is_far = dist >= (cfg["c1_rev_pct"] / 100.0)

    if is_far:
        return base_short, base_long
    return base_long, base_short


def calc_entry_sl_tp(cfg: dict, is_long: bool, entry_price: float, last_bull_ob, last_bear_ob):
    if cfg["use_auto_tp_sl"]:
        if is_long:
            sl = last_bull_ob * 0.995 if last_bull_ob and not np.isnan(last_bull_ob) else entry_price * (1 - 0.01)
            tp = entry_price + abs(entry_price - sl) * 1.618
        else:
            sl = last_bear_ob * 1.005 if last_bear_ob and not np.isnan(last_bear_ob) else entry_price * (1 + 0.01)
            tp = entry_price - abs(sl - entry_price) * 1.618
    else:
        if is_long:
            sl = entry_price * (1 - cfg["fix_sl_pct"] / 100.0)
            tp = entry_price * (1 + cfg["fix_tp_pct"] / 100.0)
        else:
            sl = entry_price * (1 + cfg["fix_sl_pct"] / 100.0)
            tp = entry_price * (1 - cfg["fix_tp_pct"] / 100.0)
    return sl, tp


# ============================================================================
# 2) SHARED STATE
# ============================================================================

DEFAULT_CONFIG = {
    "symbol": "LAB/USDT:USDT",
    "timeframe": "5m",
    "leverage": 25,
    "bot_start_date": datetime.now().strftime("%Y-%m-%d"),

    "initial_cap": 10.0,
    "base_order_usdt": 1.0,
    "fee_pct": 0.04,

    "min_dist_pct": 2.0,
    "use_rsi_div": True,
    "rsi_len": 9, "rsi_ob": 65, "rsi_os": 5,

    "use_auto_tp_sl": False,
    "fix_tp_pct": 5.5,
    "fix_sl_pct": 3.0,

    "c1_use_ema": False, "c1_ema_len": 150,
    "c1_use_rev_dist": False, "c1_rev_ema_len": 50, "c1_rev_pct": 3.0,
    "c2_use_ema": True,  "c2_ema_len": 150,
    "c3_use_ema": True,  "c3_ema_len": 100,

    "poll_seconds": 10,
}

CONDITIONS = ["C1", "C2", "C3"]


def empty_condition_state():
    return {
        "in_position": False, "pos_side": None, "entry_price": None,
        "tp": None, "sl": None, "contract_amount": None,
        "last_long_entry": None, "last_short_entry": None,
        "total_trades": 0, "wins": 0, "losses": 0, "net_profit": 0.0,
    }


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(DEFAULT_CONFIG)
        self.cond_state = {c: empty_condition_state() for c in CONDITIONS}
        self.running = False
        self.connected = False
        self.trades = []
        self.logs = []
        self.ohlcv = []
        self.indicator = {"timestamps": [], "bull": [], "bear": []}
        self.live_price = None
        self.balance = 0.0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self.config.update(data.get("config", {}))
                saved_cond = data.get("cond_state", {})
                for c in CONDITIONS:
                    if c in saved_cond:
                        self.cond_state[c].update(saved_cond[c])
                self.trades = data.get("trades", [])
                self.logs = data.get("logs", [])
            except Exception:
                pass

    def _save(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"config": self.config, "cond_state": self.cond_state,
                           "trades": self.trades[:500], "logs": self.logs[-300:]}, f)
        except Exception:
            pass

    def update_config(self, patch: dict):
        with self.lock:
            self.config.update(patch)
            self._save()

    def add_log(self, text: str):
        with self.lock:
            self.logs.append({"time": datetime.now().strftime("%H:%M:%S"), "text": text})
            self.logs = self.logs[-300:]
            self._save()

    def add_trade(self, trade: dict):
        with self.lock:
            self.trades.insert(0, trade)
            self.trades = self.trades[:500]
            self._save()

    def update_market(self, ohlcv, indicator, live_price):
        with self.lock:
            self.ohlcv = ohlcv
            self.indicator = indicator
            self.live_price = live_price

    def snapshot(self):
        with self.lock:
            combined_trades = sum(cs["total_trades"] for cs in self.cond_state.values())
            combined_wins = sum(cs["wins"] for cs in self.cond_state.values())
            combined_losses = sum(cs["losses"] for cs in self.cond_state.values())
            combined_net = sum(cs["net_profit"] for cs in self.cond_state.values())
            combined_winrate = round((combined_wins / (combined_wins + combined_losses)) * 100, 1) if (combined_wins + combined_losses) else 0.0
            return {
                "config": dict(self.config),
                "cond_state": {c: dict(v) for c, v in self.cond_state.items()},
                "running": self.running,
                "connected": self.connected,
                "live_price": self.live_price,
                "balance": self.balance,
                "combined": {
                    "total_trades": combined_trades, "wins": combined_wins, "losses": combined_losses,
                    "win_rate": combined_winrate, "net_profit": round(combined_net, 4),
                    "portfolio_balance": round(self.config["initial_cap"] + combined_net, 4),
                },
                "trades": self.trades[:60],
                "logs": self.logs[-100:],
                "ohlcv": self.ohlcv,
                "indicator": self.indicator,
            }


state = BotState()

# ============================================================================
# 3) TRADING ENGINE - ccxt / BingX
# ============================================================================

_exchange = None
_stop_flag = threading.Event()
_thread = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bingx({
            "apiKey": BINGX_API_KEY, "secret": BINGX_SECRET_KEY,
            "enableRateLimit": True, "timeout": 15000,
            "options": {"defaultType": "swap"},
        })
    return _exchange


def _is_trading_started(cfg: dict) -> bool:
    try:
        start_dt = datetime.strptime(cfg.get("bot_start_date", ""), "%Y-%m-%d").date()
    except Exception:
        return True
    return datetime.now().date() >= start_dt


def fire_entry(condition: str, symbol: str, side: str, entry_price: float, cfg: dict, manual=False):
    """Opens a real market order for this condition's ticket + real TP/SL
    bracket orders sized to just this ticket's contract_amount."""
    ex = get_exchange()
    cs = state.cond_state[condition]
    try:
        margin = cfg["base_order_usdt"]
        leverage = cfg["leverage"]
        contract_amount = round((margin * leverage) / entry_price, 4)

        sl, tp = calc_entry_sl_tp(cfg, side == "LONG", entry_price,
                                   state.indicator.get("last_bull_ob"), state.indicator.get("last_bear_ob"))
        sl = round(float(sl), 6)
        tp = round(float(tp), 6)

        try:
            ex.set_leverage(leverage, symbol)
        except Exception:
            pass

        order_side = "buy" if side == "LONG" else "sell"
        ex.create_order(symbol=symbol, type="market", side=order_side,
                         amount=contract_amount, params={"positionSide": side})
        try:
            tp_sl_side = "sell" if side == "LONG" else "buy"
            ex.create_order(symbol=symbol, type="TAKE_PROFIT_MARKET", side=tp_sl_side,
                             amount=contract_amount,
                             params={"positionSide": side, "stopPrice": tp, "workingType": "MARK_PRICE"})
            ex.create_order(symbol=symbol, type="STOP_MARKET", side=tp_sl_side,
                             amount=contract_amount,
                             params={"positionSide": side, "stopPrice": sl, "workingType": "MARK_PRICE"})
        except Exception:
            pass

        with state.lock:
            cs["in_position"] = True
            cs["pos_side"] = side
            cs["entry_price"] = entry_price
            cs["tp"] = tp
            cs["sl"] = sl
            cs["contract_amount"] = contract_amount
            if side == "LONG":
                cs["last_long_entry"] = entry_price
            else:
                cs["last_short_entry"] = entry_price
            state._save()

        tag = " [MANUAL]" if manual else ""
        state.add_trade({
            "id": uuid.uuid4().hex[:10], "condition": condition,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_ms": int(datetime.now().timestamp() * 1000),
            "type": f"OPEN ({'BUY' if side == 'LONG' else 'SELL'}) {condition}{tag}",
            "entry": entry_price, "tp": tp, "sl": sl,
            "contract_amount": contract_amount,
            "pnl": None, "status": "OPEN", "side": side,
        })
        state.add_log(f"[{condition}] ENTRY EXECUTED{tag}: {side} @ {entry_price} | TP {tp} / SL {sl}")
        return True, "order sent"
    except Exception as e:
        state.add_log(f"⚠️ [{condition}] ORDER FAILED ({side}): {e}")
        return False, str(e)


def close_condition_ticket(condition: str, exit_price: float, reason: str):
    """Software-side bookkeeping close (bracket order on the exchange does
    the REAL closing — this just records the result for the dashboard)."""
    cs = state.cond_state[condition]
    with state.lock:
        if not cs["in_position"]:
            return
        entry = cs["entry_price"] or 0
        amount = cs["contract_amount"] or 0
        side = cs["pos_side"]
        pnl = (exit_price - entry) * amount if side == "LONG" else (entry - exit_price) * amount
        notional = cfg_fee_notional(amount, entry)
        pnl -= notional  # หักค่าธรรมเนียมโดยประมาณ (เข้า+ออก)

        cs["total_trades"] += 1
        if pnl >= 0:
            cs["wins"] += 1
        else:
            cs["losses"] += 1
        cs["net_profit"] = round(cs["net_profit"] + pnl, 6)

        # อัปเดตแถวออเดอร์ล่าสุดของเงื่อนไขนี้ที่ยัง OPEN อยู่
        for t in state.trades:
            if t.get("condition") == condition and t.get("status") == "OPEN":
                t["status"] = "WIN" if pnl >= 0 else "LOSS"
                t["exit"] = exit_price
                t["exit_ms"] = int(datetime.now().timestamp() * 1000)
                t["pnl"] = round(pnl, 4)
                break

        cs["in_position"] = False
        cs["pos_side"] = None
        cs["entry_price"] = None
        cs["tp"] = None
        cs["sl"] = None
        cs["contract_amount"] = None
        state._save()

    state.add_log(f"[{condition}] Position closed ({reason}) @ ~{exit_price} | PnL {round(pnl,4)} USDT")


def cfg_fee_notional(amount, entry_price):
    cfg = state.config
    notional_val = cfg["base_order_usdt"] * cfg["leverage"]
    return notional_val * (cfg["fee_pct"] / 100.0) * 2


def open_trade_manual(condition: str, side: str):
    cfg = state.snapshot()["config"]
    symbol = cfg["symbol"]
    live_price = state.live_price
    if not live_price:
        return False, "No live price yet"
    cs = state.cond_state[condition]
    if cs["in_position"]:
        return False, f"{condition} already has an open ticket"
    return fire_entry(condition, symbol, side, live_price, cfg, manual=True)


def close_trade_manual(condition: str):
    cs = state.cond_state[condition]
    if not cs["in_position"]:
        return False, "No open ticket for this condition"
    cfg = state.snapshot()["config"]
    symbol = cfg["symbol"]
    side = cs["pos_side"]
    amount = cs["contract_amount"]
    ex = get_exchange()
    close_side = "sell" if side == "LONG" else "buy"
    try:
        ex.create_order(symbol=symbol, type="market", side=close_side, amount=amount,
                         params={"positionSide": side, "reduceOnly": True})
        exit_price = state.live_price or cs["entry_price"]
        close_condition_ticket(condition, exit_price, "MANUAL CLOSE")
        return True, "closed"
    except Exception as e:
        state.add_log(f"⚠️ [{condition}] Manual close failed: {e}")
        return False, str(e)


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (market data streaming)")
    while not _stop_flag.is_set():
        cfg = state.snapshot()["config"]
        try:
            symbol = cfg["symbol"]
            ex = get_exchange()
            state.add_log(f"Fetching {symbol} {cfg['timeframe']} candles from BingX…")
            bars = ex.fetch_ohlcv(symbol, timeframe=cfg["timeframe"], limit=500)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            close_arr = df["close"].values.astype(float)
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            live_price = float(close_arr[-1])
            state.connected = True
            print(f"[LOOP] PID={os.getpid()} - got {len(df)} candles, price={live_price}", flush=True)

            ind = compute_shared_indicator(df, cfg)
            idx = len(df) - 2  # last CLOSED candle
            last_bull_ob = ind["last_bull_ob"][idx]
            last_bear_ob = ind["last_bear_ob"][idx]

            indicator_payload = {
                "timestamps": df["timestamp"].tolist(),
                "bull": ind["near_bull_ob"].tolist(),
                "bear": ind["near_bear_ob"].tolist(),
                "last_bull_ob": None if np.isnan(last_bull_ob) else float(last_bull_ob),
                "last_bear_ob": None if np.isnan(last_bear_ob) else float(last_bear_ob),
            }
            ohlcv_payload = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
            state.update_market(ohlcv_payload, indicator_payload, live_price)

            try:
                bal = ex.fetch_balance()
                state.balance = float(bal.get("USDT", {}).get("total", 0.0))
            except Exception:
                pass

            cond_specs = [
                ("C1", cfg["c1_use_ema"], cfg["c1_ema_len"]),
                ("C2", cfg["c2_use_ema"], cfg["c2_ema_len"]),
                ("C3", cfg["c3_use_ema"], cfg["c3_ema_len"]),
            ]

            for condition, use_ema, ema_len in cond_specs:
                cs = state.cond_state[condition]

                # 1) เช็คว่าไม้ที่เปิดอยู่ของเงื่อนไขนี้ โดนTP/SL หรือยัง (บนแท่งที่ปิดล่าสุด)
                if cs["in_position"]:
                    hi = high_arr[idx]
                    lo = low_arr[idx]
                    if cs["pos_side"] == "LONG":
                        if hi >= cs["tp"]:
                            close_condition_ticket(condition, cs["tp"], "TP")
                        elif lo <= cs["sl"]:
                            close_condition_ticket(condition, cs["sl"], "SL")
                    else:
                        if lo <= cs["tp"]:
                            close_condition_ticket(condition, cs["tp"], "TP")
                        elif hi >= cs["sl"]:
                            close_condition_ticket(condition, cs["sl"], "SL")

                # 2) ถ้ายังไม่มีไม้ค้าง + บอทถูก RUN + ถึงวันเริ่มเทรดแล้ว -> เช็คสัญญาณใหม่
                cs = state.cond_state[condition]  # re-read after possible close above
                if state.running and not cs["in_position"] and _is_trading_started(cfg):
                    if condition == "C1":
                        raw_long, raw_short = condition1_signal(ind, close_arr, idx, cfg)
                    else:
                        raw_long, raw_short = condition_signal(ind, close_arr, idx, cfg, use_ema, ema_len)

                    dist_pct = cfg["min_dist_pct"] / 100.0
                    can_long = cs["last_long_entry"] is None or (abs(live_price - cs["last_long_entry"]) / cs["last_long_entry"] >= dist_pct)
                    can_short = cs["last_short_entry"] is None or (abs(live_price - cs["last_short_entry"]) / cs["last_short_entry"] >= dist_pct)

                    if raw_long and can_long:
                        fire_entry(condition, symbol, "LONG", live_price, cfg)
                    elif raw_short and can_short:
                        fire_entry(condition, symbol, "SHORT", live_price, cfg)

        except Exception as ex_err:
            state.connected = False
            state.add_log(f"⚠️ Loop error [{type(ex_err).__name__}]: {ex_err}")
            print(f"[LOOP] PID={os.getpid()} - ERROR [{type(ex_err).__name__}]: {ex_err}", flush=True)

        _stop_flag.wait(cfg.get("poll_seconds", 10))

    state.connected = False
    state.add_log("🔴 Engine stopped")


def init_engine():
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop_flag.clear()
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()


def set_running(on: bool):
    state.running = bool(on)
    state.add_log("🟢 RUN enabled - live signals will fire real orders" if on
                  else "⏸ RUN disabled - market data keeps refreshing, no orders will fire")


# ============================================================================
# 4) WEB DASHBOARD
# ============================================================================

app = Flask(__name__)
print(f"[BOOT] PID={os.getpid()} - Flask app module loading, about to init engine…", flush=True)
init_engine()
print(f"[BOOT] PID={os.getpid()} - init_engine() called", flush=True)

INDEX_HTML = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>3-Condition Confluence Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{ --bg:#0A0C12; --panel:#12151E; --panel-2:#171B27; --line:#232838; --text:#E7ECF3; --muted:#7C879C;
    --gold:#C9A24A; --gold-dim:#8A7130; --long:#3ED8A0; --short:#FF5C72; --c1:#3ED8A0; --c2:#4CC9F0; --c3:#D65DB1; --radius:10px; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;-webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  .app{display:grid;grid-template-columns:64px 320px 1fr;grid-template-rows:64px 1fr;height:100vh;}
  .brand{grid-column:1/3;grid-row:1;display:flex;align-items:center;gap:12px;padding:0 20px;border-bottom:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:14px;}
  .brand h1{font-size:15px;letter-spacing:.04em;margin:0;font-weight:600;}
  .brand .sub{color:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase;}
  .topbar{grid-column:3;grid-row:1;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--line);}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .dot.on{background:var(--long);box-shadow:0 0 8px var(--long);}
  .run-btn{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:9px 18px;border-radius:8px;font-family:inherit;font-weight:600;font-size:12px;letter-spacing:.04em;cursor:pointer;}
  .run-btn.active{background:var(--long);color:#062018;border-color:var(--long);}
  .run-btn.stopped{background:var(--short);color:#2a0509;border-color:var(--short);}
  .manual-group{display:flex;gap:6px;align-items:center;}
  .manual-btn{border:1px solid var(--line);background:var(--panel-2);color:var(--text);padding:6px 10px;border-radius:6px;font-family:inherit;font-weight:600;font-size:10.5px;cursor:pointer;}
  .manual-btn.long{color:var(--long);border-color:var(--long);}
  .manual-btn.short{color:var(--short);border-color:var(--short);}
  .close-btn{background:transparent;border:1px solid var(--short);color:var(--short);padding:3px 9px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;}
  .sidebar{grid-column:1;grid-row:2;border-right:1px solid var(--line);display:flex;flex-direction:column;align-items:center;padding-top:16px;gap:18px;color:var(--muted);}
  .sidebar .ic{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;}
  .sidebar .ic.active{background:var(--panel-2);color:var(--gold);}
  .control{grid-column:2;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:20px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .field{margin-bottom:10px;}
  .field label{display:block;font-size:11.5px;color:var(--muted);margin-bottom:5px;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .save-btn{width:100%;margin-top:14px;background:var(--gold);color:#211a06;border:none;padding:11px;border-radius:8px;font-weight:700;font-size:12.5px;letter-spacing:.03em;cursor:pointer;font-family:inherit;}
  .warn{margin-top:16px;padding:10px 12px;background:#241412;border:1px solid #4a2620;border-radius:8px;font-size:11px;color:#f0a89c;line-height:1.5;}
  .main{grid-column:3;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between;}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:280px;}
  .cond-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
  .cond-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;border-top:3px solid var(--c);}
  .cond-card.c1{--c:var(--c1);} .cond-card.c2{--c:var(--c2);} .cond-card.c3{--c:var(--c3);}
  .cond-card .title{font-weight:700;font-size:13px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;}
  .cond-card .row{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted);padding:3px 0;font-family:'IBM Plex Mono',monospace;}
  .cond-card .row b{color:var(--text);}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
  .metric-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;}
  .metric-card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .metric-card .val{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;margin-top:6px;}
  .val.pos{color:var(--long);} .val.neg{color:var(--short);}
  .split{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;}
  table{width:100%;border-collapse:collapse;font-size:11.5px;}
  th{text-align:left;color:var(--muted);font-weight:500;font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line);}
  td{padding:6px 8px;border-bottom:1px solid #191d29;font-family:'IBM Plex Mono',monospace;}
  .side-long{color:var(--long);} .side-short{color:var(--short);}
  .logbox{max-height:230px;overflow-y:auto;font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
  .logline{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #171a24;color:#B7C0D1;}
  .logline .t{color:var(--muted);}
  ::-webkit-scrollbar{width:8px;height:8px;} ::-webkit-scrollbar-thumb{background:#232838;border-radius:4px;}
  .badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;white-space:nowrap;}
  .badge-open{background:rgba(201,162,74,0.15);color:var(--gold);border:1px solid rgba(201,162,74,0.4);}
  .badge-win{background:rgba(62,216,160,0.15);color:var(--long);border:1px solid rgba(62,216,160,0.4);}
  .badge-loss{background:rgba(255,92,114,0.15);color:var(--short);border:1px solid rgba(255,92,114,0.4);}
  .tag{display:inline-block;padding:1px 6px;border-radius:5px;font-size:9.5px;font-weight:700;margin-right:4px;}
  .tag.c1{background:rgba(62,216,160,.18);color:var(--c1);} .tag.c2{background:rgba(76,201,240,.18);color:var(--c2);} .tag.c3{background:rgba(214,93,177,.18);color:var(--c3);}
  #toastContainer{position:fixed;top:76px;right:20px;z-index:80;display:flex;flex-direction:column;gap:10px;}
  .toast{min-width:250px;max-width:320px;background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:12px 16px;box-shadow:0 8px 24px rgba(0,0,0,0.5);animation:slideIn .25s ease-out, fadeOut .4s ease-in 5.6s forwards;}
  .toast.win{border-color:var(--long);} .toast.loss{border-color:var(--short);}
  .toast .ttitle{font-weight:700;font-size:13px;margin-bottom:4px;}
  .toast.win .ttitle{color:var(--long);} .toast.loss .ttitle{color:var(--short);}
  .toast .tbody{font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace;}
  @keyframes slideIn{from{transform:translateX(30px);opacity:0;} to{transform:translateX(0);opacity:1;}}
  @keyframes fadeOut{to{opacity:0; transform:translateY(-6px);}}
</style>
</head>
<body>
<div id="toastContainer"></div>
<div class="app">
  <div class="brand"><div class="mark">C3</div><div><h1>3-CONDITION CONFLUENCE ENGINE</h1><div class="sub" id="symbolLabel">LOADING…</div></div></div>
  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span></div>
    <div class="manual-group">
      <span class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></span>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>
  <div class="sidebar"><div class="ic active">◆</div><div class="ic">▤</div><div class="ic">◷</div><div class="ic">▥</div><div class="ic">⚙</div></div>

  <div class="control">
    <h2>Trading Setup</h2>
    <div class="field"><label>Symbol</label><input id="cfg_symbol" type="text" /></div>
    <div class="row2">
      <div class="field"><label>Timeframe</label>
        <select id="cfg_timeframe"><option value="1m">1m</option><option value="5m">5m</option><option value="15m">15m</option><option value="1h">1h</option></select>
      </div>
      <div class="field"><label>Leverage (x)</label><input id="cfg_leverage" type="number" step="1" /></div>
    </div>
    <div class="field"><label>วันเริ่มเทรด</label><input id="cfg_bot_start_date" type="date" /></div>

    <h2>Capital & Fee</h2>
    <div class="row2">
      <div class="field"><label>ทุนเริ่มต้น (USDT)</label><input id="cfg_initial_cap" type="number" step="0.01" /></div>
      <div class="field"><label>ไม้ต่อออเดอร์ (USDT)</label><input id="cfg_base_order_usdt" type="number" step="0.01" /></div>
    </div>
    <div class="field"><label>Fee %</label><input id="cfg_fee_pct" type="number" step="0.01" /></div>
    <div class="field"><label>ระยะห่างขั้นต่ำจากไม้เดิม (%)</label><input id="cfg_min_dist_pct" type="number" step="0.1" /></div>

    <h2>RSI Divergence (Shared)</h2>
    <div class="field"><label>ใช้ RSI Divergence</label><select id="cfg_use_rsi_div"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    <div class="row2">
      <div class="field"><label>RSI Length</label><input id="cfg_rsi_len" type="number" step="1" /></div>
      <div class="field"><label>Overbought</label><input id="cfg_rsi_ob" type="number" step="1" /></div>
    </div>
    <div class="field"><label>Oversold</label><input id="cfg_rsi_os" type="number" step="1" /></div>

    <h2>TP / SL (Shared)</h2>
    <div class="field"><label>ใช้ Auto TP/SL จาก Order Block</label><select id="cfg_use_auto_tp_sl"><option value="true">Auto</option><option value="false">Fix %</option></select></div>
    <div class="row2">
      <div class="field"><label>Fix TP %</label><input id="cfg_fix_tp_pct" type="number" step="0.1" /></div>
      <div class="field"><label>Fix SL %</label><input id="cfg_fix_sl_pct" type="number" step="0.1" /></div>
    </div>

    <h2>เงื่อนไข 1 (EMA Filter)</h2>
    <div class="row2">
      <div class="field"><label>เปิดใช้ EMA</label><select id="cfg_c1_use_ema"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>EMA Length</label><input id="cfg_c1_ema_len" type="number" step="1" /></div>
    </div>
    <div class="field"><label>เปิดใช้ Reversal จากระยะห่าง EMA</label><select id="cfg_c1_use_rev_dist"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    <div class="row2">
      <div class="field"><label>EMA วัดระยะห่าง (Length)</label><input id="cfg_c1_rev_ema_len" type="number" step="1" /></div>
      <div class="field"><label>ระยะห่าง % ที่ให้สลับฝั่ง</label><input id="cfg_c1_rev_pct" type="number" step="0.1" /></div>
    </div>
    <h2>เงื่อนไข 2 (EMA Filter)</h2>
    <div class="row2">
      <div class="field"><label>เปิดใช้ EMA</label><select id="cfg_c2_use_ema"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>EMA Length</label><input id="cfg_c2_ema_len" type="number" step="1" /></div>
    </div>
    <h2>เงื่อนไข 3 (EMA Filter)</h2>
    <div class="row2">
      <div class="field"><label>เปิดใช้ EMA</label><select id="cfg_c3_use_ema"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>EMA Length</label><input id="cfg_c3_ema_len" type="number" step="1" /></div>
    </div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS</button>
    <div class="warn">ทั้ง 3 เงื่อนไขเทรดสัญลักษณ์เดียวกัน — ถ้า 2 เงื่อนไขเปิดฝั่งเดียวกันพร้อมกัน BingX จะรวมเป็นโพซิชั่นเดียว (มี TP/SL จริงต่อไม้ แต่สถิติรายเงื่อนไขอาจคลาดเคลื่อนเล็กน้อยในกรณีนี้)</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart</span><span class="price mono" id="livePrice">--</span></div>
      <div id="priceChart"></div>
    </div>

    <div class="cond-grid">
      <div class="cond-card c1"><div class="title"><span>เงื่อนไข 1</span><span id="c1_status">--</span></div>
        <div class="row"><span>ออเดอร์</span><b id="c1_total">--</b></div>
        <div class="row"><span>ชนะ/แพ้</span><b id="c1_wl">--</b></div>
        <div class="row"><span>กำไรสุทธิ</span><b id="c1_net">--</b></div>
        <div class="manual-group" style="margin-top:8px"><button class="manual-btn long" data-c="C1" data-side="LONG">▲ LONG</button><button class="manual-btn short" data-c="C1" data-side="SHORT">▼ SHORT</button></div>
      </div>
      <div class="cond-card c2"><div class="title"><span>เงื่อนไข 2</span><span id="c2_status">--</span></div>
        <div class="row"><span>ออเดอร์</span><b id="c2_total">--</b></div>
        <div class="row"><span>ชนะ/แพ้</span><b id="c2_wl">--</b></div>
        <div class="row"><span>กำไรสุทธิ</span><b id="c2_net">--</b></div>
        <div class="manual-group" style="margin-top:8px"><button class="manual-btn long" data-c="C2" data-side="LONG">▲ LONG</button><button class="manual-btn short" data-c="C2" data-side="SHORT">▼ SHORT</button></div>
      </div>
      <div class="cond-card c3"><div class="title"><span>เงื่อนไข 3</span><span id="c3_status">--</span></div>
        <div class="row"><span>ออเดอร์</span><b id="c3_total">--</b></div>
        <div class="row"><span>ชนะ/แพ้</span><b id="c3_wl">--</b></div>
        <div class="row"><span>กำไรสุทธิ</span><b id="c3_net">--</b></div>
        <div class="manual-group" style="margin-top:8px"><button class="manual-btn long" data-c="C3" data-side="LONG">▲ LONG</button><button class="manual-btn short" data-c="C3" data-side="SHORT">▼ SHORT</button></div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">Win Rate (รวม)</div><div class="val" id="m_winrate">--</div></div>
      <div class="metric-card"><div class="lbl">Total Trades (รวม)</div><div class="val" id="m_total">--</div></div>
      <div class="metric-card"><div class="lbl">Net Profit (รวม)</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">Portfolio Balance</div><div class="val" id="m_balance">--</div></div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table><thead><tr><th></th><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>Status</th><th>Action</th></tr></thead><tbody id="ordersBody"></tbody></table>
      </div>
      <div class="panel"><div class="panel-title">Bot Activity Log</div><div class="logbox" id="logBody"></div></div>
    </div>
  </div>
</div>

<script>
const CFG_KEYS = ["symbol","timeframe","leverage","bot_start_date","initial_cap","base_order_usdt","fee_pct",
  "min_dist_pct","use_rsi_div","rsi_len","rsi_ob","rsi_os","use_auto_tp_sl","fix_tp_pct","fix_sl_pct",
  "c1_use_ema","c1_ema_len","c1_use_rev_dist","c1_rev_ema_len","c1_rev_pct",
  "c2_use_ema","c2_ema_len","c3_use_ema","c3_ema_len"];

const priceChart = LightweightCharts.createChart(document.getElementById('priceChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const candleSeries = priceChart.addCandlestickSeries({upColor:'#3ED8A0', downColor:'#FF5C72', borderVisible:false, wickUpColor:'#3ED8A0', wickDownColor:'#FF5C72'});

let cfgLoadedOnce = false;
let lastSeenTradeId = null;
let firstRender = true;

function statusBadge(t){
  if(t.status==='OPEN') return `<span class="badge badge-open">🔵 OPEN</span>`;
  if(t.status==='WIN')  return `<span class="badge badge-win">✅ TP/WIN</span>`;
  if(t.status==='LOSS') return `<span class="badge badge-loss">❌ SL/LOSS</span>`;
  return t.status||'';
}
function condTag(c){ return c ? `<span class="tag ${c.toLowerCase()}">${c}</span>` : ''; }

function showToast(t){
  const isWin = t.status==='WIN';
  const el = document.createElement('div');
  el.className = 'toast ' + (isWin?'win':'loss');
  el.innerHTML = `<div class="ttitle">${condTag(t.condition)} ${isWin?'🎯 TP HIT':'🛑 SL HIT'}</div><div class="tbody">${t.side} @ ${t.exit??''} · PnL ${(t.pnl>=0?'+':'')+t.pnl} USDT</div>`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(()=>el.remove(), 6200);
}
function maybeShowToast(trades){
  if(!trades || !trades.length) return;
  const newest = trades[0];
  if(newest.status!=='WIN' && newest.status!=='LOSS') return;
  if(firstRender){ lastSeenTradeId = newest.id; firstRender = false; return; }
  if(newest.id === lastSeenTradeId) return;
  lastSeenTradeId = newest.id;
  showToast(newest);
}

async function poll(){
  try{ const res = await fetch('/api/status'); render(await res.json()); }catch(e){ console.error(e); }
  setTimeout(poll, 4000);
}

function condCard(id, cs){
  document.getElementById(id+'_status').textContent = cs.in_position ? ('🟢 '+cs.pos_side) : '⚪ ว่าง';
  document.getElementById(id+'_total').textContent = cs.total_trades;
  document.getElementById(id+'_wl').textContent = cs.wins + ' / ' + cs.losses;
  const netEl = document.getElementById(id+'_net');
  netEl.textContent = (cs.net_profit>=0?'+':'') + cs.net_profit.toFixed(2) + ' USDT';
  netEl.style.color = cs.net_profit>=0 ? 'var(--long)' : 'var(--short)';
}

function render(data){
  document.getElementById('connDot').className = 'dot ' + (data.connected?'on':'');
  document.getElementById('connText').textContent = data.connected ? 'CONNECTED · BingX' : 'CONNECTING…';
  document.getElementById('symbolLabel').textContent = (data.config.symbol||'') + ' · ' + (data.config.timeframe||'');
  document.getElementById('livePrice').textContent = data.live_price ? ('$'+data.live_price) : '--';
  document.getElementById('balanceVal').textContent = '$' + (data.balance||0).toFixed(2);

  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running?'active':'stopped');

  condCard('c1', data.cond_state.C1); condCard('c2', data.cond_state.C2); condCard('c3', data.cond_state.C3);

  const cb = data.combined;
  document.getElementById('m_winrate').textContent = cb.win_rate + '%';
  document.getElementById('m_total').textContent = cb.total_trades;
  const pnlEl = document.getElementById('m_pnl');
  pnlEl.textContent = (cb.net_profit>=0?'+':'') + cb.net_profit + ' USDT';
  pnlEl.className = 'val ' + (cb.net_profit>=0?'pos':'neg');
  document.getElementById('m_balance').textContent = cb.portfolio_balance + ' USDT';

  if(data.ohlcv && data.ohlcv.length){
    candleSeries.setData(data.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]})));
    const markers = [];
    data.trades.forEach(t=>{
      const col = t.condition==='C1' ? '#3ED8A0' : t.condition==='C2' ? '#4CC9F0' : '#D65DB1';
      markers.push({time: Math.floor(t.entry_ms/1000), position: t.side==='LONG'?'belowBar':'aboveBar', color: col, shape: t.side==='LONG'?'arrowUp':'arrowDown', text: t.condition});
      if((t.status==='WIN'||t.status==='LOSS') && t.exit_ms){
        markers.push({time: Math.floor(t.exit_ms/1000), position: t.status==='WIN'?'aboveBar':'belowBar', color: t.status==='WIN'?'#3ED8A0':'#FF5C72', shape: t.status==='WIN'?'circle':'square', text: t.status==='WIN'?'TP':'SL'});
      }
    });
    markers.sort((a,b)=>a.time-b.time);
    candleSeries.setMarkers(markers);
  }

  const ordersBody = document.getElementById('ordersBody');
  ordersBody.innerHTML = data.trades.map(t=>`
    <tr>
      <td>${condTag(t.condition)}</td>
      <td>${t.time||''}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.type||''}</td>
      <td>${t.entry??''}</td><td>${t.tp??''}</td><td>${t.sl??''}</td>
      <td>${statusBadge(t)}</td>
      <td>${t.status==='OPEN' ? `<button class="close-btn" data-c="${t.condition}">CLOSE</button>` : ''}</td>
    </tr>`).join('');
  ordersBody.querySelectorAll('.close-btn').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      btn.disabled=true; btn.textContent='...';
      const res = await fetch('/api/close-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({condition: btn.dataset.c})});
      const j = await res.json();
      if(!j.ok) alert('Close failed: '+(j.message||j.error||'unknown error'));
    });
  });

  maybeShowToast(data.trades);

  document.getElementById('logBody').innerHTML = data.logs.slice().reverse().map(l=>`<div class="logline"><span class="t">${l.time}</span><span>${l.text}</span></div>`).join('');

  if(!cfgLoadedOnce){
    CFG_KEYS.forEach(k=>{ const el = document.getElementById('cfg_'+k); if(el && data.config[k]!==undefined) el.value = data.config[k]; });
    cfgLoadedOnce = true;
  }
}

document.querySelectorAll('.manual-btn[data-c]').forEach(btn=>{
  btn.addEventListener('click', async ()=>{
    const c = btn.dataset.c, side = btn.dataset.side;
    if(!confirm(`เปิดออเดอร์ ${side} ด้วยมือ (เงื่อนไข ${c}) ยิงจริงทันที แน่ใจไหม?`)) return;
    const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({condition:c, side})});
    const j = await res.json();
    if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: '+(j.message||j.error||'unknown error'));
  });
});

document.getElementById('runBtn').addEventListener('click', async ()=>{
  const willRun = document.getElementById('runBtn').textContent.includes('RUN');
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({running: willRun})});
});

document.getElementById('saveCfgBtn').addEventListener('click', async ()=>{
  const patch = {};
  CFG_KEYS.forEach(k=>{ const el = document.getElementById('cfg_'+k); if(el) patch[k]=el.value; });
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(patch)});
});

poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "connected": state.connected, "running": state.running})


@app.route("/api/status")
def api_status():
    return jsonify(state.snapshot())


@app.route("/api/config", methods=["POST"])
def api_config():
    patch = request.get_json(force=True) or {}
    numeric_keys = {"leverage", "initial_cap", "base_order_usdt", "fee_pct", "min_dist_pct",
                    "rsi_len", "rsi_ob", "rsi_os", "fix_tp_pct", "fix_sl_pct", "c1_ema_len", "c2_ema_len", "c3_ema_len",
                    "c1_rev_ema_len", "c1_rev_pct"}
    bool_keys = {"use_rsi_div", "use_auto_tp_sl", "c1_use_ema", "c2_use_ema", "c3_use_ema", "c1_use_rev_dist"}
    string_keys = {"symbol", "timeframe", "bot_start_date"}

    clean = {}
    for k, v in patch.items():
        if k in numeric_keys:
            try:
                clean[k] = float(v) if "." in str(v) else int(v)
            except (TypeError, ValueError):
                continue
        elif k in bool_keys:
            clean[k] = str(v).strip().lower() == "true"
        elif k in string_keys:
            clean[k] = str(v)
    state.update_config(clean)
    state.add_log(f"Config updated: {clean}")
    return jsonify(state.snapshot()["config"])


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    set_running(bool(payload.get("running")))
    return jsonify({"running": state.running})


@app.route("/api/manual-order", methods=["POST"])
def api_manual_order():
    payload = request.get_json(force=True) or {}
    condition = payload.get("condition")
    side = (payload.get("side") or "").upper()
    if condition not in CONDITIONS or side not in ("LONG", "SHORT"):
        return jsonify({"ok": False, "error": "invalid condition/side"}), 400
    ok, msg = open_trade_manual(condition, side)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/close-order", methods=["POST"])
def api_close_order():
    payload = request.get_json(force=True) or {}
    condition = payload.get("condition")
    if condition not in CONDITIONS:
        return jsonify({"ok": False, "error": "invalid condition"}), 400
    ok, msg = close_trade_manual(condition)
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
