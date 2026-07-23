"""
labusdt_zigzag_pattern_bot.py
------------------------------
Single-file drop-in bot for BingX Futures — trades the "ZigZag + Fibonacci +
Volume Profile Confirm" indicator's FINAL entry logic: pure chart-pattern
breakout (Double Top / Double Bottom / Head & Shoulders / Inverse H&S) built
on top of ZigZag pivots. Replaces the old RSI Momentum Divergence bot.

Same deploy pattern as before: one file, one process, PORT from the
environment, a background thread doing the work while a web server keeps
the service alive/health-checkable online.

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python labusdt_zigzag_pattern_bot.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT labusdt_zigzag_pattern_bot:app

⚠️ MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders
firing on the same signal.

STRATEGY NOTES
--------------
- ZigZag pivots are confirmed `zz_len` bars after they happen (lag is real,
  not a bug — same as the Pine Script version).
- A pattern (Double Top/Bottom, H&S/Inverse H&S) becomes "valid" once its
  pivot points line up within tolerance. The actual BUY/SELL signal only
  fires once price CLOSES beyond the pattern's neckline.
- Position size ("lot") compounds daily: today_lot = yesterday_lot +
  (daily_lot_pct% × current capital) − start_lot, exactly like the
  indicator. It is then hard-capped at max_lot_pct_of_capital% of current
  capital so one bad day can never risk too much of the pot.
- Only ONE thing decides entries: the pattern breakout. There is no
  Fibonacci-zone / Volume-profile / RSI-confirmation gate anymore (those
  were disabled in the final indicator version too) — kept out here to
  keep the bot's logic exactly matching what actually fires signals.
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
# 1) INDICATOR - Python port of the Pine Script "ZigZag + Pattern" logic
# ============================================================================

def find_pivot_highs_lows(high: np.ndarray, low: np.ndarray, left: int, right: int):
    """Pivot high/low at index j, confirmable once j+right <= n-1 (mirrors
    Pine's ta.pivothigh/ta.pivotlow(_, left, right))."""
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


def compute_zigzag(high: np.ndarray, low: np.ndarray, length: int, dev_pct: float):
    """Percentage-based zigzag: returns parallel lists of confirmed pivot
    price / bar-index / direction (1 = high, -1 = low). Same algorithm as
    the Pine Script version (extend the last pivot while price makes a new
    extreme in the same direction; only start a new pivot once price has
    moved by >= dev_pct against it)."""
    is_high, is_low = find_pivot_highs_lows(high, low, length, length)
    n = len(high)
    prices, bars, dirs = [], [], []

    def last_price():
        return prices[-1] if prices else None

    def last_dir():
        return dirs[-1] if dirs else 0

    for j in range(n):
        if is_high[j]:
            price = high[j]
            lp, ld = last_price(), last_dir()
            if lp is None:
                prices.append(price); bars.append(j); dirs.append(1)
            else:
                pct = abs(price - lp) / lp * 100.0
                if ld != 1 and pct >= dev_pct:
                    prices.append(price); bars.append(j); dirs.append(1)
                elif ld == 1 and price >= lp:
                    prices[-1] = price; bars[-1] = j
        if is_low[j]:
            price = low[j]
            lp, ld = last_price(), last_dir()
            if lp is None:
                prices.append(price); bars.append(j); dirs.append(-1)
            else:
                pct = abs(price - lp) / lp * 100.0
                if ld != -1 and pct >= dev_pct:
                    prices.append(price); bars.append(j); dirs.append(-1)
                elif ld == -1 and price <= lp:
                    prices[-1] = price; bars[-1] = j
    return prices, bars, dirs


def detect_patterns(prices, dirs, tol_pct: float, head_min_diff_pct: float,
                     use_double: bool, use_hs: bool):
    """Look at the last 3 (double top/bottom) and last 5 (H&S/inverse H&S)
    confirmed zigzag pivots. Returns a dict of neckline levels (or None)."""
    out = {"double_top": None, "double_bot": None, "hs_top": None, "hs_bot": None}
    n = len(dirs)

    if use_double and n >= 3:
        d2, d1, d0 = dirs[-3], dirs[-2], dirs[-1]
        if d2 == 1 and d1 == -1 and d0 == 1:
            p_top1, p_trough, p_top2 = prices[-3], prices[-2], prices[-1]
            if p_top1 and abs(p_top2 - p_top1) / p_top1 * 100.0 <= tol_pct:
                out["double_top"] = p_trough
        elif d2 == -1 and d1 == 1 and d0 == -1:
            p_bot1, p_peak, p_bot2 = prices[-3], prices[-2], prices[-1]
            if p_bot1 and abs(p_bot2 - p_bot1) / p_bot1 * 100.0 <= tol_pct:
                out["double_bot"] = p_peak

    if use_hs and n >= 5:
        d_oldest = dirs[-5]
        p_ls, p_a, p_head, p_b, p_rs = prices[-5], prices[-4], prices[-3], prices[-2], prices[-1]
        if d_oldest == 1 and p_ls and p_rs:
            shoulders_sim = abs(p_rs - p_ls) / p_ls * 100.0 <= tol_pct
            head_higher = ((p_head - p_ls) / p_ls * 100.0 >= head_min_diff_pct and
                            (p_head - p_rs) / p_rs * 100.0 >= head_min_diff_pct)
            if shoulders_sim and head_higher:
                out["hs_top"] = (p_a + p_b) / 2.0
        elif d_oldest == -1 and p_ls and p_rs:
            shoulders_sim = abs(p_rs - p_ls) / p_ls * 100.0 <= tol_pct
            head_lower = ((p_ls - p_head) / p_ls * 100.0 >= head_min_diff_pct and
                           (p_rs - p_head) / p_rs * 100.0 >= head_min_diff_pct)
            if shoulders_sim and head_lower:
                out["hs_bot"] = (p_a + p_b) / 2.0
    return out


def latest_signal(df: pd.DataFrame, cfg: dict):
    """Uses the last CLOSED candle (iloc[-2], not the still-forming one) to
    decide LONG / SHORT / HOLD. Returns (signal, pivot_bar_reference,
    neckline_price) — pivot_bar_reference is used to avoid re-firing the
    same pattern instance twice."""
    if len(df) < 10:
        return "HOLD", None, None
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    prices, bars, dirs = compute_zigzag(high, low, cfg["zz_len"], cfg["zz_dev_pct"])
    patt = detect_patterns(prices, dirs, cfg["pattern_tol_pct"], cfg["head_min_diff_pct"],
                            cfg["use_double_tb"], cfg["use_hs"])

    last_idx = len(df) - 2  # last fully closed candle
    last_close = close[last_idx]
    pivot_ref = bars[-1] if bars else None

    if patt["double_bot"] is not None and last_close > patt["double_bot"]:
        return "LONG", pivot_ref, patt["double_bot"]
    if patt["hs_bot"] is not None and last_close > patt["hs_bot"]:
        return "LONG", pivot_ref, patt["hs_bot"]
    if patt["double_top"] is not None and last_close < patt["double_top"]:
        return "SHORT", pivot_ref, patt["double_top"]
    if patt["hs_top"] is not None and last_close < patt["hs_top"]:
        return "SHORT", pivot_ref, patt["hs_top"]
    return "HOLD", pivot_ref, None


def calc_atr(df: pd.DataFrame, length: int) -> np.ndarray:
    """Simple rolling-mean ATR (approximation of Wilder's ATR — close
    enough for TP/SL sizing purposes)."""
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).rolling(max(length, 1)).mean().values
    return atr


def build_signal_markers(df: pd.DataFrame, cfg: dict):
    """Reconstructs historical BUY/SELL breakout bars for the chart, using
    the same rule as latest_signal() but walked forward pivot-by-pivot."""
    n = len(df)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    if n < 10:
        return bull, bear
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    prices, bars, dirs = compute_zigzag(high, low, cfg["zz_len"], cfg["zz_dev_pct"])

    fired_long, fired_short = set(), set()
    for k in range(3, len(dirs) + 1):
        patt = detect_patterns(prices[:k], dirs[:k], cfg["pattern_tol_pct"], cfg["head_min_diff_pct"],
                                cfg["use_double_tb"], cfg["use_hs"])
        start_bar = bars[k - 1] + 1
        neck_short = patt["double_top"] if patt["double_top"] is not None else patt["hs_top"]
        neck_long = patt["double_bot"] if patt["double_bot"] is not None else patt["hs_bot"]
        if neck_short is not None:
            for b in range(start_bar, n):
                if b in fired_short:
                    continue
                if close[b] < neck_short:
                    bear[b] = True
                    fired_short.add(b)
                    break
        if neck_long is not None:
            for b in range(start_bar, n):
                if b in fired_long:
                    continue
                if close[b] > neck_long:
                    bull[b] = True
                    fired_long.add(b)
                    break
    return bull, bear


def calc_sl_tp(cfg: dict, is_long: bool, entry_price: float, atr_val: float, structure_ref):
    """TP/SL modes: 'fixed' (% of entry), 'atr' (ATR multiples), or
    'structure' (SL at the pattern's neckline, TP at Risk:Reward × that
    distance). Falls back to 'fixed' if the chosen mode's inputs aren't
    available (e.g. no ATR yet, or no neckline for a manual order)."""
    mode = cfg.get("tpsl_mode", "fixed")

    if mode == "atr" and atr_val and atr_val > 0:
        if is_long:
            sl = entry_price - atr_val * cfg["atr_sl_mult"]
            tp = entry_price + atr_val * cfg["atr_tp_mult"]
        else:
            sl = entry_price + atr_val * cfg["atr_sl_mult"]
            tp = entry_price - atr_val * cfg["atr_tp_mult"]
        return sl, tp

    if mode == "structure" and structure_ref:
        sl = structure_ref
        risk = abs(entry_price - sl)
        tp = entry_price + risk * cfg["rr_ratio"] if is_long else entry_price - risk * cfg["rr_ratio"]
        return sl, tp

    # fixed % (default / fallback)
    if is_long:
        sl = entry_price * (1 - cfg["fixed_sl_pct"] / 100.0)
        tp = entry_price * (1 + cfg["fixed_tp_pct"] / 100.0)
    else:
        sl = entry_price * (1 + cfg["fixed_sl_pct"] / 100.0)
        tp = entry_price * (1 - cfg["fixed_tp_pct"] / 100.0)
    return sl, tp


# ============================================================================
# 2) SHARED STATE (config / capital / trades / logs), one lock, thread-safe
# ============================================================================

DEFAULT_CONFIG = {
    "symbol": "LAB/USDT:USDT",
    "timeframe": "5m",
    "leverage": 25,
    "max_tickets": 10,

    # --- Capital & Risk (ตามค่าในรูปที่ส่งมา) ---
    "starting_capital": 10.0,
    "start_lot": 1.0,
    "daily_lot_pct": 10.0,          # ไม้วันนี้ = ไม้เมื่อวาน + (%×ทุนล่าสุด) - ไม้เริ่มต้น
    "max_lot_pct_of_capital": 20.0, # NEW: ห้ามไม้เกิน % นี้ของทุนปัจจุบัน ไม่ว่าสูตรทบต้นจะให้เท่าไหร่

    # --- TP / SL ---
    "tpsl_mode": "fixed",           # "fixed" | "atr" | "structure"
    "fixed_sl_pct": 5.0,
    "fixed_tp_pct": 5.0,
    "atr_length": 2,
    "atr_sl_mult": 4.0,
    "atr_tp_mult": 5.0,
    "rr_ratio": 1.0,

    # --- ZigZag + Pattern ---
    "zz_len": 9,
    "zz_dev_pct": 3.2,
    "use_double_tb": True,
    "use_hs": True,
    "pattern_tol_pct": 2.7,
    "head_min_diff_pct": 1.7,

    "poll_seconds": 10,
}


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(DEFAULT_CONFIG)
        self.capital_state = {"capital": None, "today_lot_size": None, "last_lot_date": None}
        self.running = False
        self.connected = False
        self.trades = []
        self.logs = []
        self.ohlcv = []
        self.indicator = {"timestamps": [], "bull": [], "bear": []}
        self.live_price = None
        self.balance = 0.0
        self.active_long = 0
        self.active_short = 0
        self.last_atr = 0.0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self.config.update(data.get("config", {}))
                self.capital_state.update(data.get("capital_state", {}))
                self.trades = data.get("trades", [])
                self.logs = data.get("logs", [])
            except Exception:
                pass

    def _save(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"config": self.config, "capital_state": self.capital_state,
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

    def close_open_trades(self, side: str, exit_price: float):
        """Mark every still-OPEN trade on this side as closed, computing
        each trade's own PnL, and feed that PnL back into capital_state so
        tomorrow's lot size compounds correctly."""
        with self.lock:
            changed = False
            for t in self.trades:
                if t.get("status") == "OPEN" and t.get("side") == side:
                    entry = t.get("entry", 0) or 0
                    amount = t.get("contract_amount", 0) or 0
                    pnl = (exit_price - entry) * amount if side == "LONG" else (entry - exit_price) * amount
                    t["status"] = "WIN" if pnl >= 0 else "LOSS"
                    t["exit"] = exit_price
                    t["exit_ms"] = int(datetime.now().timestamp() * 1000)
                    t["pnl"] = round(pnl, 4)
                    cap = self.capital_state.get("capital")
                    if cap is None:
                        cap = self.config["starting_capital"]
                    self.capital_state["capital"] = round(cap + pnl, 6)
                    changed = True
            if changed:
                self._save()
            return changed

    def update_market(self, ohlcv, indicator, live_price):
        with self.lock:
            self.ohlcv = ohlcv
            self.indicator = indicator
            self.live_price = live_price

    def update_positions(self, active_long, active_short, balance):
        with self.lock:
            self.active_long = active_long
            self.active_short = active_short
            self.balance = balance

    def snapshot(self):
        with self.lock:
            total = len(self.trades)
            wins = sum(1 for t in self.trades if t.get("status") == "WIN")
            losses = sum(1 for t in self.trades if t.get("status") == "LOSS")
            closed = wins + losses
            win_rate = round((wins / closed) * 100, 1) if closed else 0.0
            pnl = round(sum(t.get("pnl", 0) or 0 for t in self.trades), 4)
            return {
                "config": dict(self.config),
                "capital_state": dict(self.capital_state),
                "running": self.running,
                "connected": self.connected,
                "live_price": self.live_price,
                "balance": self.balance,
                "active_long": self.active_long,
                "active_short": self.active_short,
                "metrics": {"total_trades": total, "wins": wins, "losses": losses,
                            "win_rate": win_rate, "pnl": pnl},
                "trades": self.trades[:50],
                "logs": self.logs[-100:],
                "ohlcv": self.ohlcv,
                "indicator": self.indicator,
            }


state = BotState()

# ============================================================================
# 3) CAPITAL / LOT-SIZE ENGINE  (ทบต้นรายวัน + ขีดจำกัด % ของทุน)
# ============================================================================

def get_today_lot_size(cfg: dict) -> float:
    """today_lot = yesterday_lot + (daily_lot_pct% × current capital) −
    start_lot, recomputed once per calendar day, then hard-capped at
    max_lot_pct_of_capital% of current capital."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    with state.lock:
        cs = state.capital_state
        if cs.get("capital") is None:
            cs["capital"] = cfg["starting_capital"]
        if cs.get("today_lot_size") is None:
            cs["today_lot_size"] = cfg["start_lot"]

        if cs.get("last_lot_date") != today_str:
            if cs.get("last_lot_date") is not None:  # skip recompute on the very first run
                new_lot = cs["today_lot_size"] + (cfg["daily_lot_pct"] / 100.0) * cs["capital"] - cfg["start_lot"]
                cs["today_lot_size"] = max(new_lot, 0.01)
            cs["last_lot_date"] = today_str
            state._save()

        cap_limit = cs["capital"] * cfg["max_lot_pct_of_capital"] / 100.0
        return round(min(cs["today_lot_size"], max(cap_limit, 0.01)), 6)


# ============================================================================
# 4) TRADING ENGINE - ccxt / BingX, TP/SL, ticket counting, margin
# ============================================================================

_exchange = None
_last_fired_pivot = {"LONG": None, "SHORT": None}
_stop_flag = threading.Event()
_thread = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bingx({
            "apiKey": BINGX_API_KEY,
            "secret": BINGX_SECRET_KEY,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": "swap"},
        })
    return _exchange


def fetch_trades_safe(symbol):
    ex = get_exchange()
    try:
        trades = ex.fetch_my_trades(symbol=symbol, limit=30)
        rows = []
        for t in trades:
            rows.append({
                "timestamp": t["timestamp"],
                "side": (t.get("info", {}).get("positionSide") or "").upper(),
                "price": float(t["price"]),
                "amount": float(t["amount"]),
                "trade_side": (t.get("side") or "").lower(),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def count_active_tickets(symbol, df_trades, max_tickets):
    ex = get_exchange()
    l_count, s_count = 0, 0
    try:
        positions = ex.fetch_positions()
        active = {}
        for pos in positions:
            size = float(pos.get("contracts", 0) or pos.get("size", 0) or 0)
            if size != 0 and symbol.split("/")[0] in (pos.get("symbol") or "").upper():
                side = (pos.get("side") or "").upper() or ("LONG" if size > 0 else "SHORT")
                active[side] = abs(size)

        if active and not df_trades.empty:
            for side, current_size in active.items():
                target_side = "buy" if side == "LONG" else "sell"
                filtered = df_trades[(df_trades["side"] == side) & (df_trades["trade_side"] == target_side)]
                acc = 0.0
                for _, row in filtered.iterrows():
                    if acc >= current_size:
                        break
                    amt = row["amount"]
                    if acc + amt > current_size:
                        amt = current_size - acc
                    if amt > 0.0001:
                        if side == "LONG" and l_count < max_tickets:
                            l_count += 1
                        elif side == "SHORT" and s_count < max_tickets:
                            s_count += 1
                    acc += row["amount"]
    except Exception:
        pass
    return l_count, s_count


def fire_execution_order(symbol, side, entry_price, margin_size, leverage, sl_price, tp_price, manual=False):
    ex = get_exchange()
    try:
        contract_amount = round((margin_size * leverage) / entry_price, 4)
        order_side = "buy" if side == "LONG" else "sell"
        tp_price = round(float(tp_price), 6)
        sl_price = round(float(sl_price), 6)

        try:
            ex.set_leverage(leverage, symbol)
        except Exception:
            pass

        ex.create_order(symbol=symbol, type="market", side=order_side,
                         amount=contract_amount, params={"positionSide": side})
        try:
            tp_sl_side = "sell" if side == "LONG" else "buy"
            ex.create_order(symbol=symbol, type="TAKE_PROFIT_MARKET", side=tp_sl_side,
                             amount=contract_amount,
                             params={"positionSide": side, "stopPrice": tp_price, "workingType": "MARK_PRICE"})
            ex.create_order(symbol=symbol, type="STOP_MARKET", side=tp_sl_side,
                             amount=contract_amount,
                             params={"positionSide": side, "stopPrice": sl_price, "workingType": "MARK_PRICE"})
        except Exception:
            pass

        tag = " [MANUAL]" if manual else ""
        state.add_trade({
            "id": uuid.uuid4().hex[:10],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_ms": int(datetime.now().timestamp() * 1000),
            "type": f"OPEN ({'BUY' if side == 'LONG' else 'SELL'}){tag}",
            "entry": entry_price, "tp": tp_price, "sl": sl_price,
            "contract_amount": contract_amount,
            "pnl": None, "status": "OPEN", "side": side,
        })
        state.add_log(f"ENTRY EXECUTED{tag}: {side} @ {entry_price} | TP {tp_price} / SL {sl_price} | lot {margin_size} USDT")
        return True, "order sent"
    except Exception as e:
        state.add_log(f"⚠️ ORDER FAILED ({side}): {e}")
        return False, str(e)


def open_trade_manual(side):
    cfg = state.snapshot()["config"]
    symbol = cfg["symbol"]
    live_price = state.live_price
    if not live_price:
        return False, "No live price yet - wait for market data to load first"
    lot = get_today_lot_size(cfg)
    sl, tp = calc_sl_tp(cfg, side == "LONG", live_price, state.last_atr, None)
    return fire_execution_order(symbol, side, live_price, lot, cfg["leverage"], sl, tp, manual=True)


def close_trade_manual(trade_id):
    trade = next((t for t in state.trades if t.get("id") == trade_id), None)
    if not trade:
        return False, "Trade not found"
    if trade.get("status") != "OPEN":
        return False, "Trade already closed"

    cfg = state.snapshot()["config"]
    symbol = cfg["symbol"]
    side = trade["side"]
    amount = trade.get("contract_amount")
    ex = get_exchange()
    close_side = "sell" if side == "LONG" else "buy"
    try:
        ex.create_order(symbol=symbol, type="market", side=close_side, amount=amount,
                         params={"positionSide": side, "reduceOnly": True})
        exit_price = state.live_price or trade["entry"]
        pnl = ((exit_price - trade["entry"]) * amount if side == "LONG"
               else (trade["entry"] - exit_price) * amount)
        with state.lock:
            trade["status"] = "WIN" if pnl >= 0 else "LOSS"
            trade["exit"] = exit_price
            trade["exit_ms"] = int(datetime.now().timestamp() * 1000)
            trade["pnl"] = round(pnl, 4)
            trade["type"] = trade["type"] + " [MANUAL CLOSE]"
            cap = state.capital_state.get("capital")
            if cap is None:
                cap = cfg["starting_capital"]
            state.capital_state["capital"] = round(cap + pnl, 6)
            state._save()
        state.add_log(f"Manual close: {side} ticket {trade_id} closed @ ~{exit_price}")
        return True, "closed"
    except Exception as e:
        state.add_log(f"⚠️ Manual close failed ({trade_id}): {e}")
        return False, str(e)


def _find_recent_close_price(df_trades, side, close_trade_side):
    if df_trades is None or df_trades.empty:
        return None
    closes = df_trades[(df_trades["side"] == side) & (df_trades["trade_side"] == close_trade_side)]
    if closes.empty:
        return None
    return float(closes.iloc[0]["price"])  # df_trades is sorted newest-first


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (market data streaming)")
    while not _stop_flag.is_set():
        cfg = state.snapshot()["config"]
        try:
            symbol = cfg["symbol"]
            ex = get_exchange()

            state.add_log(f"Fetching {symbol} {cfg['timeframe']} candles from BingX…")
            bars = ex.fetch_ohlcv(symbol, timeframe=cfg["timeframe"], limit=1000)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            live_price = float(df.iloc[-1]["close"])
            state.connected = True
            state.add_log(f"Got {len(df)} candles, price={live_price}")
            print(f"[LOOP] PID={os.getpid()} - got {len(df)} candles, price={live_price}", flush=True)

            atr_series = calc_atr(df, cfg["atr_length"])
            state.last_atr = float(atr_series[-2]) if len(atr_series) > 1 and not np.isnan(atr_series[-2]) else 0.0

            bull_markers, bear_markers = build_signal_markers(df, cfg)
            indicator_payload = {
                "timestamps": df["timestamp"].tolist(),
                "bull": bull_markers.tolist(),
                "bear": bear_markers.tolist(),
            }
            ohlcv_payload = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
            state.update_market(ohlcv_payload, indicator_payload, live_price)

            df_trades = fetch_trades_safe(symbol)
            active_l, active_s = count_active_tickets(symbol, df_trades, cfg["max_tickets"])

            has_open_long = any(t.get("status") == "OPEN" and t.get("side") == "LONG" for t in state.trades)
            has_open_short = any(t.get("status") == "OPEN" and t.get("side") == "SHORT" for t in state.trades)
            if active_l == 0 and has_open_long:
                exit_px = _find_recent_close_price(df_trades, "LONG", "sell") or live_price
                state.close_open_trades("LONG", exit_px)
                state.add_log(f"Position closed: LONG @ ~{exit_px} (TP or SL hit)")
            if active_s == 0 and has_open_short:
                exit_px = _find_recent_close_price(df_trades, "SHORT", "buy") or live_price
                state.close_open_trades("SHORT", exit_px)
                state.add_log(f"Position closed: SHORT @ ~{exit_px} (TP or SL hit)")

            try:
                bal = ex.fetch_balance()
                total_cap = float(bal.get("USDT", {}).get("total", 0.0))
            except Exception:
                total_cap = state.balance
            state.update_positions(active_l, active_s, total_cap)

            if state.running:
                sig, pivot_ref, neckline = latest_signal(df, cfg)
                lot = get_today_lot_size(cfg)
                if sig == "LONG" and active_l < cfg["max_tickets"] and pivot_ref is not None and pivot_ref != _last_fired_pivot["LONG"]:
                    sl, tp = calc_sl_tp(cfg, True, live_price, state.last_atr, neckline)
                    fire_execution_order(symbol, "LONG", live_price, lot, cfg["leverage"], sl, tp)
                    _last_fired_pivot["LONG"] = pivot_ref
                elif sig == "SHORT" and active_s < cfg["max_tickets"] and pivot_ref is not None and pivot_ref != _last_fired_pivot["SHORT"]:
                    sl, tp = calc_sl_tp(cfg, False, live_price, state.last_atr, neckline)
                    fire_execution_order(symbol, "SHORT", live_price, lot, cfg["leverage"], sl, tp)
                    _last_fired_pivot["SHORT"] = pivot_ref
                else:
                    state.add_log(f"SIGNAL DETECTED: {sig} (no new confirmed entry)")

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
# 5) WEB DASHBOARD - Flask app + embedded HTML (no separate template file)
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
<title>ZigZag Pattern Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{
    --bg:#0A0C12; --panel:#12151E; --panel-2:#171B27; --line:#232838;
    --text:#E7ECF3; --muted:#7C879C; --gold:#C9A24A; --gold-dim:#8A7130;
    --long:#3ED8A0; --short:#FF5C72; --radius:10px;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
    font-family:'Space Grotesk',sans-serif; -webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  a{color:inherit;}

  .app{display:grid;grid-template-columns:64px 320px 1fr;grid-template-rows:64px 1fr;height:100vh;}
  .brand{grid-column:1/3;grid-row:1;display:flex;align-items:center;gap:12px;
    padding:0 20px;border-bottom:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));
    display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:14px;}
  .brand h1{font-size:15px;letter-spacing:.04em;margin:0;font-weight:600;}
  .brand .sub{color:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase;}

  .topbar{grid-column:3;grid-row:1;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--line);}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .dot.on{background:var(--long);box-shadow:0 0 8px var(--long);}
  .run-switch{display:flex;align-items:center;gap:10px;}
  .run-btn{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:9px 18px;border-radius:8px;
    font-family:inherit;font-weight:600;font-size:12px;letter-spacing:.04em;cursor:pointer;transition:.15s;}
  .run-btn.active{background:var(--long);color:#062018;border-color:var(--long);}
  .run-btn.stopped{background:var(--short);color:#2a0509;border-color:var(--short);}
  .manual-btn{border:1px solid var(--line);background:var(--panel-2);color:var(--text);padding:9px 14px;border-radius:8px;
    font-family:inherit;font-weight:600;font-size:12px;letter-spacing:.03em;cursor:pointer;}
  .manual-btn.long{color:var(--long);border-color:var(--long);}
  .manual-btn.short{color:var(--short);border-color:var(--short);}
  .close-btn{background:transparent;border:1px solid var(--short);color:var(--short);padding:3px 9px;border-radius:6px;
    font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;}
  .close-btn:hover{background:var(--short);color:#2a0509;}

  .sidebar{grid-column:1;grid-row:2;border-right:1px solid var(--line);display:flex;flex-direction:column;
    align-items:center;padding-top:16px;gap:18px;color:var(--muted);}
  .sidebar .ic{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;}
  .sidebar .ic.active{background:var(--panel-2);color:var(--gold);}

  .control{grid-column:2;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:20px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .field{margin-bottom:10px;}
  .field label{display:block;font-size:11.5px;color:var(--muted);margin-bottom:5px;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--text);
    padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
  .field input:focus, .field select:focus{outline:none;border-color:var(--gold-dim);}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .save-btn{width:100%;margin-top:14px;background:var(--gold);color:#211a06;border:none;padding:11px;border-radius:8px;
    font-weight:700;font-size:12.5px;letter-spacing:.03em;cursor:pointer;font-family:inherit;}
  .warn{margin-top:16px;padding:10px 12px;background:#241412;border:1px solid #4a2620;border-radius:8px;
    font-size:11px;color:#f0a89c;line-height:1.5;}

  .main{grid-column:3;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;
    display:flex;justify-content:space-between;}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:280px;}
  .chart-wrap{position:relative;}
  #priceTooltip{position:absolute;top:8px;left:8px;background:rgba(18,21,30,0.95);border:1px solid var(--line);
    border-radius:8px;padding:10px 12px;font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.7;
    pointer-events:none;display:none;z-index:5;min-width:180px;box-shadow:0 4px 14px rgba(0,0,0,0.4);}
  #priceTooltip .row{display:flex;justify-content:space-between;gap:16px;}
  #priceTooltip .lbl{color:var(--muted);}
  #priceTooltip .pos{color:var(--long);} #priceTooltip .neg{color:var(--short);}

  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
  .metric-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;}
  .metric-card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .metric-card .val{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;margin-top:6px;}
  .val.pos{color:var(--long);} .val.neg{color:var(--short);}

  .split{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;}
  table{width:100%;border-collapse:collapse;font-size:12px;}
  th{text-align:left;color:var(--muted);font-weight:500;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
    padding:6px 8px;border-bottom:1px solid var(--line);}
  td{padding:7px 8px;border-bottom:1px solid #191d29;font-family:'IBM Plex Mono',monospace;}
  .side-long{color:var(--long);} .side-short{color:var(--short);}
  .logbox{max-height:230px;overflow-y:auto;font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
  .logline{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #171a24;color:#B7C0D1;}
  .logline .t{color:var(--muted);}
  ::-webkit-scrollbar{width:8px;height:8px;}
  ::-webkit-scrollbar-thumb{background:#232838;border-radius:4px;}

  /* --- TP/SL badges + toast --- */
  .badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;
    font-size:10.5px;font-weight:700;letter-spacing:.02em;white-space:nowrap;}
  .badge-open{background:rgba(201,162,74,0.15);color:var(--gold);border:1px solid rgba(201,162,74,0.4);}
  .badge-win{background:rgba(62,216,160,0.15);color:var(--long);border:1px solid rgba(62,216,160,0.4);}
  .badge-loss{background:rgba(255,92,114,0.15);color:var(--short);border:1px solid rgba(255,92,114,0.4);}
  #toastContainer{position:fixed;top:76px;right:20px;z-index:80;display:flex;flex-direction:column;gap:10px;}
  .toast{min-width:250px;max-width:320px;background:var(--panel-2);border:1px solid var(--line);border-radius:10px;
    padding:12px 16px;box-shadow:0 8px 24px rgba(0,0,0,0.5);
    animation:slideIn .25s ease-out, fadeOut .4s ease-in 5.6s forwards;}
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
  <div class="brand">
    <div class="mark">Z</div>
    <div>
      <h1>ZIGZAG PATTERN ENGINE</h1>
      <div class="sub" id="symbolLabel">LOADING…</div>
    </div>
  </div>

  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span></div>
    <div class="run-switch">
      <div class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></div>
      <button class="manual-btn long" id="manualLongBtn">▲ LONG</button>
      <button class="manual-btn short" id="manualShortBtn">▼ SHORT</button>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>

  <div class="sidebar">
    <div class="ic active">◆</div><div class="ic">▤</div><div class="ic">◷</div><div class="ic">▥</div><div class="ic">⚙</div>
  </div>

  <div class="control">
    <h2>Trading Setup</h2>
    <div class="field"><label>Symbol</label><input id="cfg_symbol" type="text" /></div>
    <div class="row2">
      <div class="field"><label>Timeframe</label>
        <select id="cfg_timeframe">
          <option value="1m">1m</option><option value="5m">5m</option>
          <option value="15m">15m</option><option value="1h">1h</option>
        </select>
      </div>
      <div class="field"><label>Leverage (x)</label><input id="cfg_leverage" type="number" step="1" /></div>
    </div>
    <div class="field"><label>Max Tickets / Side</label><input id="cfg_max_tickets" type="number" step="1" /></div>

    <h2>Capital & Risk</h2>
    <div class="row2">
      <div class="field"><label>ทุนรวมเริ่มต้น (USDT)</label><input id="cfg_starting_capital" type="number" step="0.01" /></div>
      <div class="field"><label>ไม้เริ่มต้น (USDT)</label><input id="cfg_start_lot" type="number" step="0.01" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>% เพิ่มไม้ต่อวัน</label><input id="cfg_daily_lot_pct" type="number" step="0.1" /></div>
      <div class="field"><label>ไม้สูงสุด (% ของทุน)</label><input id="cfg_max_lot_pct_of_capital" type="number" step="0.1" /></div>
    </div>

    <h2>TP / SL</h2>
    <div class="field"><label>TP/SL Mode</label>
      <select id="cfg_tpsl_mode">
        <option value="fixed">Fixed %</option>
        <option value="atr">Auto (ATR)</option>
        <option value="structure">Auto (Structure/Neckline)</option>
      </select>
    </div>
    <div class="row2">
      <div class="field"><label>Fixed SL %</label><input id="cfg_fixed_sl_pct" type="number" step="0.1" /></div>
      <div class="field"><label>Fixed TP %</label><input id="cfg_fixed_tp_pct" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>ATR Length</label><input id="cfg_atr_length" type="number" step="1" /></div>
      <div class="field"><label>RR Ratio (Structure)</label><input id="cfg_rr_ratio" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>ATR SL Mult</label><input id="cfg_atr_sl_mult" type="number" step="0.1" /></div>
      <div class="field"><label>ATR TP Mult</label><input id="cfg_atr_tp_mult" type="number" step="0.1" /></div>
    </div>

    <h2>ZigZag + Pattern</h2>
    <div class="row2">
      <div class="field"><label>ZigZag Length</label><input id="cfg_zz_len" type="number" step="1" /></div>
      <div class="field"><label>ZigZag Dev %</label><input id="cfg_zz_dev_pct" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Pattern Tolerance %</label><input id="cfg_pattern_tol_pct" type="number" step="0.1" /></div>
      <div class="field"><label>Head Min Diff %</label><input id="cfg_head_min_diff_pct" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Double Top/Bottom</label>
        <select id="cfg_use_double_tb"><option value="true">เปิด</option><option value="false">ปิด</option></select>
      </div>
      <div class="field"><label>Head & Shoulders</label>
        <select id="cfg_use_hs"><option value="true">เปิด</option><option value="false">ปิด</option></select>
      </div>
    </div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS</button>
    <div class="warn">สัญญาณยืนยันช้ากว่ากราฟที่เห็นเสมอ (ZigZag Length) และไม้จะไม่มีวันเกิน "ไม้สูงสุด (% ของทุน)" ไม่ว่าสูตรทบต้นจะให้เท่าไหร่ ทดสอบด้วยทุนน้อยก่อนเพิ่ม Leverage</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart</span><span class="price mono" id="livePrice">--</span></div>
      <div class="chart-wrap"><div id="priceChart"></div><div id="priceTooltip"></div></div>
    </div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">Win Rate</div><div class="val" id="m_winrate">--</div></div>
      <div class="metric-card"><div class="lbl">Total Trades</div><div class="val" id="m_total">--</div></div>
      <div class="metric-card"><div class="lbl">Profit / Loss</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">ทุนปัจจุบัน / ไม้วันนี้</div><div class="val" id="m_capital" style="font-size:15px">--</div></div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table>
          <thead><tr><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>Status</th><th>Action</th></tr></thead>
          <tbody id="ordersBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <div class="panel-title">Bot Activity Log</div>
        <div class="logbox" id="logBody"></div>
      </div>
    </div>
  </div>
</div>

<script>
const CFG_KEYS = ["symbol","timeframe","leverage","max_tickets","starting_capital","start_lot",
  "daily_lot_pct","max_lot_pct_of_capital","tpsl_mode","fixed_sl_pct","fixed_tp_pct","atr_length",
  "atr_sl_mult","atr_tp_mult","rr_ratio","zz_len","zz_dev_pct","pattern_tol_pct","head_min_diff_pct",
  "use_double_tb","use_hs"];

const priceChart = LightweightCharts.createChart(document.getElementById('priceChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const candleSeries = priceChart.addCandlestickSeries({
  upColor:'#3ED8A0', downColor:'#FF5C72', borderVisible:false,
  wickUpColor:'#3ED8A0', wickDownColor:'#FF5C72',
});

const tooltipEl = document.getElementById('priceTooltip');
const chartWrapEl = document.querySelector('.chart-wrap');
function fmtPct(v){ return (v>=0?'+':'')+v.toFixed(2)+'%'; }
function pctClass(v){ return v>=0 ? 'pos' : 'neg'; }

priceChart.subscribeCrosshairMove(param=>{
  if(!param || !param.time || !param.seriesData || !param.seriesData.get(candleSeries)){
    tooltipEl.style.display = 'none'; return;
  }
  const bar = param.seriesData.get(candleSeries);
  const {open, high, low, close} = bar;
  const change = close - open;
  const changePct = open ? (change/open*100) : 0;
  tooltipEl.innerHTML = `
    <div class="row"><span class="lbl">Open</span><span>${open}</span></div>
    <div class="row"><span class="lbl">High</span><span>${high}</span></div>
    <div class="row"><span class="lbl">Low</span><span>${low}</span></div>
    <div class="row"><span class="lbl">Close</span><span>${close}</span></div>
    <div class="row"><span class="lbl">Change</span><span class="${pctClass(change)}">${change>=0?'+':''}${change.toFixed(6)} (${fmtPct(changePct)})</span></div>`;
  tooltipEl.style.display = 'block';
  const wrapRect = chartWrapEl.getBoundingClientRect();
  let left = param.point.x + 16;
  if(left + 190 > wrapRect.width) left = param.point.x - 190 - 10;
  tooltipEl.style.left = Math.max(4, left) + 'px';
  tooltipEl.style.top = '8px';
});

let cfgLoadedOnce = false;
let lastSeenTradeId = null;
let firstRender = true;

function statusBadge(t){
  if(t.status === 'OPEN') return `<span class="badge badge-open">🔵 OPEN</span>`;
  if(t.status === 'WIN')  return `<span class="badge badge-win">✅ TP / WIN</span>`;
  if(t.status === 'LOSS') return `<span class="badge badge-loss">❌ SL / LOSS</span>`;
  return t.status || '';
}

function showToast(t){
  const isWin = t.status === 'WIN';
  const isManual = (t.type||'').includes('MANUAL');
  const title = isManual ? (isWin? '✋ ปิดไม้เอง (กำไร)':'✋ ปิดไม้เอง (ขาดทุน)') : (isWin? '🎯 TP HIT':'🛑 SL HIT');
  const el = document.createElement('div');
  el.className = 'toast ' + (isWin?'win':'loss');
  el.innerHTML = `<div class="ttitle">${title}</div><div class="tbody">${t.side} @ ${t.exit ?? ''} · PnL ${(t.pnl>=0?'+':'')+t.pnl} USDT</div>`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(()=> el.remove(), 6200);
}

function maybeShowToast(trades){
  if(!trades || !trades.length) return;
  const newest = trades[0];
  if(newest.status !== 'WIN' && newest.status !== 'LOSS'){ return; }
  if(firstRender){ lastSeenTradeId = newest.id; firstRender = false; return; }
  if(newest.id === lastSeenTradeId) return;
  lastSeenTradeId = newest.id;
  showToast(newest);
}

async function poll(){
  try{
    const res = await fetch('/api/status');
    const data = await res.json();
    render(data);
  }catch(e){ console.error(e); }
  setTimeout(poll, 4000);
}

function render(data){
  document.getElementById('connDot').className = 'dot ' + (data.connected ? 'on' : '');
  document.getElementById('connText').textContent = data.connected ? 'CONNECTED · BingX' : 'CONNECTING…';
  document.getElementById('symbolLabel').textContent = (data.config.symbol || '') + ' · ' + (data.config.timeframe || '');
  document.getElementById('livePrice').textContent = data.live_price ? ('$'+data.live_price) : '--';

  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running ? 'active' : 'stopped');

  const m = data.metrics;
  document.getElementById('m_winrate').textContent = m.win_rate + '%';
  document.getElementById('m_total').textContent = m.total_trades;
  const pnlEl = document.getElementById('m_pnl');
  pnlEl.textContent = (m.pnl>=0?'+':'') + m.pnl + ' USDT';
  pnlEl.className = 'val ' + (m.pnl>=0?'pos':'neg');

  const cs = data.capital_state || {};
  document.getElementById('m_capital').textContent =
    (cs.capital!==undefined && cs.capital!==null ? cs.capital.toFixed(2) : '--') + ' USDT / ' +
    (cs.today_lot_size!==undefined && cs.today_lot_size!==null ? cs.today_lot_size.toFixed(2) : '--') + ' USDT · ' +
    (data.balance ? '$'+data.balance.toFixed(2)+' บนเอ็กซ์เชนจ์' : '');

  if(data.ohlcv && data.ohlcv.length){
    const candles = data.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]}));
    candleSeries.setData(candles);

    const markers = [];
    data.indicator.timestamps.forEach((ts,i)=>{
      if(data.indicator.bull[i]) markers.push({time:Math.floor(ts/1000), position:'belowBar', color:'#3ED8A0', shape:'arrowUp', text:'BUY'});
      if(data.indicator.bear[i]) markers.push({time:Math.floor(ts/1000), position:'aboveBar', color:'#FF5C72', shape:'arrowDown', text:'SELL'});
    });
    data.trades.forEach(t=>{
      if((t.status==='WIN'||t.status==='LOSS') && t.exit_ms){
        markers.push({
          time: Math.floor(t.exit_ms/1000),
          position: t.status==='WIN' ? 'aboveBar' : 'belowBar',
          color: t.status==='WIN' ? '#3ED8A0' : '#FF5C72',
          shape: t.status==='WIN' ? 'circle' : 'square',
          text: t.status==='WIN' ? 'TP' : 'SL',
        });
      }
    });
    markers.sort((a,b)=>a.time-b.time);
    candleSeries.setMarkers(markers);
  }

  const ordersBody = document.getElementById('ordersBody');
  ordersBody.innerHTML = data.trades.map(t=>`
    <tr>
      <td>${t.time||''}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.type||''}</td>
      <td>${t.entry??''}</td>
      <td>${t.tp??''}</td>
      <td>${t.sl??''}</td>
      <td>${statusBadge(t)}</td>
      <td>${t.status==='OPEN' && t.id ? `<button class="close-btn" data-id="${t.id}">CLOSE</button>` : ''}</td>
    </tr>`).join('');
  ordersBody.querySelectorAll('.close-btn').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      btn.disabled = true; btn.textContent = '...';
      const res = await fetch('/api/close-order', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id: btn.dataset.id})});
      const j = await res.json();
      if(!j.ok) alert('Close failed: ' + (j.message||j.error||'unknown error'));
    });
  });

  maybeShowToast(data.trades);

  const logBody = document.getElementById('logBody');
  logBody.innerHTML = data.logs.slice().reverse().map(l=>`<div class="logline"><span class="t">${l.time}</span><span>${l.text}</span></div>`).join('');

  if(!cfgLoadedOnce){
    CFG_KEYS.forEach(k=>{
      const el = document.getElementById('cfg_'+k);
      if(el && data.config[k]!==undefined) el.value = data.config[k];
    });
    cfgLoadedOnce = true;
  }
}

document.getElementById('manualLongBtn').addEventListener('click', async ()=>{
  if(!confirm('เปิดออเดอร์ LONG ด้วยมือ ยิงจริงทันทีตาม config ปัจจุบัน แน่ใจไหม?')) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({side:'LONG'})});
  const j = await res.json();
  if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: ' + (j.message||j.error||'unknown error'));
});

document.getElementById('manualShortBtn').addEventListener('click', async ()=>{
  if(!confirm('เปิดออเดอร์ SHORT ด้วยมือ ยิงจริงทันทีตาม config ปัจจุบัน แน่ใจไหม?')) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({side:'SHORT'})});
  const j = await res.json();
  if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: ' + (j.message||j.error||'unknown error'));
});

document.getElementById('runBtn').addEventListener('click', async ()=>{
  const willRun = document.getElementById('runBtn').textContent.includes('RUN');
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({running:willRun})});
});

document.getElementById('saveCfgBtn').addEventListener('click', async ()=>{
  const patch = {};
  CFG_KEYS.forEach(k=>{
    const el = document.getElementById('cfg_'+k);
    if(el) patch[k] = el.value;
  });
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(patch)});
});

poll();
</script>
</body>
</html>
"""


@app.route("/api/test-connection")
def api_test_connection():
    import time as _time
    result = {"symbol": DEFAULT_CONFIG["symbol"]}
    try:
        test_ex = ccxt.bingx({
            "apiKey": BINGX_API_KEY, "secret": BINGX_SECRET_KEY,
            "enableRateLimit": True, "timeout": 8000, "options": {"defaultType": "swap"},
        })
        t0 = _time.time()
        bars = test_ex.fetch_ohlcv(DEFAULT_CONFIG["symbol"], timeframe="1m", limit=5)
        result["ohlcv_ok"] = True
        result["ohlcv_seconds"] = round(_time.time() - t0, 2)
        result["last_candle"] = bars[-1] if bars else None
    except Exception as e:
        result["ohlcv_ok"] = False
        result["ohlcv_error_type"] = type(e).__name__
        result["ohlcv_error"] = str(e)

    try:
        t0 = _time.time()
        bal = test_ex.fetch_balance()
        result["balance_ok"] = True
        result["balance_seconds"] = round(_time.time() - t0, 2)
        result["usdt_total"] = bal.get("USDT", {}).get("total")
    except Exception as e:
        result["balance_ok"] = False
        result["balance_error_type"] = type(e).__name__
        result["balance_error"] = str(e)

    result["api_key_set"] = bool(BINGX_API_KEY)
    result["secret_set"] = bool(BINGX_SECRET_KEY)
    return jsonify(result)


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
    numeric_keys = {
        "leverage", "max_tickets", "starting_capital", "start_lot", "daily_lot_pct",
        "max_lot_pct_of_capital", "fixed_sl_pct", "fixed_tp_pct", "atr_length", "atr_sl_mult",
        "atr_tp_mult", "rr_ratio", "zz_len", "zz_dev_pct", "pattern_tol_pct", "head_min_diff_pct",
        "poll_seconds",
    }
    bool_keys = {"use_double_tb", "use_hs"}
    string_keys = {"symbol", "timeframe", "tpsl_mode"}

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
    side = (payload.get("side") or "").upper()
    if side not in ("LONG", "SHORT"):
        return jsonify({"ok": False, "error": "side must be LONG or SHORT"}), 400
    ok, msg = open_trade_manual(side)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/close-order", methods=["POST"])
def api_close_order():
    payload = request.get_json(force=True) or {}
    trade_id = payload.get("id")
    if not trade_id:
        return jsonify({"ok": False, "error": "id required"}), 400
    ok, msg = close_trade_manual(trade_id)
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
