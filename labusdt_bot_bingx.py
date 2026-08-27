"""
whale_hunter_dual_tf_bot.py
------------------------
Single-file drop-in bot for BingX Futures — trades the
"Whale Hunter V10 - Real Fee & Growth (No Repaint)" Pine Script logic,
run as TWO FULLY INDEPENDENT STRATEGY INSTANCES in parallel: one on the
3-minute timeframe and one on the 5-minute timeframe.

WHALE HUNTER ENTRY LOGIC (per timeframe, identical rule set, applied
independently to that timeframe's own candles):
  - Volume spike:  volume > SMA(volume, vol_ma_len) * vol_mult
  - CCI (on CLOSE, not typical price): cci_len period
  - Trend filter:  EMA(close, ema_len)
  - Long  = volume spike AND cci > ob_level AND close > ema AND
            cci crosses OVER ob_level
  - Short = volume spike AND cci < os_level AND close < ema AND
            cci crosses UNDER os_level
  - NO-REPAINT: exactly like the Pine script's `nz(sig_raw[1], false)`,
    the raw condition is evaluated on a fully closed candle, then the
    bot only acts on it one candle later (once that raw-signal candle
    itself is confirmed closed) — see compute_whale_indicators() /
    the `raw_long[idx-1]` lookups in the engine loop.

TWO INDEPENDENT TIMEFRAMES ("3m" and "5m"):
  - Each timeframe has its OWN enable switch, its OWN indicator
    parameters (CCI length, OB/OS level, volume MA length/multiplier,
    EMA length), its OWN margin-per-order, its OWN TP % / SL %, and its
    OWN max-simultaneous-tickets-per-side — completely separate from
    the other timeframe. A 3m ticket's TP/SL/exit is judged purely
    against 3m candles; a 5m ticket's TP/SL/exit is judged purely
    against 5m candles. Ticket slots (max_trades_per_side) are also
    counted separately per timeframe, so the 3m strategy filling its
    quota never blocks the 5m strategy from opening, and vice versa.
  - Fee %, Leverage, initial capital, and the kill switch are shared
    account-level settings (there is only one real BingX account/
    position), everything else above is per-timeframe.

⚠️ REAL-EXCHANGE CAVEAT for Live mode: BingX only knows "symbol +
positionSide" — it has NO concept of "ticket" or "timeframe". If both
the 3m strategy and the 5m strategy are simultaneously holding, say,
LONG tickets on the same symbol, BingX merges ALL of them (3m + 5m +
any multi-trade stacking within a timeframe) into ONE real position.
Each ticket still gets its own real TP/SL bracket order sized to its
own contract amount for real money-management protection, but the
dashboard's per-ticket / per-timeframe win-loss bookkeeping is done in
SOFTWARE (checking each ticket's own TP/SL against its OWN timeframe's
latest closed candle), exactly like the Pine script does with
high/low. Attribution between simultaneously-open tickets (whether
same-TF stacked or cross-TF) can be imprecise at the edges; the money-
management (bracket orders) is still real and correct either way.

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python whale_hunter_dual_tf_bot.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT whale_hunter_dual_tf_bot:app

⚠️ MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders
firing on the same signal.

DEFAULT PARAMETERS
---------------------------------------------------------------
  ทุนเริ่มต้น (USD): 10          Leverage: 20x
  ค่าธรรมเนียม: 0.04% ของมูลค่าสัญญาจริง ต่อฝั่ง
  หยุดเปิดไม้ใหม่เมื่อพอร์ตติดลบ (Equity <= 0): เปิด
  โหมด: Paper (ไม่ยิงออเดอร์จริง) ตั้งต้น

  ทั้ง TF 3 นาที และ TF 5 นาที (ค่าเริ่มต้นเหมือนกัน แต่ปรับแยกอิสระได้):
    เปิดใช้งาน: เปิด        มูลค่ามาร์จิ้นต่อไม้: $1
    เปิดใช้งานการเปิดไม้ซ้อน: เปิด   จำนวนไม้เปิดพร้อมกันสูงสุดต่อฝั่ง: 3
    CCI Length: 20   Overbought: 100   Oversold: -100
    Volume MA Length: 20   Volume Multiplier: 2.0x
    EMA Length: 200
    Take Profit: 3%   Stop Loss: 5%
    อนุญาตเปิด Long / Short: เปิดทั้งคู่
    ถ้า TP และ SL ถูกแตะในแท่งเดียวกัน ให้นับ SL ก่อน: เปิด
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

TF_KEYS = ["3m", "5m"]

# ============================================================================
# 1) INDICATOR - Python port of the "Whale Hunter V10" Pine Script logic
# ============================================================================

def cci_close_series(close: pd.Series, length: int) -> np.ndarray:
    """Matches Pine's ta.cci(close, length): CCI computed directly on the
    `close` source (NOT typical price like a classic CCI):
        cci = (close - SMA(close,len)) / (0.015 * mean_abs_dev(close,len))
    """
    sma = close.rolling(length).mean()
    mad = close.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (close - sma) / (0.015 * mad)
    return cci.values.astype(float)


def compute_whale_indicators(df: pd.DataFrame, tf_cfg: dict) -> dict:
    """Reproduces is_w / c_val / e_200 / l_sig_raw / s_sig_raw from the
    Whale Hunter Pine script, vectorized over the whole candle history."""
    close = df["close"]
    volume = df["volume"]

    vol_ma = volume.rolling(tf_cfg["vol_ma_len"]).mean().values
    is_whale = (volume.values.astype(float) > (vol_ma * tf_cfg["vol_mult"]))

    cci = cci_close_series(close, tf_cfg["cci_len"])
    ema = close.ewm(span=tf_cfg["ema_len"], adjust=False).mean().values
    close_arr = close.values.astype(float)

    ob = tf_cfg["ob_level"]
    os_ = tf_cfg["os_level"]
    n = len(df)
    raw_long = np.zeros(n, dtype=bool)
    raw_short = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(cci[i]) or np.isnan(cci[i - 1]) or np.isnan(ema[i]) or np.isnan(vol_ma[i]):
            continue
        cross_up = cci[i - 1] <= ob and cci[i] > ob
        cross_dn = cci[i - 1] >= os_ and cci[i] < os_
        raw_long[i] = bool(is_whale[i] and cci[i] > ob and close_arr[i] > ema[i] and cross_up)
        raw_short[i] = bool(is_whale[i] and cci[i] < os_ and close_arr[i] < ema[i] and cross_dn)

    return {
        "cci": cci, "ema": ema, "vol_ma": vol_ma, "is_whale": is_whale,
        "raw_long": raw_long, "raw_short": raw_short,
    }


def compute_whale_signals(ind: dict, idx: int, tf_cfg: dict):
    """No-repaint action: mirrors Pine's `l_sig = nz(l_sig_raw[1], false)`.
    The raw cross must have been confirmed on the PREVIOUS closed candle
    (idx-1); we act on it now that idx itself has also fully closed."""
    if idx <= 0 or idx - 1 < 0:
        return False, False
    raw_long_prev = bool(ind["raw_long"][idx - 1])
    raw_short_prev = bool(ind["raw_short"][idx - 1])
    long_condition = tf_cfg["allow_long"] and raw_long_prev
    short_condition = tf_cfg["allow_short"] and raw_short_prev
    return long_condition, short_condition


def calc_tp_sl(tf_cfg: dict, is_long: bool, entry_price: float):
    if is_long:
        tp = entry_price * (1 + tf_cfg["tp_pct"] / 100.0)
        sl = entry_price * (1 - tf_cfg["sl_pct"] / 100.0)
    else:
        tp = entry_price * (1 - tf_cfg["tp_pct"] / 100.0)
        sl = entry_price * (1 + tf_cfg["sl_pct"] / 100.0)
    return tp, sl


def fee_usd(notional_value: float, fee_pct: float) -> float:
    return notional_value * fee_pct / 100.0


# ============================================================================
# 2) SHARED STATE
# ============================================================================

def _default_tf_settings():
    return {
        "enabled": True,
        "margin_usdt": 1.0,
        "allow_multi": True,
        "max_trades_per_side": 3,
        "cci_len": 20,
        "ob_level": 100,
        "os_level": -100,
        "vol_ma_len": 20,
        "vol_mult": 2.0,
        "ema_len": 200,
        "tp_pct": 3.0,
        "sl_pct": 5.0,
        "allow_long": True,
        "allow_short": True,
        "sl_first_if_both_hit": True,
    }


DEFAULT_CONFIG = {
    "symbol": "BEAT/USDT:USDT",
    "bot_start_date": datetime.now().strftime("%Y-%m-%d"),

    "initial_cap": 10.0,     # ทุนเริ่มต้น (USD) - account-level
    "leverage": 20,          # Leverage (เท่า) - account-level, shared by both TF
    "fee_pct": 0.04,         # ค่าธรรมเนียม (% ของมูลค่าสัญญาจริง ต่อฝั่ง) - account-level

    "stop_when_blown": True,  # หยุดเปิดไม้ใหม่เมื่อพอร์ตติดลบ (Equity <= 0) - account-level
    "paper_mode": True,       # โหมดสมุดทดลอง (Paper/Dry-run)

    "chart_tf": "5m",         # which TF's candles are drawn on the price chart (display only)
    "poll_seconds": 10,
    "candle_limit": 400,      # bars fetched per TF per poll (must cover ema_len of both TFs)

    "tf_settings": {
        "3m": _default_tf_settings(),
        "5m": _default_tf_settings(),
    },
}


def empty_stats():
    return {"total_trades": 0, "wins": 0, "losses": 0, "net_profit": 0.0}


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = json.loads(json.dumps(DEFAULT_CONFIG))
        self.tickets = []          # open tickets: list of dicts, each has "tf": "3m"|"5m"
        self.stats = empty_stats()  # account-level cumulative stats (both TF combined)
        self.bot_stopped = False   # kill switch latch (equity <= 0)
        self.running = False
        self.connected = False
        self.trades = []           # closed/open trade log rows, each has "tf"
        self.logs = []
        self.market = {tf: {"ohlcv": [], "cci": None, "ema": None, "vol_ratio": None} for tf in TF_KEYS}
        self.live_price = None
        self.balance = 0.0
        self.last_signal_bar_ts = {tf: None for tf in TF_KEYS}  # per-TF new-bar guard
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                cfg = data.get("config", {})
                # merge nested tf_settings safely so old state files / new keys don't crash
                tfset = cfg.pop("tf_settings", {})
                self.config.update(cfg)
                for tf in TF_KEYS:
                    self.config["tf_settings"][tf].update(tfset.get(tf, {}))
                self.tickets = data.get("tickets", [])
                self.stats.update(data.get("stats", {}))
                self.bot_stopped = data.get("bot_stopped", False)
                self.trades = data.get("trades", [])
                self.logs = data.get("logs", [])
                lsbt = data.get("last_signal_bar_ts", {})
                for tf in TF_KEYS:
                    self.last_signal_bar_ts[tf] = lsbt.get(tf)
            except Exception:
                pass

    def _save(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "config": self.config, "tickets": self.tickets, "stats": self.stats,
                    "bot_stopped": self.bot_stopped,
                    "trades": self.trades[:500], "logs": self.logs[-300:],
                    "last_signal_bar_ts": self.last_signal_bar_ts,
                }, f)
        except Exception:
            pass

    def update_config(self, patch_global: dict, patch_tf: dict):
        with self.lock:
            self.config.update(patch_global)
            for tf, sub in patch_tf.items():
                if tf in self.config["tf_settings"]:
                    self.config["tf_settings"][tf].update(sub)
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

    def update_market(self, tf: str, ohlcv, cci_val, ema_val, vol_ratio):
        with self.lock:
            self.market[tf] = {"ohlcv": ohlcv, "cci": cci_val, "ema": ema_val, "vol_ratio": vol_ratio}

    def count_tf_side(self, tf: str, side: str) -> int:
        return sum(1 for t in self.tickets if t["tf"] == tf and t["side"] == side)

    def count_tf(self, tf: str) -> int:
        return sum(1 for t in self.tickets if t["tf"] == tf)

    def equity(self):
        return round(self.config["initial_cap"] + self.stats["net_profit"], 6)

    def open_mark_value(self):
        """Unrealized mark-to-market PnL of ALL open tickets (both TFs) at
        the single shared live_price (same symbol, one real market price)."""
        if not self.tickets or not self.live_price:
            return 0.0
        px = self.live_price
        total = 0.0
        for t in self.tickets:
            if t["side"] == "LONG":
                total += (px - t["entry_price"]) * t["contract_amount"]
            else:
                total += (t["entry_price"] - px) * t["contract_amount"]
        return round(total, 6)

    def tf_stats(self):
        """Derive per-timeframe win/loss/net-profit breakdown from the trade log."""
        out = {}
        for tf in TF_KEYS:
            wins = losses = 0
            net = 0.0
            for row in self.trades:
                if row.get("tf") != tf or row.get("status") not in ("WIN", "LOSS"):
                    continue
                if row["status"] == "WIN":
                    wins += 1
                else:
                    losses += 1
                net += row.get("pnl") or 0.0
            total = wins + losses
            out[tf] = {
                "total_trades": total, "wins": wins, "losses": losses,
                "net_profit": round(net, 6),
                "winrate": round((wins / total) * 100, 2) if total else 0.0,
                "open_long": self.count_tf_side(tf, "LONG"),
                "open_short": self.count_tf_side(tf, "SHORT"),
            }
        return out

    def snapshot(self):
        with self.lock:
            equity = self.equity()
            unrealized_equity = round(equity + self.open_mark_value(), 6)
            winrate = round((self.stats["wins"] / self.stats["total_trades"]) * 100, 2) if self.stats["total_trades"] else 0.0
            net_pct = round((self.stats["net_profit"] / self.config["initial_cap"]) * 100, 2) if self.config["initial_cap"] else 0.0
            return {
                "config": json.loads(json.dumps(self.config)),
                "tickets": list(self.tickets),
                "stats": dict(self.stats),
                "tf_stats": self.tf_stats(),
                "bot_stopped": self.bot_stopped,
                "running": self.running,
                "connected": self.connected,
                "live_price": self.live_price,
                "balance": self.balance,
                "equity": equity,
                "unrealized_equity": unrealized_equity,
                "winrate": winrate,
                "net_pct": net_pct,
                "market": self.market,
                "trades": self.trades[:80],
                "logs": self.logs[-100:],
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


def can_open(tf: str, side: str, tf_cfg: dict) -> bool:
    if tf_cfg["allow_multi"]:
        return state.count_tf_side(tf, side) < tf_cfg["max_trades_per_side"]
    return state.count_tf(tf) == 0


def open_ticket(symbol: str, tf: str, side: str, entry_price: float, cfg: dict, tf_cfg: dict, manual=False):
    """Opens one ticket tagged to timeframe `tf`. margin x leverage = notional
    (margin is per-TF, leverage is account-level); contract units =
    notional / entry_price. Entry fee is deducted from net_profit immediately."""
    paper = bool(cfg.get("paper_mode"))
    try:
        notional = tf_cfg["margin_usdt"] * cfg["leverage"]
        contract_amount = round(notional / entry_price, 4)
        tp, sl = calc_tp_sl(tf_cfg, side == "LONG", entry_price)
        tp = round(float(tp), 6)
        sl = round(float(sl), 6)
        entry_fee = fee_usd(notional, cfg["fee_pct"])

        if not paper:
            ex = get_exchange()
            try:
                ex.set_leverage(cfg["leverage"], symbol)
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

        ticket_id = uuid.uuid4().hex[:10]
        with state.lock:
            state.tickets.append({
                "id": ticket_id, "tf": tf, "side": side, "entry_price": entry_price,
                "tp": tp, "sl": sl, "contract_amount": contract_amount,
                "notional": notional, "opened_ms": int(datetime.now().timestamp() * 1000),
            })
            state.stats["net_profit"] = round(state.stats["net_profit"] - entry_fee, 6)
            state._save()

        tag = " [PAPER]" if paper else (" [MANUAL]" if manual else "")
        state.add_trade({
            "id": ticket_id, "tf": tf,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_ms": int(datetime.now().timestamp() * 1000),
            "type": f"OPEN ({'BUY' if side == 'LONG' else 'SELL'}) {tf} Whale{tag}",
            "entry": entry_price, "tp": tp, "sl": sl,
            "contract_amount": contract_amount, "notional": notional,
            "pnl": None, "status": "OPEN", "side": side,
        })
        state.add_log(f"{'PAPER ' if paper else ''}ENTRY {'SIMULATED' if paper else 'EXECUTED'}{tag} [{tf}]: "
                       f"{side} @ {entry_price} | notional ${notional:.2f} | TP {tp} / SL {sl} | entry fee ${entry_fee:.4f}")
        return True, ticket_id
    except Exception as e:
        state.add_log(f"⚠️ ORDER FAILED [{tf}] ({side}): {e}")
        return False, str(e)


def close_ticket_by_id(ticket_id: str, exit_price: float, reason: str):
    """Software-side bookkeeping close — records realized PnL for one ticket
    (using the ACCOUNT-level fee_pct, since fees are charged by the real
    exchange the same way regardless of which TF strategy opened it) and
    removes it from the open list. In LIVE mode the caller is responsible
    for sending the real reduceOnly close order first."""
    with state.lock:
        idx = next((i for i, t in enumerate(state.tickets) if t["id"] == ticket_id), None)
        if idx is None:
            return
        t = state.tickets.pop(idx)
        entry = t["entry_price"]
        amount = t["contract_amount"]
        side = t["side"]
        gross_pnl = (exit_price - entry) * amount if side == "LONG" else (entry - exit_price) * amount
        exit_fee = fee_usd(amount * exit_price, state.config["fee_pct"])
        net_pnl = gross_pnl - exit_fee

        state.stats["total_trades"] += 1
        if net_pnl > 0:
            state.stats["wins"] += 1
        else:
            state.stats["losses"] += 1
        state.stats["net_profit"] = round(state.stats["net_profit"] + net_pnl, 6)

        for row in state.trades:
            if row.get("id") == ticket_id and row.get("status") == "OPEN":
                row["status"] = "WIN" if net_pnl > 0 else "LOSS"
                row["exit"] = exit_price
                row["exit_ms"] = int(datetime.now().timestamp() * 1000)
                row["pnl"] = round(net_pnl, 4)
                break
        state._save()

    state.add_log(f"Ticket {ticket_id} [{t['tf']}] closed ({reason}) {side} @ ~{exit_price} | net PnL {round(net_pnl, 4)} USDT")


def close_all_tickets_kill_switch(symbol: str):
    """Kill switch: flattens every open ticket across BOTH timeframes,
    grouped by side so at most one real close order per side is sent in
    LIVE mode (mirrors how BingX merges same-side tickets — regardless of
    which TF opened them — into a single real position)."""
    if not state.tickets:
        return
    px = state.live_price or state.tickets[0]["entry_price"]
    paper = bool(state.config.get("paper_mode"))

    if not paper:
        ex = get_exchange()
        for side in ("LONG", "SHORT"):
            side_tickets = [t for t in state.tickets if t["side"] == side]
            if not side_tickets:
                continue
            total_amount = round(sum(t["contract_amount"] for t in side_tickets), 4)
            close_side = "sell" if side == "LONG" else "buy"
            try:
                ex.create_order(symbol=symbol, type="market", side=close_side, amount=total_amount,
                                 params={"positionSide": side, "reduceOnly": True})
            except Exception as e:
                state.add_log(f"⚠️ Kill-switch close failed ({side}): {e}")

    for t in list(state.tickets):
        close_ticket_by_id(t["id"], px, "ACCOUNT BLOWN - STOPPED")


def open_trade_manual(tf: str, side: str):
    snap_cfg = state.snapshot()["config"]
    if tf not in TF_KEYS:
        return False, "invalid timeframe"
    tf_cfg = snap_cfg["tf_settings"][tf]
    if state.bot_stopped:
        return False, "Bot is stopped (equity <= 0)"
    if not state.live_price:
        return False, "No live price yet"
    if not can_open(tf, side, tf_cfg):
        return False, f"Max trades reached for {tf} {side}" if tf_cfg["allow_multi"] else f"A {tf} ticket is already open"
    ok, msg = open_ticket(snap_cfg["symbol"], tf, side, state.live_price, snap_cfg, tf_cfg, manual=True)
    return ok, msg


def close_trade_manual(ticket_id: str):
    t = next((t for t in state.tickets if t["id"] == ticket_id), None)
    if t is None:
        return False, "Ticket not found"
    cfg = state.snapshot()["config"]
    exit_price = state.live_price or t["entry_price"]

    if cfg.get("paper_mode"):
        close_ticket_by_id(ticket_id, exit_price, "MANUAL CLOSE (PAPER)")
        return True, "closed (paper)"

    ex = get_exchange()
    close_side = "sell" if t["side"] == "LONG" else "buy"
    try:
        ex.create_order(symbol=cfg["symbol"], type="market", side=close_side, amount=t["contract_amount"],
                         params={"positionSide": t["side"], "reduceOnly": True})
        close_ticket_by_id(ticket_id, exit_price, "MANUAL CLOSE")
        return True, "closed"
    except Exception as e:
        state.add_log(f"⚠️ Manual close failed: {e}")
        return False, str(e)


def _process_timeframe(ex, symbol: str, tf: str, cfg: dict):
    """Fully independent per-timeframe cycle: fetch candles, compute Whale
    Hunter indicators for THIS timeframe only, manage THIS timeframe's own
    open tickets against THIS timeframe's own candles, and look for a new
    entry using THIS timeframe's own signal/TP/SL/ticket-slot settings."""
    tf_cfg = cfg["tf_settings"][tf]
    if not tf_cfg["enabled"]:
        return

    needed = max(tf_cfg["ema_len"] + 50, tf_cfg["cci_len"] + 50, tf_cfg["vol_ma_len"] + 50, cfg["candle_limit"])
    bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=needed)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)

    ind = compute_whale_indicators(df, tf_cfg)
    idx = len(df) - 2  # last CLOSED candle on THIS timeframe

    cci_val = None if idx < 0 or np.isnan(ind["cci"][idx]) else round(float(ind["cci"][idx]), 2)
    ema_val = None if idx < 0 or np.isnan(ind["ema"][idx]) else round(float(ind["ema"][idx]), 4)
    vol_ratio = None
    if idx >= 0 and not np.isnan(ind["vol_ma"][idx]) and ind["vol_ma"][idx] > 0:
        vol_ratio = round(float(df["volume"].values[idx]) / float(ind["vol_ma"][idx]), 2)

    ohlcv_payload = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
    state.update_market(tf, ohlcv_payload, cci_val, ema_val, vol_ratio)

    # 1) Manage this timeframe's own open tickets against this timeframe's own last closed candle
    if not state.bot_stopped and idx >= 0:
        hi, lo = high_arr[idx], low_arr[idx]
        my_tickets = [t for t in state.tickets if t["tf"] == tf]
        for t in my_tickets:
            side, tp, sl = t["side"], t["tp"], t["sl"]
            if side == "LONG":
                hit_tp, hit_sl = hi >= tp, lo <= sl
            else:
                hit_tp, hit_sl = lo <= tp, hi >= sl

            if hit_tp and hit_sl:
                exit_price, reason = (sl, "SL") if tf_cfg["sl_first_if_both_hit"] else (tp, "TP")
            elif hit_sl:
                exit_price, reason = sl, "SL"
            elif hit_tp:
                exit_price, reason = tp, "TP"
            else:
                continue

            if not cfg.get("paper_mode"):
                close_side = "sell" if side == "LONG" else "buy"
                try:
                    ex.create_order(symbol=symbol, type="market", side=close_side,
                                     amount=t["contract_amount"],
                                     params={"positionSide": side, "reduceOnly": True})
                except Exception as e:
                    state.add_log(f"⚠️ Auto-close order failed for ticket {t['id']} [{tf}]: {e}")
            close_ticket_by_id(t["id"], exit_price, reason)

    # 2) Look for a new entry on this timeframe — only ONCE per newly-closed candle on it
    bar_ts = int(df["timestamp"].iloc[idx]) if idx >= 0 else None
    is_new_bar = bar_ts is not None and bar_ts != state.last_signal_bar_ts.get(tf)

    if (not state.bot_stopped and state.running and _is_trading_started(cfg)
            and idx > 0 and is_new_bar):
        long_condition, short_condition = compute_whale_signals(ind, idx, tf_cfg)
        entry_price = float(df["close"].values[idx])  # this TF's own last closed price
        if long_condition and can_open(tf, "LONG", tf_cfg):
            open_ticket(symbol, tf, "LONG", entry_price, cfg, tf_cfg)
        elif short_condition and can_open(tf, "SHORT", tf_cfg):
            open_ticket(symbol, tf, "SHORT", entry_price, cfg, tf_cfg)
        state.last_signal_bar_ts[tf] = bar_ts


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (3m + 5m Whale Hunter, independent strategies)")
    while not _stop_flag.is_set():
        cfg = state.snapshot()["config"]
        try:
            symbol = cfg["symbol"]
            ex = get_exchange()

            # one shared real-market price for balance/equity/manual-order/kill-switch,
            # independent of either TF's candle close
            try:
                ticker = ex.fetch_ticker(symbol)
                live_price = float(ticker["last"])
            except Exception:
                live_price = state.live_price
            if live_price:
                state.live_price = live_price
            state.connected = True

            if cfg.get("paper_mode"):
                state.balance = round(state.equity() + state.open_mark_value(), 4)
            else:
                try:
                    bal = ex.fetch_balance()
                    state.balance = float(bal.get("USDT", {}).get("total", 0.0))
                except Exception:
                    pass

            # 1) Kill switch check FIRST (account-level) — realized equity <= 0 halts everything
            if cfg["stop_when_blown"] and not state.bot_stopped and state.equity() <= 0:
                state.bot_stopped = True
                state.running = False
                state.add_log("🛑 ACCOUNT BLOWN (equity <= 0) — closing all open tickets (both TF) + halting new entries")
                close_all_tickets_kill_switch(symbol)

            # 2) Run each enabled timeframe's fully independent cycle
            for tf in TF_KEYS:
                try:
                    _process_timeframe(ex, symbol, tf, cfg)
                except Exception as tf_err:
                    state.add_log(f"⚠️ [{tf}] processing error [{type(tf_err).__name__}]: {tf_err}")

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
    if on and state.bot_stopped:
        state.add_log("⚠️ Cannot RUN — bot is stopped (equity <= 0). Reset via /api/reset-kill-switch first.")
        return
    state.running = bool(on)
    state.add_log("🟢 RUN enabled - both TF strategies will open tickets on new signals" if on
                  else "⏸ RUN disabled - market data keeps refreshing, no new tickets will open")


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
<title>Whale Hunter Bot — 3m + 5m Independent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{ --bg:#0A0C12; --panel:#12151E; --panel-2:#171B27; --line:#232838; --text:#E7ECF3; --muted:#7C879C;
    --gold:#C9A24A; --gold-dim:#8A7130; --long:#3ED8A0; --short:#FF5C72; --radius:10px; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;-webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  .app{display:grid;grid-template-columns:340px 1fr;grid-template-rows:64px 1fr;height:100vh;}
  .brand{grid-column:1/2;grid-row:1;display:flex;align-items:center;gap:12px;padding:0 20px;border-bottom:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:12px;}
  .brand h1{font-size:13px;letter-spacing:.04em;margin:0;font-weight:600;}
  .brand .sub{color:var(--muted);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;}
  .topbar{grid-column:2;grid-row:1;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--line);}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .dot.on{background:var(--long);box-shadow:0 0 8px var(--long);}
  .dot.stopped{background:var(--short);box-shadow:0 0 8px var(--short);}
  .run-btn{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:9px 18px;border-radius:8px;font-family:inherit;font-weight:600;font-size:12px;letter-spacing:.04em;cursor:pointer;}
  .run-btn.active{background:var(--long);color:#062018;border-color:var(--long);}
  .run-btn.stopped{background:var(--short);color:#2a0509;border-color:var(--short);}
  .manual-group{display:flex;gap:6px;align-items:center;}
  select.tfsel{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:7px 8px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:11px;}
  .manual-btn{border:1px solid var(--line);background:var(--panel-2);color:var(--text);padding:7px 12px;border-radius:6px;font-family:inherit;font-weight:600;font-size:11px;cursor:pointer;}
  .manual-btn.long{color:var(--long);border-color:var(--long);}
  .manual-btn.short{color:var(--short);border-color:var(--short);}
  .close-btn{background:transparent;border:1px solid var(--short);color:var(--short);padding:3px 9px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;}
  .control{grid-column:1;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:20px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .control h3{font-size:12.5px;color:var(--gold);margin:16px 0 8px;padding-top:10px;border-top:1px dashed var(--line);}
  .field{margin-bottom:10px;}
  .field label{display:block;font-size:11.5px;color:var(--muted);margin-bottom:5px;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .save-btn{width:100%;margin-top:14px;background:var(--gold);color:#211a06;border:none;padding:11px;border-radius:8px;font-weight:700;font-size:12.5px;letter-spacing:.03em;cursor:pointer;font-family:inherit;}
  .reset-btn{width:100%;margin-top:8px;background:transparent;border:1px solid var(--short);color:var(--short);padding:9px;border-radius:8px;font-weight:700;font-size:11.5px;cursor:pointer;font-family:inherit;}
  .main{grid-column:2;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;}
  .chart-toggle{display:flex;gap:4px;}
  .chart-toggle button{background:var(--panel-2);border:1px solid var(--line);color:var(--muted);padding:4px 10px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:11px;cursor:pointer;}
  .chart-toggle button.active{color:var(--gold);border-color:var(--gold);}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:280px;}
  .tf-cci-cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  .tf-block{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px;}
  .tf-block .tfh{font-size:12px;font-weight:700;color:var(--gold);margin-bottom:10px;display:flex;justify-content:space-between;}
  .cci-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
  .cci-card{background:var(--panel-2);border:1px solid var(--line);border-radius:8px;padding:10px 12px;}
  .cci-card .lbl{font-size:10px;color:var(--muted);text-transform:uppercase;}
  .cci-card .val{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:600;margin-top:3px;}
  .tf-mini-stats{display:flex;gap:10px;margin-top:10px;font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;}
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
  .tf-tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;background:var(--panel-2);border:1px solid var(--line);color:var(--gold);}
  .logbox{max-height:230px;overflow-y:auto;font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
  .logline{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #171a24;color:#B7C0D1;}
  .logline .t{color:var(--muted);}
  ::-webkit-scrollbar{width:8px;height:8px;} ::-webkit-scrollbar-thumb{background:#232838;border-radius:4px;}
  .badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;white-space:nowrap;}
  .badge-open{background:rgba(201,162,74,0.15);color:var(--gold);border:1px solid rgba(201,162,74,0.4);}
  .badge-win{background:rgba(62,216,160,0.15);color:var(--long);border:1px solid rgba(62,216,160,0.4);}
  .badge-loss{background:rgba(255,92,114,0.15);color:var(--short);border:1px solid rgba(255,92,114,0.4);}
  .warn{margin-top:16px;padding:10px 12px;background:#241412;border:1px solid #4a2620;border-radius:8px;font-size:11px;color:#f0a89c;line-height:1.5;}
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
  <div class="brand"><div class="mark">WH</div><div><h1>WHALE HUNTER · 3m + 5m INDEPENDENT</h1><div class="sub" id="symbolLabel">LOADING…</div></div></div>
  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span><span class="badge" id="modeBadge" style="margin-left:10px">--</span></div>
    <div class="manual-group">
      <span class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></span>
      <select class="tfsel" id="manualTfSel"><option value="3m">TF 3m</option><option value="5m">TF 5m</option></select>
      <button class="manual-btn long" id="manualLongBtn">▲ LONG</button>
      <button class="manual-btn short" id="manualShortBtn">▼ SHORT</button>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>

  <div class="control">
    <h2>โหมดการทำงาน / บัญชี</h2>
    <div class="field"><label>โหมด</label>
      <select id="cfg_paper_mode"><option value="true">📝 Paper (สมุดทด — ไม่ยิงออเดอร์จริง)</option><option value="false">🔴 Live (ยิงออเดอร์จริง)</option></select>
    </div>
    <div class="field"><label>Symbol</label><input id="cfg_symbol" type="text" /></div>
    <div class="row2">
      <div class="field"><label>ทุนเริ่มต้น (USD)</label><input id="cfg_initial_cap" type="number" step="0.01" /></div>
      <div class="field"><label>Leverage (เท่า, ใช้ร่วมกันทั้ง 2 TF)</label><input id="cfg_leverage" type="number" step="1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>ค่าธรรมเนียม (% ของมูลค่าสัญญาจริง)</label><input id="cfg_fee_pct" type="number" step="0.01" /></div>
      <div class="field"><label>วันเริ่มเทรด</label><input id="cfg_bot_start_date" type="date" /></div>
    </div>
    <div class="field"><label>หยุดเปิดไม้ใหม่เมื่อพอร์ตติดลบ (Equity &lt;= 0)</label><select id="cfg_stop_when_blown"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>

    <div id="tfPanels"></div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS</button>
    <button class="reset-btn" id="resetKillBtn">RESET KILL SWITCH</button>
    <div class="warn">TF 3 นาที และ TF 5 นาที เป็นสองกลยุทธ์อิสระต่อกัน (สัญญาณ/TP/SL/จำนวนไม้ แยกกันคนละชุด) แต่เทรดสัญลักษณ์เดียวกันบนบัญชีเดียวกัน — ถ้าทั้งสอง TF เปิดฝั่งเดียวกันพร้อมกันใน Live mode, BingX จะรวมเป็นโพซิชั่นจริงเดียว (มี TP/SL จริงต่อไม้ แต่สถิติรายไม้/รายTF อาจคลาดเคลื่อนเล็กน้อยที่ขอบเขต)</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart</span>
        <div class="chart-toggle" id="chartToggle"><button data-tf="3m">3m</button><button data-tf="5m">5m</button></div>
        <span class="price mono" id="livePrice">--</span>
      </div>
      <div id="priceChart"></div>
    </div>

    <div class="tf-cci-cols" id="tfCciCols"></div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">สถานะบอท</div><div class="val" id="m_status">--</div></div>
      <div class="metric-card"><div class="lbl">Equity / Mark-to-Market</div><div class="val" id="m_equity">--</div></div>
      <div class="metric-card"><div class="lbl">กำไร/ขาดทุนสุทธิ (รวม)</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">Winrate (รวม)</div><div class="val" id="m_winrate">--</div></div>
    </div>

    <div class="panel">
      <div class="panel-title"><span>ไม้ที่ถืออยู่ (Open Tickets — ทั้ง 2 TF)</span><span id="openCountLabel">--</span></div>
      <table><thead><tr><th>TF</th><th>Side</th><th>Entry</th><th>TP</th><th>SL</th><th>Units</th><th>Notional</th><th>Unrealized</th><th>Action</th></tr></thead><tbody id="openTicketsBody"></tbody></table>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table><thead><tr><th>TF</th><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>Status</th></tr></thead><tbody id="ordersBody"></tbody></table>
      </div>
      <div class="panel"><div class="panel-title">Bot Activity Log</div><div class="logbox" id="logBody"></div></div>
    </div>
  </div>
</div>

<script>
const GLOBAL_KEYS = ["symbol","bot_start_date","initial_cap","leverage","fee_pct","stop_when_blown","paper_mode"];
const TF_FIELD_KEYS = ["enabled","margin_usdt","allow_multi","max_trades_per_side","cci_len","ob_level","os_level",
  "vol_ma_len","vol_mult","ema_len","tp_pct","sl_pct","allow_long","allow_short","sl_first_if_both_hit"];
const TF_LABELS = {"3m":"TF 3 นาที","5m":"TF 5 นาที"};
const BOOL_TF_FIELDS = new Set(["enabled","allow_multi","allow_long","allow_short","sl_first_if_both_hit"]);

let chartTf = "5m";
let cfgLoadedOnce = false;
let lastSeenTradeId = null;
let firstRender = true;
let lastData = null;

const priceChart = LightweightCharts.createChart(document.getElementById('priceChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const candleSeries = priceChart.addCandlestickSeries({upColor:'#3ED8A0', downColor:'#FF5C72', borderVisible:false, wickUpColor:'#3ED8A0', wickDownColor:'#FF5C72'});

function buildTfPanels(){
  const wrap = document.getElementById('tfPanels');
  wrap.innerHTML = ["3m","5m"].map(tf => `
    <h3>${TF_LABELS[tf]}</h3>
    <div class="field"><label>เปิดใช้งานกลยุทธ์นี้</label><select id="cfg_${tf}_enabled"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    <div class="row2">
      <div class="field"><label>มูลค่ามาร์จิ้นต่อไม้ (USD)</label><input id="cfg_${tf}_margin_usdt" type="number" step="0.01" /></div>
      <div class="field"><label>จำนวนไม้เปิดพร้อมกันสูงสุดต่อฝั่ง</label><input id="cfg_${tf}_max_trades_per_side" type="number" step="1" /></div>
    </div>
    <div class="field"><label>เปิดใช้งานการเปิดไม้ซ้อน</label><select id="cfg_${tf}_allow_multi"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    <div class="row2">
      <div class="field"><label>CCI Length</label><input id="cfg_${tf}_cci_len" type="number" step="1" /></div>
      <div class="field"><label>Overbought</label><input id="cfg_${tf}_ob_level" type="number" step="1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Oversold</label><input id="cfg_${tf}_os_level" type="number" step="1" /></div>
      <div class="field"><label>EMA Length</label><input id="cfg_${tf}_ema_len" type="number" step="1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Volume MA Length</label><input id="cfg_${tf}_vol_ma_len" type="number" step="1" /></div>
      <div class="field"><label>Volume Multiplier (x)</label><input id="cfg_${tf}_vol_mult" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Take Profit (%)</label><input id="cfg_${tf}_tp_pct" type="number" step="0.1" /></div>
      <div class="field"><label>Stop Loss (%)</label><input id="cfg_${tf}_sl_pct" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>อนุญาตเปิด Long</label><select id="cfg_${tf}_allow_long"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>อนุญาตเปิด Short</label><select id="cfg_${tf}_allow_short"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    </div>
    <div class="field"><label>ถ้า TP/SL แตะพร้อมกัน นับ SL ก่อน</label><select id="cfg_${tf}_sl_first_if_both_hit"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
  `).join('');
}
buildTfPanels();

function statusBadge(t){
  if(t.status==='OPEN') return `<span class="badge badge-open">🔵 OPEN</span>`;
  if(t.status==='WIN')  return `<span class="badge badge-win">✅ TP/WIN</span>`;
  if(t.status==='LOSS') return `<span class="badge badge-loss">❌ SL/LOSS</span>`;
  return t.status||'';
}
function showToast(t){
  const isWin = t.status==='WIN';
  const el = document.createElement('div');
  el.className = 'toast ' + (isWin?'win':'loss');
  el.innerHTML = `<div class="ttitle">${isWin?'🎯 TP HIT':'🛑 SL HIT'} · <span class="tf-tag">${t.tf||''}</span></div><div class="tbody">${t.side} @ ${t.exit??''} · PnL ${(t.pnl>=0?'+':'')+t.pnl} USDT</div>`;
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
  try{ const res = await fetch('/api/status'); lastData = await res.json(); render(lastData); }catch(e){ console.error(e); }
  setTimeout(poll, 4000);
}

function isPaper(){ return document.getElementById('cfg_paper_mode').value === 'true'; }

function renderChart(data){
  const mkt = data.market[chartTf];
  if(mkt && mkt.ohlcv && mkt.ohlcv.length){
    candleSeries.setData(mkt.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]})));
    const markers = [];
    data.trades.filter(t=>t.tf===chartTf).forEach(t=>{
      markers.push({time: Math.floor(t.entry_ms/1000), position: t.side==='LONG'?'belowBar':'aboveBar', color: t.side==='LONG'?'#3ED8A0':'#FF5C72', shape: t.side==='LONG'?'arrowUp':'arrowDown', text: t.side});
      if((t.status==='WIN'||t.status==='LOSS') && t.exit_ms){
        markers.push({time: Math.floor(t.exit_ms/1000), position: t.status==='WIN'?'aboveBar':'belowBar', color: t.status==='WIN'?'#3ED8A0':'#FF5C72', shape: t.status==='WIN'?'circle':'square', text: t.status==='WIN'?'TP':'SL'});
      }
    });
    markers.sort((a,b)=>a.time-b.time);
    candleSeries.setMarkers(markers);
  }
  document.querySelectorAll('#chartToggle button').forEach(b=>b.classList.toggle('active', b.dataset.tf===chartTf));
}

function render(data){
  document.getElementById('connDot').className = 'dot ' + (data.bot_stopped?'stopped':(data.connected?'on':''));
  document.getElementById('connText').textContent = data.bot_stopped ? 'STOPPED (พอร์ตแตก)' : (data.connected ? 'CONNECTED · BingX' : 'CONNECTING…');
  const modeBadge = document.getElementById('modeBadge');
  modeBadge.textContent = data.config.paper_mode ? '📝 PAPER MODE' : '🔴 LIVE';
  modeBadge.className = 'badge ' + (data.config.paper_mode ? 'badge-open' : 'badge-loss');
  document.getElementById('symbolLabel').textContent = (data.config.symbol||'') + ' · 3m + 5m';
  document.getElementById('livePrice').textContent = data.live_price ? ('$'+data.live_price) : '--';
  document.getElementById('balanceVal').textContent = '$' + (data.balance||0).toFixed(2);

  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running?'active':'stopped');
  runBtn.disabled = data.bot_stopped;

  const cciWrap = document.getElementById('tfCciCols');
  cciWrap.innerHTML = ["3m","5m"].map(tf=>{
    const m = data.market[tf] || {};
    const ts = data.tf_stats[tf] || {};
    const tfc = data.config.tf_settings[tf] || {};
    return `<div class="tf-block">
      <div class="tfh"><span>${TF_LABELS[tf]}</span><span>${tfc.enabled ? '🟢 เปิด' : '⚪ ปิด'}</span></div>
      <div class="cci-row">
        <div class="cci-card"><div class="lbl">CCI</div><div class="val">${m.cci ?? '--'}</div></div>
        <div class="cci-card"><div class="lbl">EMA${tfc.ema_len||''}</div><div class="val">${m.ema ?? '--'}</div></div>
        <div class="cci-card"><div class="lbl">Vol / MA</div><div class="val">${m.vol_ratio!=null ? m.vol_ratio+'x' : '--'}</div></div>
      </div>
      <div class="tf-mini-stats">
        <span>เปิดอยู่: <span class="side-long">${ts.open_long||0}L</span> / <span class="side-short">${ts.open_short||0}S</span></span>
        <span>Winrate: ${ts.winrate||0}% (${ts.wins||0}W/${ts.losses||0}L)</span>
        <span class="${(ts.net_profit||0)>=0?'side-long':'side-short'}">Net: ${(ts.net_profit||0)>=0?'+':''}${(ts.net_profit||0).toFixed(4)}</span>
      </div>
    </div>`;
  }).join('');

  const statusEl = document.getElementById('m_status');
  const openCount = data.tickets.length;
  statusEl.textContent = data.bot_stopped ? 'STOPPED' : (openCount>0 ? `RUNNING (ถืออยู่ ${openCount} ไม้)` : (data.running ? 'RUNNING (รอสัญญาณ)' : 'IDLE'));
  statusEl.className = 'val ' + (data.bot_stopped ? 'neg' : (data.running ? 'pos' : ''));

  document.getElementById('m_equity').textContent = data.equity + ' / ' + data.unrealized_equity + ' USDT';
  const pnlEl = document.getElementById('m_pnl');
  pnlEl.textContent = (data.stats.net_profit>=0?'+':'') + data.stats.net_profit.toFixed(4) + ' USDT (' + (data.net_pct>=0?'+':'') + data.net_pct + '%)';
  pnlEl.className = 'val ' + (data.stats.net_profit>=0?'pos':'neg');
  document.getElementById('m_winrate').textContent = data.winrate + '% (' + data.stats.wins + 'W / ' + data.stats.losses + 'L)';

  const t3 = data.tf_stats['3m']||{}, t5 = data.tf_stats['5m']||{};
  document.getElementById('openCountLabel').textContent = `3m: ${t3.open_long||0}L/${t3.open_short||0}S (max ${data.config.tf_settings['3m'].max_trades_per_side}) · 5m: ${t5.open_long||0}L/${t5.open_short||0}S (max ${data.config.tf_settings['5m'].max_trades_per_side})`;
  const px = data.live_price;
  const openBody = document.getElementById('openTicketsBody');
  openBody.innerHTML = data.tickets.map(t=>{
    const unreal = px ? (t.side==='LONG' ? (px-t.entry_price)*t.contract_amount : (t.entry_price-px)*t.contract_amount) : 0;
    return `<tr>
      <td><span class="tf-tag">${t.tf}</span></td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.side}</td>
      <td>${t.entry_price}</td><td>${t.tp}</td><td>${t.sl}</td>
      <td>${t.contract_amount}</td><td>$${t.notional.toFixed(2)}</td>
      <td class="${unreal>=0?'side-long':'side-short'}">${unreal>=0?'+':''}${unreal.toFixed(4)}</td>
      <td><button class="close-btn" data-id="${t.id}">CLOSE</button></td>
    </tr>`;
  }).join('') || `<tr><td colspan="9" style="color:var(--muted)">ไม่มีไม้ที่เปิดอยู่</td></tr>`;
  openBody.querySelectorAll('.close-btn').forEach(btn=>{
    btn.addEventListener('click', async ()=>{
      btn.disabled=true; btn.textContent='...';
      const res = await fetch('/api/close-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticket_id: btn.dataset.id})});
      const j = await res.json();
      if(!j.ok) alert('Close failed: '+(j.message||j.error||'unknown error'));
    });
  });

  renderChart(data);

  const ordersBody = document.getElementById('ordersBody');
  ordersBody.innerHTML = data.trades.map(t=>`
    <tr>
      <td><span class="tf-tag">${t.tf||''}</span></td>
      <td>${t.time||''}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.type||''}</td>
      <td>${t.entry??''}</td><td>${t.tp??''}</td><td>${t.sl??''}</td>
      <td>${statusBadge(t)}</td>
    </tr>`).join('');

  maybeShowToast(data.trades);
  document.getElementById('logBody').innerHTML = data.logs.slice().reverse().map(l=>`<div class="logline"><span class="t">${l.time}</span><span>${l.text}</span></div>`).join('');

  if(!cfgLoadedOnce){
    GLOBAL_KEYS.forEach(k=>{ const el = document.getElementById('cfg_'+k); if(el && data.config[k]!==undefined) el.value = data.config[k]; });
    ["3m","5m"].forEach(tf=>{
      const tfc = data.config.tf_settings[tf] || {};
      TF_FIELD_KEYS.forEach(k=>{ const el = document.getElementById(`cfg_${tf}_${k}`); if(el && tfc[k]!==undefined) el.value = tfc[k]; });
    });
    chartTf = data.config.chart_tf || "5m";
    cfgLoadedOnce = true;
  }
}

document.getElementById('chartToggle').addEventListener('click', (e)=>{
  const btn = e.target.closest('button'); if(!btn) return;
  chartTf = btn.dataset.tf;
  if(lastData) renderChart(lastData);
});

document.getElementById('manualLongBtn').addEventListener('click', async ()=>{
  const tf = document.getElementById('manualTfSel').value;
  const msg = isPaper() ? `บันทึกออเดอร์ LONG จำลอง (Paper) บน TF ${tf} ตอนนี้เลยไหม?` : `เปิดออเดอร์ LONG ด้วยมือบน TF ${tf} ยิงจริงทันที แน่ใจไหม?`;
  if(!confirm(msg)) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({side:'LONG', tf})});
  const j = await res.json();
  if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: '+(j.message||j.error||'unknown error'));
});
document.getElementById('manualShortBtn').addEventListener('click', async ()=>{
  const tf = document.getElementById('manualTfSel').value;
  const msg = isPaper() ? `บันทึกออเดอร์ SHORT จำลอง (Paper) บน TF ${tf} ตอนนี้เลยไหม?` : `เปิดออเดอร์ SHORT ด้วยมือบน TF ${tf} ยิงจริงทันที แน่ใจไหม?`;
  if(!confirm(msg)) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({side:'SHORT', tf})});
  const j = await res.json();
  if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: '+(j.message||j.error||'unknown error'));
});
document.getElementById('runBtn').addEventListener('click', async ()=>{
  const willRun = document.getElementById('runBtn').textContent.includes('RUN');
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({running: willRun})});
});
document.getElementById('resetKillBtn').addEventListener('click', async ()=>{
  if(!confirm('รีเซ็ต Kill Switch? บอทจะสามารถ RUN ได้อีกครั้ง')) return;
  await fetch('/api/reset-kill-switch', {method:'POST'});
});
document.getElementById('saveCfgBtn').addEventListener('click', async ()=>{
  const patchGlobal = {};
  GLOBAL_KEYS.forEach(k=>{ const el = document.getElementById('cfg_'+k); if(el) patchGlobal[k]=el.value; });
  patchGlobal['chart_tf'] = chartTf;
  const patchTf = {"3m":{}, "5m":{}};
  ["3m","5m"].forEach(tf=>{
    TF_FIELD_KEYS.forEach(k=>{ const el = document.getElementById(`cfg_${tf}_${k}`); if(el) patchTf[tf][k]=el.value; });
  });
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({global: patchGlobal, tf: patchTf})});
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
    return jsonify({"status": "ok", "connected": state.connected, "running": state.running, "bot_stopped": state.bot_stopped})


@app.route("/api/status")
def api_status():
    return jsonify(state.snapshot())


@app.route("/api/config", methods=["POST"])
def api_config():
    payload = request.get_json(force=True) or {}
    patch_global_raw = payload.get("global", {})
    patch_tf_raw = payload.get("tf", {})

    numeric_global = {"leverage", "initial_cap", "fee_pct"}
    bool_global = {"stop_when_blown", "paper_mode"}
    string_global = {"symbol", "bot_start_date", "chart_tf"}

    numeric_tf = {"margin_usdt", "max_trades_per_side", "cci_len", "ob_level", "os_level",
                  "vol_ma_len", "vol_mult", "ema_len", "tp_pct", "sl_pct"}
    bool_tf = {"enabled", "allow_multi", "allow_long", "allow_short", "sl_first_if_both_hit"}

    clean_global = {}
    for k, v in patch_global_raw.items():
        if k in numeric_global:
            try:
                clean_global[k] = float(v) if "." in str(v) else int(v)
            except (TypeError, ValueError):
                continue
        elif k in bool_global:
            clean_global[k] = str(v).strip().lower() == "true"
        elif k in string_global:
            clean_global[k] = str(v)

    clean_tf = {}
    for tf, sub in patch_tf_raw.items():
        if tf not in TF_KEYS:
            continue
        clean_sub = {}
        for k, v in (sub or {}).items():
            if k in numeric_tf:
                try:
                    clean_sub[k] = float(v) if "." in str(v) else int(v)
                except (TypeError, ValueError):
                    continue
            elif k in bool_tf:
                clean_sub[k] = str(v).strip().lower() == "true"
        clean_tf[tf] = clean_sub

    state.update_config(clean_global, clean_tf)
    state.add_log(f"Config updated: global={clean_global} tf={clean_tf}")
    return jsonify(state.snapshot()["config"])


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    set_running(bool(payload.get("running")))
    return jsonify({"running": state.running, "bot_stopped": state.bot_stopped})


@app.route("/api/reset-kill-switch", methods=["POST"])
def api_reset_kill_switch():
    state.bot_stopped = False
    state.add_log("♻️ Kill switch reset by user")
    state._save()
    return jsonify({"bot_stopped": state.bot_stopped})


@app.route("/api/manual-order", methods=["POST"])
def api_manual_order():
    payload = request.get_json(force=True) or {}
    side = (payload.get("side") or "").upper()
    tf = payload.get("tf") or "5m"
    if side not in ("LONG", "SHORT"):
        return jsonify({"ok": False, "error": "invalid side"}), 400
    if tf not in TF_KEYS:
        return jsonify({"ok": False, "error": "invalid timeframe"}), 400
    ok, msg = open_trade_manual(tf, side)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/close-order", methods=["POST"])
def api_close_order():
    payload = request.get_json(force=True) or {}
    ticket_id = payload.get("ticket_id")
    if not ticket_id:
        return jsonify({"ok": False, "error": "missing ticket_id"}), 400
    ok, msg = close_trade_manual(ticket_id)
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
