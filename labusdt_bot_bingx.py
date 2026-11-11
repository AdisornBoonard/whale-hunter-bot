"""
cci_5pair_bot.py
-----------------
Single-file drop-in bot for BingX Futures — trades the Pine Script
"CCI Divergence Entry Signal [Winrate Backtest Dashboard]".

  * 5 independent coin-pair slots (P1..P5). EVERY slot has its own full set of
    settings (symbol, timeframe, leverage, EMA filter, CCI, divergence pivots,
    TP/SL, fee, compound/margin ...).
  * Each slot has an ENABLE checkbox (only enabled slots are traded).
  * Each slot has a REVERSE (เทรดสวน) checkbox: when ticked, a BUY signal opens
    a SELL and a SELL signal opens a BUY — per slot.
  * One open ticket per slot at a time (same as the Pine `in_trade` flag).
    Slots never block each other.

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python cci_5pair_bot.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT cci_5pair_bot:app

!! MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders.

NOTES
-----
* Do not give two enabled slots the SAME symbol — BingX would merge them into
  one real position. The dashboard warns you if you do.
* Signals are evaluated once per CLOSED candle (like the Pine script on bar
  close). After you press RUN, the bot "arms" on the current closed candle and
  only acts on signals from candles that close AFTER that.
* Equity / win-rate / fees on the dashboard are SIMULATED the same way the
  Pine dashboard does it (TP hit = +notional*TP%, SL hit = -notional*SL%,
  minus fees). Real money management is done by the real TP/SL bracket orders
  placed on BingX with every entry.
"""

import os
import json
import math
import copy
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

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_cci.json")

PAIR_IDS = ["P1", "P2", "P3", "P4", "P5"]
TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}

# ============================================================================
# 1) INDICATOR - Python port of the Pine Script (CCI regular divergence)
# ============================================================================

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


def bars_since(cond: np.ndarray):
    """Python port of ta.barssince(cond) (NaN until cond was true once)."""
    n = len(cond)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if cond[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def ema_series(close: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(close).ewm(span=length, adjust=False).mean().values


def cci_series(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.cci: (src - sma) / (0.015 * mean-absolute-deviation)."""
    s = pd.Series(src)
    sma = s.rolling(length).mean()
    dev = s.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (s - sma) / (0.015 * dev)
    return cci.replace([np.inf, -np.inf], np.nan).values


def pivots(values: np.ndarray, left: int, right: int):
    """ta.pivothigh / ta.pivotlow flags at the pivot bar itself."""
    n = len(values)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for j in range(left, n - right):
        w = values[j - left: j + right + 1]
        if np.isnan(w).any():
            continue
        v = values[j]
        if v == w.max():
            is_high[j] = True
        if v == w.min():
            is_low[j] = True
    return is_high, is_low


def compute_pair_signals(df: pd.DataFrame, p: dict) -> dict:
    """Full port of sections 2-5 of the Pine script for ONE pair's settings.
    Returns arrays aligned with df rows: buy / sell (signal at bar i, i.e. the
    bar on which the pivot gets confirmed, exactly like Pine)."""
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    lbL, lbR = int(p["lbL"]), int(p["lbR"])

    osc = cci_series(close, int(p["cci_len"]))
    is_high, is_low = pivots(osc, lbL, lbR)
    pl = shift_by(is_low, lbR)      # plFound  (pivot confirmed at this bar)
    ph = shift_by(is_high, lbR)     # phFound
    osc_s = shift_by(osc, lbR)      # osc[lbR]
    low_s = shift_by(low, lbR)      # low[lbR]
    high_s = shift_by(high, lbR)    # high[lbR]

    def in_range(cond_prev):
        bs = bars_since(cond_prev)
        with np.errstate(invalid="ignore"):
            return (bs >= p["range_lower"]) & (bs <= p["range_upper"])

    with np.errstate(invalid="ignore"):
        # Bullish: price Lower Low, CCI Higher Low
        osc_hl = (osc_s > value_when(pl, osc_s, 1)) & in_range(shift_by(pl, 1))
        price_ll = low_s < value_when(pl, low_s, 1)
        bull_all = pl & osc_hl & price_ll
        # Bearish: price Higher High, CCI Lower High
        osc_lh = (osc_s < value_when(ph, osc_s, 1)) & in_range(shift_by(ph, 1))
        price_hh = high_s > value_when(ph, high_s, 1)
        bear_all = ph & osc_lh & price_hh

        if p["use_ob_os"]:
            bull = bull_all & (osc_s <= p["os_level"])
            bear = bear_all & (osc_s >= p["ob_level"])
        else:
            bull, bear = bull_all, bear_all

    ema = ema_series(close, int(p["ema_len"]))
    if p["use_ema"]:
        up = close > ema
        dn = close < ema
    else:
        up = np.ones(len(close), dtype=bool)
        dn = np.ones(len(close), dtype=bool)

    return {"osc": osc, "ema": ema,
            "buy": np.asarray(bull & up, dtype=bool),
            "sell": np.asarray(bear & dn, dtype=bool)}


# ============================================================================
# 2) CONFIG + SHARED STATE
# ============================================================================

def default_pair(symbol, enabled=False):
    return {
        "enabled": enabled, "reverse": False,
        "symbol": symbol, "timeframe": "5m", "leverage": 25,
        # Trend filter
        "use_ema": True, "ema_len": 200,
        # CCI
        "cci_len": 20, "ob_level": 100.0, "os_level": -100.0, "use_ob_os": True,
        # Divergence
        "lbL": 5, "lbR": 5, "range_upper": 60, "range_lower": 5,
        # Risk
        "tp_pct": 3.0, "sl_pct": 1.5, "fee_pct": 0.1,
        # Compound / margin
        "initial_cap": 10.0, "use_compound": True, "compound_rate": 3.0,
        "fixed_margin_usdt": 0.0,   # >0 = use this fixed margin per trade instead of compound formula
    }


DEFAULT_GLOBAL = {
    "poll_seconds": 10,
    "position_mode": "hedge",   # "hedge" (Long/Short) or "oneway"
    "bot_start_date": datetime.now().strftime("%Y-%m-%d"),
}

DEFAULT_PAIRS = {
    "P1": default_pair("BEAT/USDT:USDT", enabled=True),
    "P2": default_pair("BTC/USDT:USDT"),
    "P3": default_pair("ETH/USDT:USDT"),
    "P4": default_pair("SOL/USDT:USDT"),
    "P5": default_pair("XRP/USDT:USDT"),
}

PAIR_SCHEMA = {
    "enabled": "bool", "reverse": "bool", "symbol": "str", "timeframe": "str", "leverage": "int",
    "use_ema": "bool", "ema_len": "int",
    "cci_len": "int", "ob_level": "float", "os_level": "float", "use_ob_os": "bool",
    "lbL": "int", "lbR": "int", "range_upper": "int", "range_lower": "int",
    "tp_pct": "float", "sl_pct": "float", "fee_pct": "float",
    "initial_cap": "float", "use_compound": "bool", "compound_rate": "float", "fixed_margin_usdt": "float",
}
GLOBAL_SCHEMA = {"poll_seconds": "int", "position_mode": "str", "bot_start_date": "str"}

PAIR_MIN = {"leverage": 1, "ema_len": 1, "cci_len": 2, "lbL": 1, "lbR": 1, "range_lower": 1, "range_upper": 2,
            "tp_pct": 0.01, "sl_pct": 0.01, "fee_pct": 0.0, "initial_cap": 0.01, "compound_rate": 0.0,
            "fixed_margin_usdt": 0.0}
PAIR_MAX = {"leverage": 125}


def empty_pair_state(initial_cap):
    return {
        "in_position": False, "pos_side": None, "signal_side": None, "reversed": False,
        "entry_price": None, "tp": None, "sl": None,
        "amount": None, "margin": None, "notional": None, "open_fee": 0.0,
        "entry_candle_ts": None, "bracket_ids": [], "brackets_ok": True,
        "last_checked_ts": None, "last_day": None, "daily_margin": None,
        "total_trades": 0, "wins": 0, "losses": 0, "total_fees": 0.0,
        "equity": initial_cap, "start_equity": initial_cap,
    }


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = {"global": dict(DEFAULT_GLOBAL),
                       "pairs": {pid: dict(DEFAULT_PAIRS[pid]) for pid in PAIR_IDS}}
        self.pair_state = {pid: empty_pair_state(self.config["pairs"][pid]["initial_cap"]) for pid in PAIR_IDS}
        self.order_locks = {pid: threading.Lock() for pid in PAIR_IDS}
        self.running = False
        self.connected = False
        self.trades = []
        self.logs = []
        self.market = {}
        self.balance = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            saved_cfg = data.get("config", {})
            self.config["global"].update({k: v for k, v in saved_cfg.get("global", {}).items() if k in GLOBAL_SCHEMA})
            for pid in PAIR_IDS:
                saved = saved_cfg.get("pairs", {}).get(pid, {})
                self.config["pairs"][pid].update({k: v for k, v in saved.items() if k in PAIR_SCHEMA})
                self.pair_state[pid].update(data.get("pair_state", {}).get(pid, {}))
            self.trades = data.get("trades", [])
            self.logs = data.get("logs", [])
        except Exception:
            pass

    def _save(self):
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"config": self.config, "pair_state": self.pair_state,
                           "trades": self.trades[:500], "logs": self.logs[-300:]}, f)
            os.replace(tmp, STATE_FILE)
        except Exception:
            pass

    def get_config(self):
        with self.lock:
            return copy.deepcopy(self.config)

    def apply_patch(self, g_patch: dict, pair_patch: dict):
        with self.lock:
            self.config["global"].update(g_patch)
            for pid, kv in pair_patch.items():
                self.config["pairs"][pid].update(kv)
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

    def set_market(self, pid, m):
        with self.lock:
            self.market[pid] = m

    def set_market_error(self, pid, msg):
        with self.lock:
            m = self.market.get(pid, {})
            m["status"] = "error"
            m["error"] = msg
            self.market[pid] = m

    def snapshot(self, chart_pair="P1"):
        with self.lock:
            pairs_out = {}
            for pid in PAIR_IDS:
                ps = dict(self.pair_state[pid])
                w, l = ps["wins"], ps["losses"]
                ps["win_rate"] = round(w / (w + l) * 100, 1) if (w + l) else 0.0
                ps["net_profit"] = round(ps["equity"] - ps["start_equity"], 4)
                ps["equity"] = round(ps["equity"], 4)
                ps["total_fees"] = round(ps["total_fees"], 4)
                m = self.market.get(pid, {})
                ps["live_price"] = m.get("live_price")
                ps["osc"] = m.get("osc_last")
                ps["status"] = m.get("status", "idle")
                ps["error"] = m.get("error")
                pairs_out[pid] = ps

            tot = sum(p["total_trades"] for p in pairs_out.values())
            wins = sum(p["wins"] for p in pairs_out.values())
            losses = sum(p["losses"] for p in pairs_out.values())
            net = sum(p["net_profit"] for p in pairs_out.values())
            fees = sum(p["total_fees"] for p in pairs_out.values())
            equity = sum(p["equity"] for p in pairs_out.values())

            cm = self.market.get(chart_pair, {})
            return {
                "config": copy.deepcopy(self.config),
                "pairs": pairs_out,
                "running": self.running,
                "connected": self.connected,
                "balance": self.balance,
                "combined": {
                    "total_trades": tot, "wins": wins, "losses": losses,
                    "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
                    "net_profit": round(net, 4), "fees": round(fees, 4), "equity": round(equity, 4),
                },
                "trades": self.trades[:80],
                "logs": self.logs[-100:],
                "chart_pair": chart_pair,
                "chart": {
                    "ohlcv": cm.get("ohlcv", []), "ema": cm.get("ema", []),
                    "buy_ts": cm.get("buy_ts", []), "sell_ts": cm.get("sell_ts", []),
                },
            }


state = BotState()

# ============================================================================
# 3) TRADING ENGINE - ccxt / BingX
# ============================================================================

_exchange = None
_markets_loaded = False
_ex_lock = threading.Lock()
_stop_flag = threading.Event()
_thread = None


def get_exchange():
    global _exchange
    with _ex_lock:
        if _exchange is None:
            _exchange = ccxt.bingx({
                "apiKey": BINGX_API_KEY, "secret": BINGX_SECRET_KEY,
                "enableRateLimit": True, "timeout": 15000,
                "options": {"defaultType": "swap"},
            })
        return _exchange


def ensure_markets(ex):
    global _markets_loaded
    with _ex_lock:
        if not _markets_loaded:
            ex.load_markets()
            _markets_loaded = True


def normalize_symbol(raw: str) -> str:
    """BEATUSDT / BEAT-USDT / BEAT/USDT  ->  BEAT/USDT:USDT (ccxt swap symbol)."""
    s = (raw or "").strip().upper().replace(" ", "")
    if not s or ":" in s:
        return s
    if "/" in s:
        base, quote = s.split("/", 1)
        return f"{base}/{quote}:{quote}"
    if "-" in s:
        base, quote = s.split("-", 1)
        return f"{base}/{quote}:{quote}"
    for q in ("USDT", "USDC"):
        if s.endswith(q) and len(s) > len(q):
            return f"{s[:-len(q)]}/{q}:{q}"
    return s


def verify_symbol(sym: str):
    """Returns (True,'') if the symbol is a real BingX swap market,
    (False,msg) if it is not, (None,msg) if it could not be verified."""
    if not sym:
        return False, "Symbol is empty"
    try:
        ex = get_exchange()
        ensure_markets(ex)
    except Exception as e:
        return None, f"Could not verify {sym} against BingX ({type(e).__name__}); saved unverified."
    mk = ex.markets.get(sym)
    if mk and mk.get("swap"):
        return True, ""
    base = sym.split("/")[0]
    sugg = [s for s, m in ex.markets.items() if m.get("swap") and s.endswith(":USDT") and base in s.split("/")[0]][:6]
    hint = (" Similar: " + ", ".join(sugg)) if sugg else ""
    return False, f"Symbol {sym} was not found on BingX USDT futures.{hint}"


def _is_trading_started(g: dict) -> bool:
    try:
        start_dt = datetime.strptime(g.get("bot_start_date", ""), "%Y-%m-%d").date()
    except Exception:
        return True
    return datetime.now().date() >= start_dt


def _pos_side_param(g: dict, side: str) -> str:
    return side if g.get("position_mode", "hedge") == "hedge" else "BOTH"


def _order_params(g: dict, side: str, extra=None, closing=False) -> dict:
    params = {"positionSide": _pos_side_param(g, side)}
    if closing and g.get("position_mode", "hedge") != "hedge":
        params["reduceOnly"] = True
    if extra:
        params.update(extra)
    return params


def _fmt_price(ex, symbol, price):
    try:
        return float(ex.price_to_precision(symbol, price))
    except Exception:
        return float(f"{price:.8g}")


def roll_day(pid):
    """Pine: on a new day daily_margin = (equity if compound else initial_cap) * rate%."""
    cfg = state.config["pairs"][pid]
    ps = state.pair_state[pid]
    today = datetime.now().strftime("%Y-%m-%d")
    with state.lock:
        if ps.get("last_day") != today or ps.get("daily_margin") is None:
            ps["last_day"] = today
            base = ps["equity"] if cfg["use_compound"] else cfg["initial_cap"]
            dm = base * cfg["compound_rate"] / 100.0
            if dm <= 0:
                dm = cfg["initial_cap"] * cfg["compound_rate"] / 100.0
            ps["daily_margin"] = dm
            state._save()


def current_margin(pid) -> float:
    cfg = state.config["pairs"][pid]
    if cfg["fixed_margin_usdt"] > 0:
        return float(cfg["fixed_margin_usdt"])
    return float(state.pair_state[pid]["daily_margin"] or 0.0)


def cancel_brackets(symbol, ids):
    if not ids:
        return
    ex = get_exchange()
    for oid in ids:
        try:
            ex.cancel_order(oid, symbol)
        except Exception:
            pass  # already filled / already gone


def fire_entry(pid, side, price, manual=False, signal_side=None, reversed_flag=False):
    """Open a real market order for this slot + real TP/SL bracket orders."""
    lk = state.order_locks[pid]
    if not lk.acquire(blocking=False):
        return False, "another order action for this slot is in progress"
    try:
        cfg = state.get_config()
        g, p = cfg["global"], cfg["pairs"][pid]
        symbol = p["symbol"]
        ps = state.pair_state[pid]
        with state.lock:
            if ps["in_position"]:
                return False, f"{pid} already has an open ticket"
        roll_day(pid)
        ex = get_exchange()
        try:
            ensure_markets(ex)
            leverage = int(p["leverage"])
            margin = current_margin(pid)
            if margin <= 0:
                raise ValueError("margin per trade is 0 (check compound rate / fixed margin)")
            amount = float(ex.amount_to_precision(symbol, margin * leverage / price))
            if amount <= 0:
                raise ValueError("order amount rounds to 0 — increase margin or leverage")
            mk = ex.markets.get(symbol) or {}
            min_amt = ((mk.get("limits") or {}).get("amount") or {}).get("min")
            if min_amt and amount < min_amt:
                raise ValueError(f"amount {amount} is below exchange minimum {min_amt} — increase margin/leverage")
            notional = amount * price
            is_long = side == "LONG"
            if is_long:
                sl = price * (1 - p["sl_pct"] / 100.0)
                tp = price * (1 + p["tp_pct"] / 100.0)
            else:
                sl = price * (1 + p["sl_pct"] / 100.0)
                tp = price * (1 - p["tp_pct"] / 100.0)
            sl = _fmt_price(ex, symbol, sl)
            tp = _fmt_price(ex, symbol, tp)

            try:
                ex.set_leverage(leverage, symbol, params={"side": _pos_side_param(g, side)})
            except Exception as e:
                state.add_log(f"⚠️ [{pid}] set_leverage warning: {e}")

            ex.create_order(symbol=symbol, type="market", side="buy" if is_long else "sell",
                            amount=amount, params=_order_params(g, side))

            close_side = "sell" if is_long else "buy"
            bracket_ids, brackets_ok = [], True
            for typ, trig in (("TAKE_PROFIT_MARKET", tp), ("STOP_MARKET", sl)):
                try:
                    o = ex.create_order(symbol=symbol, type=typ, side=close_side, amount=amount,
                                        params=_order_params(g, side, {"stopPrice": trig, "workingType": "MARK_PRICE"},
                                                             closing=True))
                    if o and o.get("id"):
                        bracket_ids.append(str(o["id"]))
                except Exception as be:
                    brackets_ok = False
                    state.add_log(f"🚨 [{pid}] {typ} bracket FAILED: {be} — bot will close this ticket itself when TP/SL is reached")

            open_fee = notional * p["fee_pct"] / 100.0 / 2.0
            last_ts = (state.market.get(pid) or {}).get("last_ts") or int(datetime.now().timestamp() * 1000)
            with state.lock:
                ps.update({
                    "in_position": True, "pos_side": side, "signal_side": signal_side or side,
                    "reversed": bool(reversed_flag), "entry_price": price, "tp": tp, "sl": sl,
                    "amount": amount, "margin": margin, "notional": notional, "open_fee": open_fee,
                    "entry_candle_ts": int(last_ts), "bracket_ids": bracket_ids, "brackets_ok": brackets_ok,
                })
                ps["equity"] -= open_fee
                ps["total_fees"] += open_fee
                state._save()

            tag = " [MANUAL]" if manual else ""
            rv = " ↺" if reversed_flag else ""
            state.add_trade({
                "id": uuid.uuid4().hex[:10], "pair": pid, "symbol": symbol,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_ms": int(datetime.now().timestamp() * 1000),
                "type": f"OPEN ({'BUY' if is_long else 'SELL'}) {pid}{rv}{tag}",
                "entry": price, "tp": tp, "sl": sl, "amount": amount,
                "pnl": None, "status": "OPEN", "side": side, "reversed": bool(reversed_flag),
            })
            state.add_log(f"[{pid}] ENTRY{tag}: {side}{' (reverse of ' + str(signal_side) + ' signal)' if reversed_flag else ''} "
                          f"{symbol} @ {price} | margin {round(margin, 4)} x{leverage} | TP {tp} / SL {sl}")
            return True, "order sent"
        except Exception as e:
            state.add_log(f"⚠️ [{pid}] ORDER FAILED ({side}): {e}")
            return False, str(e)
    finally:
        lk.release()


def close_ticket(pid, exit_price, reason, result):
    """Bookkeeping close (result = TP / SL / MANUAL). Mirrors the Pine backtest
    math: TP = +notional*TP%, SL = -notional*SL%, minus the closing fee."""
    ps = state.pair_state[pid]
    p = state.config["pairs"][pid]
    with state.lock:
        if not ps["in_position"]:
            return
        entry = ps["entry_price"] or 0.0
        side = ps["pos_side"]
        notional = ps["notional"] or 0.0
        open_fee = ps["open_fee"] or 0.0
        close_fee = notional * p["fee_pct"] / 100.0 / 2.0
        if result == "TP":
            gross = notional * p["tp_pct"] / 100.0
        elif result == "SL":
            gross = -notional * p["sl_pct"] / 100.0
        else:
            move = ((exit_price - entry) / entry) if side == "LONG" else ((entry - exit_price) / entry)
            gross = notional * move
        pnl = gross - close_fee
        net_trade = gross - close_fee - open_fee
        ps["equity"] += pnl
        ps["total_fees"] += close_fee
        ps["total_trades"] += 1
        is_win = (result == "TP") or (result == "MANUAL" and net_trade > 0)
        if is_win:
            ps["wins"] += 1
        else:
            ps["losses"] += 1
        for t in state.trades:
            if t.get("pair") == pid and t.get("status") == "OPEN":
                t["status"] = "WIN" if is_win else "LOSS"
                t["exit"] = exit_price
                t["exit_ms"] = int(datetime.now().timestamp() * 1000)
                t["pnl"] = round(net_trade, 4)
                t["reason"] = reason
                break
        ids = list(ps.get("bracket_ids") or [])
        ps.update({"in_position": False, "pos_side": None, "signal_side": None, "reversed": False,
                   "entry_price": None, "tp": None, "sl": None, "amount": None, "margin": None,
                   "notional": None, "open_fee": 0.0, "entry_candle_ts": None, "bracket_ids": [],
                   "brackets_ok": True})
        state._save()
    cancel_brackets(p["symbol"], ids)   # remove the leftover bracket order(s)
    state.add_log(f"[{pid}] Position closed ({reason}) @ ~{exit_price} | trade PnL {round(net_trade, 4)} USDT")


def exchange_close_position(pid):
    cfg = state.get_config()
    g, p = cfg["global"], cfg["pairs"][pid]
    ps = state.pair_state[pid]
    side, amount = ps["pos_side"], ps["amount"]
    ex = get_exchange()
    ex.create_order(symbol=p["symbol"], type="market", side="sell" if side == "LONG" else "buy",
                    amount=amount, params=_order_params(g, side, closing=True))


def software_exit(pid, exit_price, result):
    ps = state.pair_state[pid]
    if not ps.get("brackets_ok", True):
        # no real TP/SL on the exchange for this ticket -> we must close it ourselves
        try:
            exchange_close_position(pid)
        except Exception as e:
            state.add_log(f"⚠️ [{pid}] could not close on exchange ({result}): {e} — will retry")
            return
    close_ticket(pid, exit_price, result, result)


def check_exit(pid, ts_arr, high_arr, low_arr, live_price):
    """Check this slot's TP/SL against every candle AFTER the entry candle,
    then against the live price (Pine order: TP first for longs / shorts)."""
    ps = state.pair_state[pid]
    if not ps["in_position"]:
        return
    entry_ts = ps.get("entry_candle_ts") or 0
    tp, sl, side = ps["tp"], ps["sl"], ps["pos_side"]
    for i in range(len(ts_arr)):
        if ts_arr[i] <= entry_ts:
            continue
        hi, lo = high_arr[i], low_arr[i]
        if side == "LONG":
            if hi >= tp:
                return software_exit(pid, tp, "TP")
            if lo <= sl:
                return software_exit(pid, sl, "SL")
        else:
            if lo <= tp:
                return software_exit(pid, tp, "TP")
            if hi >= sl:
                return software_exit(pid, sl, "SL")
    if side == "LONG":
        if live_price >= tp:
            return software_exit(pid, tp, "TP")
        if live_price <= sl:
            return software_exit(pid, sl, "SL")
    else:
        if live_price <= tp:
            return software_exit(pid, tp, "TP")
        if live_price >= sl:
            return software_exit(pid, sl, "SL")


def open_trade_manual(pid, side):
    m = state.market.get(pid) or {}
    live_price = m.get("live_price")
    if not live_price:
        return False, "No live price yet for this slot (enable it and wait for data)"
    return fire_entry(pid, side, live_price, manual=True)


def close_trade_manual(pid):
    ps = state.pair_state[pid]
    if not ps["in_position"]:
        return False, "No open ticket for this slot"
    lk = state.order_locks[pid]
    if not lk.acquire(blocking=False):
        return False, "another order action for this slot is in progress"
    try:
        try:
            exchange_close_position(pid)
        except Exception as e:
            state.add_log(f"⚠️ [{pid}] Manual close failed: {e}")
            return False, str(e)
        exit_price = (state.market.get(pid) or {}).get("live_price") or ps["entry_price"]
        close_ticket(pid, exit_price, "MANUAL CLOSE", "MANUAL")
        return True, "closed"
    finally:
        lk.release()


def process_pair(pid, cfg, ex):
    g, p = cfg["global"], cfg["pairs"][pid]
    symbol = p["symbol"]
    ps = state.pair_state[pid]

    bars = ex.fetch_ohlcv(symbol, timeframe=p["timeframe"], limit=500)
    if not bars or len(bars) < 60:
        raise RuntimeError(f"not enough candles returned ({len(bars) if bars else 0})")
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ts_arr = df["timestamp"].values.astype("int64")
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    close_arr = df["close"].values.astype(float)
    live_price = float(close_arr[-1])
    idx = len(df) - 2  # last CLOSED candle

    sig = compute_pair_signals(df, p)
    osc_last = sig["osc"][idx]
    state.set_market(pid, {
        "ohlcv": df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist(),
        "ema": [float(x) for x in sig["ema"]] if p["use_ema"] else [],
        "buy_ts": [int(t) for t in ts_arr[sig["buy"]]],
        "sell_ts": [int(t) for t in ts_arr[sig["sell"]]],
        "live_price": live_price,
        "osc_last": None if np.isnan(osc_last) else round(float(osc_last), 2),
        "last_ts": int(ts_arr[-1]),
        "status": "ok", "error": None,
    })

    check_exit(pid, ts_arr, high_arr, low_arr, live_price)
    roll_day(pid)

    if state.running and p["enabled"] and _is_trading_started(g):
        bar_ts = int(ts_arr[idx])
        last = ps.get("last_checked_ts")
        if last is None:
            with state.lock:
                ps["last_checked_ts"] = bar_ts      # arm: only candles closing from now on can trigger
        elif bar_ts > last:
            with state.lock:
                ps["last_checked_ts"] = bar_ts
            if not ps["in_position"]:
                buy, sell = bool(sig["buy"][idx]), bool(sig["sell"][idx])
                if buy or sell:
                    sig_side = "LONG" if buy else "SHORT"   # Pine: buy has priority
                    side = sig_side
                    if p["reverse"]:
                        side = "SHORT" if sig_side == "LONG" else "LONG"
                    state.add_log(f"[{pid}] {'BUY' if buy else 'SELL'} signal on closed candle "
                                  f"{datetime.fromtimestamp(bar_ts / 1000).strftime('%H:%M')}"
                                  f"{' -> REVERSE -> ' + side if p['reverse'] else ''}")
                    fire_entry(pid, side, live_price, signal_side=sig_side, reversed_flag=bool(p["reverse"]))


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (market data streaming for enabled slots)")
    while not _stop_flag.is_set():
        cfg = state.get_config()
        any_ok = False
        try:
            ex = get_exchange()
            ensure_markets(ex)
            for pid in PAIR_IDS:
                p = cfg["pairs"][pid]
                ps = state.pair_state[pid]
                if not p["enabled"] and not ps["in_position"]:
                    continue    # disabled + flat -> nothing to do (open tickets are still managed)
                try:
                    process_pair(pid, cfg, ex)
                    any_ok = True
                except Exception as pe:
                    msg = f"{type(pe).__name__}: {pe}"
                    prev = (state.market.get(pid) or {}).get("error")
                    state.set_market_error(pid, msg)
                    if prev != msg:
                        state.add_log(f"⚠️ [{pid}] {p['symbol']} error: {msg}")
            try:
                bal = ex.fetch_balance()
                state.balance = float(bal.get("USDT", {}).get("total", 0.0) or 0.0)
            except Exception:
                pass
        except Exception as ex_err:
            state.add_log(f"⚠️ Loop error [{type(ex_err).__name__}]: {ex_err}")
            print(f"[LOOP] ERROR [{type(ex_err).__name__}]: {ex_err}", flush=True)
        state.connected = any_ok
        _stop_flag.wait(max(3, int(cfg["global"].get("poll_seconds", 10))))

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
    if on:
        with state.lock:
            for pid in PAIR_IDS:
                state.pair_state[pid]["last_checked_ts"] = None   # re-arm on the current candle
    state.add_log("🟢 RUN enabled - signals from newly closed candles will fire real orders" if on
                  else "⏸ RUN disabled - market data keeps refreshing, no new orders will fire")


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
<title>CCI Divergence · 5-Pair Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{ --bg:#0A0C12; --panel:#12151E; --panel-2:#171B27; --line:#232838; --text:#E7ECF3; --muted:#7C879C;
    --gold:#C9A24A; --gold-dim:#8A7130; --long:#3ED8A0; --short:#FF5C72; --radius:10px; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;-webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  .app{display:grid;grid-template-columns:350px 1fr;grid-template-rows:64px 1fr;height:100vh;}
  .brand{grid-column:1;grid-row:1;display:flex;align-items:center;gap:12px;padding:0 20px;border-bottom:1px solid var(--line);border-right:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:13px;}
  .brand h1{font-size:14px;letter-spacing:.04em;margin:0;font-weight:600;}
  .brand .sub{color:var(--muted);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;}
  .topbar{grid-column:2;grid-row:1;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--line);}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .dot.on{background:var(--long);box-shadow:0 0 8px var(--long);}
  .run-btn{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:9px 18px;border-radius:8px;font-family:inherit;font-weight:600;font-size:12px;letter-spacing:.04em;cursor:pointer;}
  .run-btn.active{background:var(--long);color:#062018;border-color:var(--long);}
  .run-btn.stopped{background:var(--short);color:#2a0509;border-color:var(--short);}
  .manual-group{display:flex;gap:6px;align-items:center;}
  .manual-btn{border:1px solid var(--line);background:var(--panel-2);color:var(--text);padding:5px 9px;border-radius:6px;font-family:inherit;font-weight:600;font-size:10.5px;cursor:pointer;}
  .manual-btn.long{color:var(--long);border-color:var(--long);}
  .manual-btn.short{color:var(--short);border-color:var(--short);}
  .manual-btn.reset{color:var(--muted);}
  .close-btn{background:transparent;border:1px solid var(--short);color:var(--short);padding:3px 9px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;}
  .control{grid-column:1;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:18px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:20px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .tabs{display:flex;gap:6px;margin-bottom:14px;}
  .tab{flex:1;background:var(--panel-2);border:1px solid var(--line);color:var(--muted);padding:8px 0;border-radius:8px;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:12px;cursor:pointer;border-bottom:3px solid var(--tc);}
  .tab.active{color:var(--text);background:#1d2231;}
  .pairform{display:none;} .pairform.active{display:block;}
  .fgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .span2{grid-column:1/3;}
  .field label{display:block;font-size:11px;color:var(--muted);margin-bottom:5px;line-height:1.3;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
  .save-btn{width:100%;margin-top:16px;background:var(--gold);color:#211a06;border:none;padding:11px;border-radius:8px;font-weight:700;font-size:12.5px;letter-spacing:.03em;cursor:pointer;font-family:inherit;}
  .warn{margin-top:16px;padding:10px 12px;background:#241412;border:1px solid #4a2620;border-radius:8px;font-size:11px;color:#f0a89c;line-height:1.5;}
  .main{grid-column:2;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;gap:10px;}
  .panel-title select{background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:4px 8px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:11px;}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:300px;}
  .pair-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;}
  .pair-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:13px 14px;border-top:3px solid var(--c);}
  .pair-card.off{opacity:.55;}
  .pair-card .title{font-weight:700;font-size:12.5px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;gap:6px;}
  .pair-card .row{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);padding:2.5px 0;font-family:'IBM Plex Mono',monospace;}
  .pair-card .row b{color:var(--text);}
  .chk{display:flex;align-items:center;gap:7px;font-size:11.5px;padding:3px 0;cursor:pointer;}
  .chk input{accent-color:var(--gold);width:15px;height:15px;}
  .chk.rev{color:#f3c969;}
  .st-err{font-size:10px;color:var(--short);margin-top:4px;word-break:break-word;font-family:'IBM Plex Mono',monospace;}
  .metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;}
  .metric-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;}
  .metric-card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .metric-card .val{font-family:'IBM Plex Mono',monospace;font-size:19px;font-weight:600;margin-top:6px;}
  .val.pos{color:var(--long);} .val.neg{color:var(--short);}
  .split{display:grid;grid-template-columns:1.6fr 1fr;gap:16px;}
  table{width:100%;border-collapse:collapse;font-size:11.5px;}
  th{text-align:left;color:var(--muted);font-weight:500;font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line);}
  td{padding:6px 8px;border-bottom:1px solid #191d29;font-family:'IBM Plex Mono',monospace;}
  .side-long{color:var(--long);} .side-short{color:var(--short);}
  .logbox{max-height:260px;overflow-y:auto;font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
  .logline{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #171a24;color:#B7C0D1;}
  .logline .t{color:var(--muted);}
  ::-webkit-scrollbar{width:8px;height:8px;} ::-webkit-scrollbar-thumb{background:#232838;border-radius:4px;}
  .badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;white-space:nowrap;}
  .badge-open{background:rgba(201,162,74,0.15);color:var(--gold);border:1px solid rgba(201,162,74,0.4);}
  .badge-win{background:rgba(62,216,160,0.15);color:var(--long);border:1px solid rgba(62,216,160,0.4);}
  .badge-loss{background:rgba(255,92,114,0.15);color:var(--short);border:1px solid rgba(255,92,114,0.4);}
  .tag{display:inline-block;padding:1px 6px;border-radius:5px;font-size:9.5px;font-weight:700;margin-right:4px;background:color-mix(in srgb,var(--tc) 20%,transparent);color:var(--tc);}
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
  <div class="brand"><div class="mark">CCI</div><div><h1>CCI DIVERGENCE · 5 PAIRS</h1><div class="sub">BingX Futures</div></div></div>
  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span></div>
    <div class="manual-group">
      <span class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></span>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>

  <div class="control">
    <h2>ตั้งค่ารายคู่เหรียญ</h2>
    <div class="tabs" id="tabs"></div>
    <div id="pairForms"></div>

    <h2>Global</h2>
    <div class="fgrid">
      <div class="field"><label>Position Mode (บัญชี BingX)</label>
        <select id="g_position_mode"><option value="hedge">Hedge (Long/Short แยก)</option><option value="oneway">One-way</option></select></div>
      <div class="field"><label>Poll (วินาที)</label><input id="g_poll_seconds" type="number" step="1" /></div>
      <div class="field span2"><label>วันเริ่มเทรด</label><input id="g_bot_start_date" type="date" /></div>
    </div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS (ทุกคู่เหรียญ)</button>
    <div class="warn">ติ๊ก "เปิดใช้งาน" / "เทรดสวน" ได้ที่การ์ดของแต่ละคู่ด้านขวา (บันทึกทันที) — อย่าตั้ง 2 สล็อตที่เปิดใช้งานเป็นเหรียญเดียวกัน เพราะ BingX จะรวมเป็นโพซิชั่นเดียว</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart <select id="chartPairSel"></select></span><span class="price mono" id="livePrice">--</span></div>
      <div id="priceChart"></div>
    </div>

    <div class="pair-grid" id="pairGrid"></div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">Win Rate (รวม)</div><div class="val" id="m_winrate">--</div></div>
      <div class="metric-card"><div class="lbl">Total Trades</div><div class="val" id="m_total">--</div></div>
      <div class="metric-card"><div class="lbl">Net Profit (sim)</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">Fees (sim)</div><div class="val" id="m_fees">--</div></div>
      <div class="metric-card"><div class="lbl">Equity รวม (sim)</div><div class="val" id="m_equity">--</div></div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table><thead><tr><th></th><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>PnL</th><th>Status</th><th>Action</th></tr></thead><tbody id="ordersBody"></tbody></table>
      </div>
      <div class="panel"><div class="panel-title">Bot Activity Log</div><div class="logbox" id="logBody"></div></div>
    </div>
  </div>
</div>

<script>
const PAIRS = ['P1','P2','P3','P4','P5'];
const COLORS = {P1:'#3ED8A0',P2:'#4CC9F0',P3:'#D65DB1',P4:'#C9A24A',P5:'#8E7CFF'};
const TF_SEC = {'1m':60,'3m':180,'5m':300,'15m':900,'30m':1800,'1h':3600,'2h':7200,'4h':14400,'1d':86400};

const PAIR_FIELDS = [
  {sec:'Market', items:[
    {k:'symbol', l:'Symbol (เช่น BEAT/USDT:USDT หรือพิมพ์ BEATUSDT)', t:'text', span:true},
    {k:'timeframe', l:'Timeframe', t:'select', o:['1m','3m','5m','15m','30m','1h','2h','4h','1d']},
    {k:'leverage', l:'Leverage (x)', t:'number', s:1}]},
  {sec:'Trend Filter', items:[
    {k:'use_ema', l:'กรองเทรนด์จาก EMA', t:'bool'},
    {k:'ema_len', l:'EMA Length', t:'number', s:1}]},
  {sec:'CCI Settings', items:[
    {k:'cci_len', l:'CCI Length', t:'number', s:1},
    {k:'use_ob_os', l:'เข้าเฉพาะโซน OB/OS', t:'bool'},
    {k:'ob_level', l:'Overbought Level', t:'number', s:'any'},
    {k:'os_level', l:'Oversold Level', t:'number', s:'any'}]},
  {sec:'Divergence Settings', items:[
    {k:'lbL', l:'Pivot Lookback Left', t:'number', s:1},
    {k:'lbR', l:'Pivot Lookback Right', t:'number', s:1},
    {k:'range_upper', l:'ระยะห่างสูงสุดระหว่าง Pivot (bars)', t:'number', s:1},
    {k:'range_lower', l:'ระยะห่างต่ำสุดระหว่าง Pivot (bars)', t:'number', s:1}]},
  {sec:'Risk Management', items:[
    {k:'tp_pct', l:'Take Profit (% ราคา)', t:'number', s:'any'},
    {k:'sl_pct', l:'Stop Loss (% ราคา)', t:'number', s:'any'},
    {k:'fee_pct', l:'Fee % (Open+Close)', t:'number', s:'any'}]},
  {sec:'Compound / Margin', items:[
    {k:'initial_cap', l:'ทุนเริ่มต้นจำลอง (USD)', t:'number', s:'any'},
    {k:'use_compound', l:'ทบทุนตามพอร์ตโต', t:'bool'},
    {k:'compound_rate', l:'อัตราขนาดไม้ต่อวัน (% ของพอร์ต)', t:'number', s:'any'},
    {k:'fixed_margin_usdt', l:'Margin คงที่ต่อไม้ (USDT, 0 = ใช้สูตรทบทุน)', t:'number', s:'any'}]},
];

const priceChart = LightweightCharts.createChart(document.getElementById('priceChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const candleSeries = priceChart.addCandlestickSeries({upColor:'#3ED8A0', downColor:'#FF5C72', borderVisible:false, wickUpColor:'#3ED8A0', wickDownColor:'#FF5C72'});
const emaSeries = priceChart.addLineSeries({color:'#F5D90A', lineWidth:1, priceLineVisible:false, lastValueVisible:false});

let chartPair = 'P1', lastChartPair = null, activeTab = 'P1';
let cfgLoadedOnce = false, lastSeenTradeId = null, firstRender = true;
const $ = id => document.getElementById(id);
const esc = s => String(s??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function fieldHTML(pid, f){
  const id = `cfg_${pid}_${f.k}`;
  const span = f.span ? ' span2' : '';
  if(f.t==='bool') return `<div class="field${span}"><label>${f.l}</label><select id="${id}"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>`;
  if(f.t==='select') return `<div class="field${span}"><label>${f.l}</label><select id="${id}">${f.o.map(o=>`<option value="${o}">${o}</option>`).join('')}</select></div>`;
  return `<div class="field${span}"><label>${f.l}</label><input id="${id}" type="${f.t==='text'?'text':'number'}" step="${f.s||'any'}" /></div>`;
}

function buildForms(){
  $('tabs').innerHTML = PAIRS.map(pid=>`<button class="tab${pid===activeTab?' active':''}" data-p="${pid}" style="--tc:${COLORS[pid]}">${pid}</button>`).join('');
  $('pairForms').innerHTML = PAIRS.map(pid=>`
    <div class="pairform${pid===activeTab?' active':''}" id="form_${pid}">
      ${PAIR_FIELDS.map(sec=>`<h2>${sec.sec}</h2><div class="fgrid">${sec.items.map(f=>fieldHTML(pid,f)).join('')}</div>`).join('')}
    </div>`).join('');
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click', ()=>{
    activeTab = b.dataset.p;
    document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active', x.dataset.p===activeTab));
    PAIRS.forEach(pid=>$('form_'+pid).classList.toggle('active', pid===activeTab));
  }));
}

function buildCards(){
  $('pairGrid').innerHTML = PAIRS.map(pid=>`
    <div class="pair-card off" id="card_${pid}" style="--c:${COLORS[pid]}">
      <div class="title"><span>${pid} · <span id="sym_${pid}">--</span></span><span id="pos_${pid}">--</span></div>
      <label class="chk"><input type="checkbox" id="en_${pid}"> เปิดใช้งานคู่นี้</label>
      <label class="chk rev"><input type="checkbox" id="rv_${pid}"> เทรดสวน (Buy→Sell / Sell→Buy)</label>
      <div class="row"><span>ราคา</span><b id="px_${pid}">--</b></div>
      <div class="row"><span>CCI</span><b id="cci_${pid}">--</b></div>
      <div class="row"><span>ออเดอร์</span><b id="tot_${pid}">--</b></div>
      <div class="row"><span>ชนะ/แพ้</span><b id="wl_${pid}">--</b></div>
      <div class="row"><span>Equity</span><b id="eq_${pid}">--</b></div>
      <div class="row"><span>กำไรสุทธิ</span><b id="net_${pid}">--</b></div>
      <div class="st-err" id="err_${pid}"></div>
      <div class="manual-group" style="margin-top:8px;flex-wrap:wrap">
        <button class="manual-btn long" data-p="${pid}" data-side="LONG">▲ LONG</button>
        <button class="manual-btn short" data-p="${pid}" data-side="SHORT">▼ SHORT</button>
        <button class="manual-btn reset" data-reset="${pid}" title="รีเซ็ตสถิติคู่นี้">↺ สถิติ</button>
      </div>
    </div>`).join('');
  PAIRS.forEach(pid=>{
    $('en_'+pid).addEventListener('change', e=>postConfig({pairs:{[pid]:{enabled:e.target.checked}}}));
    $('rv_'+pid).addEventListener('change', e=>postConfig({pairs:{[pid]:{reverse:e.target.checked}}}));
  });
  document.querySelectorAll('.manual-btn[data-side]').forEach(btn=>btn.addEventListener('click', async ()=>{
    const pid=btn.dataset.p, side=btn.dataset.side;
    if(!confirm(`เปิดออเดอร์ ${side} ด้วยมือ (${pid}) ยิงจริงทันที แน่ใจไหม?`)) return;
    const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pair:pid, side})});
    const j = await res.json();
    if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: '+(j.message||j.error||'unknown error'));
  }));
  document.querySelectorAll('.manual-btn[data-reset]').forEach(btn=>btn.addEventListener('click', async ()=>{
    const pid=btn.dataset.reset;
    if(!confirm(`รีเซ็ตสถิติ/Equity ของ ${pid} ?`)) return;
    const res = await fetch('/api/reset-pair', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pair:pid})});
    const j = await res.json();
    if(!j.ok) alert(j.message||'reset failed');
  }));
}

function buildChartSelect(){
  $('chartPairSel').innerHTML = PAIRS.map(pid=>`<option value="${pid}">${pid}</option>`).join('');
  $('chartPairSel').addEventListener('change', e=>{ chartPair = e.target.value; poll(true); });
}

async function postConfig(patch){
  try{
    const res = await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(patch)});
    const j = await res.json();
    if(j.warnings && j.warnings.length) alert(j.warnings.join('\n\n'));
    return j;
  }catch(e){ alert('Save failed: '+e); }
}

function populateForm(cfg){
  PAIRS.forEach(pid=>{
    const pc = cfg.pairs[pid];
    PAIR_FIELDS.forEach(sec=>sec.items.forEach(f=>{
      const el = $(`cfg_${pid}_${f.k}`); if(el && pc[f.k]!==undefined) el.value = String(pc[f.k]);
    }));
  });
  $('g_position_mode').value = cfg.global.position_mode;
  $('g_poll_seconds').value = cfg.global.poll_seconds;
  $('g_bot_start_date').value = cfg.global.bot_start_date;
}

function collectPatch(){
  const pairs = {};
  PAIRS.forEach(pid=>{
    pairs[pid] = {};
    PAIR_FIELDS.forEach(sec=>sec.items.forEach(f=>{ const el=$(`cfg_${pid}_${f.k}`); if(el) pairs[pid][f.k]=el.value; }));
  });
  return {global:{position_mode:$('g_position_mode').value, poll_seconds:$('g_poll_seconds').value, bot_start_date:$('g_bot_start_date').value}, pairs};
}

function statusBadge(t){
  if(t.status==='OPEN') return `<span class="badge badge-open">🔵 OPEN</span>`;
  if(t.status==='WIN')  return `<span class="badge badge-win">✅ WIN</span>`;
  if(t.status==='LOSS') return `<span class="badge badge-loss">❌ LOSS</span>`;
  return esc(t.status);
}
function pairTag(pid){ return pid ? `<span class="tag" style="--tc:${COLORS[pid]||'#999'}">${esc(pid)}</span>` : ''; }

function showToast(t){
  const isWin = t.status==='WIN';
  const el = document.createElement('div');
  el.className = 'toast ' + (isWin?'win':'loss');
  el.innerHTML = `<div class="ttitle">${pairTag(t.pair)} ${isWin?'🎯 TP HIT':'🛑 SL / LOSS'}</div><div class="tbody">${esc(t.side)} @ ${esc(t.exit)} · PnL ${(t.pnl>=0?'+':'')+esc(t.pnl)} USDT</div>`;
  $('toastContainer').appendChild(el);
  setTimeout(()=>el.remove(), 6200);
}
function maybeShowToast(trades){
  if(!trades || !trades.length) return;
  const newest = trades.find(t=>t.status==='WIN'||t.status==='LOSS');
  if(!newest) return;
  if(firstRender){ lastSeenTradeId = newest.id; firstRender = false; return; }
  if(newest.id === lastSeenTradeId) return;
  lastSeenTradeId = newest.id;
  showToast(newest);
}

let polling = false;
async function poll(once){
  if(polling && !once) return;
  try{ const res = await fetch('/api/status?pair='+chartPair); render(await res.json()); }catch(e){ console.error(e); }
  if(!once) setTimeout(poll, 4000);
}

function updateCard(pid, pc, ps){
  $('sym_'+pid).textContent = (pc.symbol||'').replace(':USDT','');
  $('en_'+pid).checked = !!pc.enabled;
  $('rv_'+pid).checked = !!pc.reverse;
  $('card_'+pid).classList.toggle('off', !pc.enabled && !ps.in_position);
  $('pos_'+pid).textContent = ps.in_position ? ('🟢 '+ps.pos_side+(ps.reversed?' ↺':'')) : (pc.enabled ? '⚪ ว่าง' : '⏸ ปิด');
  $('px_'+pid).textContent = ps.live_price ?? '--';
  $('cci_'+pid).textContent = ps.osc ?? '--';
  $('tot_'+pid).textContent = ps.total_trades;
  $('wl_'+pid).textContent = `${ps.wins} / ${ps.losses} (${ps.win_rate}%)`;
  $('eq_'+pid).textContent = ps.equity.toFixed(2);
  const net = $('net_'+pid);
  net.textContent = (ps.net_profit>=0?'+':'') + ps.net_profit.toFixed(2);
  net.style.color = ps.net_profit>=0 ? 'var(--long)' : 'var(--short)';
  $('err_'+pid).textContent = ps.status==='error' ? ('⚠ '+(ps.error||'error')) : '';
}

function renderChart(data){
  const cfg = data.config.pairs[data.chart_pair];
  const tf = TF_SEC[cfg.timeframe] || 300;
  const snap = ms => Math.floor(ms/1000/tf)*tf;
  const ch = data.chart;
  if(!ch.ohlcv.length){ candleSeries.setData([]); emaSeries.setData([]); candleSeries.setMarkers([]); return; }
  const bars = ch.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]}));
  candleSeries.setData(bars);
  emaSeries.setData(ch.ema.length ? ch.ema.map((v,i)=>({time:bars[i].time, value:v})) : []);
  const first = bars[0].time;
  const markers = [];
  ch.buy_ts.forEach(ts=>markers.push({time:snap(ts), position:'belowBar', color:'#3ED8A0', shape:'circle', text:'Bull'}));
  ch.sell_ts.forEach(ts=>markers.push({time:snap(ts), position:'aboveBar', color:'#FF5C72', shape:'circle', text:'Bear'}));
  data.trades.filter(t=>t.pair===data.chart_pair).forEach(t=>{
    markers.push({time:snap(t.entry_ms), position:t.side==='LONG'?'belowBar':'aboveBar', color:'#F5D90A', shape:t.side==='LONG'?'arrowUp':'arrowDown', text:(t.reversed?'↺ ':'')+t.side});
    if((t.status==='WIN'||t.status==='LOSS') && t.exit_ms)
      markers.push({time:snap(t.exit_ms), position:t.status==='WIN'?'aboveBar':'belowBar', color:t.status==='WIN'?'#3ED8A0':'#FF5C72', shape:t.status==='WIN'?'circle':'square', text:t.status==='WIN'?'TP':'SL'});
  });
  candleSeries.setMarkers(markers.filter(m=>m.time>=first).sort((a,b)=>a.time-b.time));
  if(lastChartPair!==data.chart_pair){ priceChart.timeScale().fitContent(); lastChartPair = data.chart_pair; }
}

function render(data){
  $('connDot').className = 'dot ' + (data.connected?'on':'');
  $('connText').textContent = data.connected ? 'CONNECTED · BingX' : 'CONNECTING…';
  $('balanceVal').textContent = '$' + (data.balance||0).toFixed(2);
  const runBtn = $('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running?'active':'stopped');

  PAIRS.forEach(pid=>updateCard(pid, data.config.pairs[pid], data.pairs[pid]));
  const sel = $('chartPairSel');
  PAIRS.forEach((pid,i)=>{ sel.options[i].text = pid+' · '+(data.config.pairs[pid].symbol||'').replace(':USDT',''); });
  sel.value = chartPair;
  const lp = data.pairs[chartPair].live_price;
  $('livePrice').textContent = lp ? ('$'+lp) : '--';

  const cb = data.combined;
  $('m_winrate').textContent = cb.win_rate + '%';
  $('m_total').textContent = cb.total_trades;
  const pnlEl = $('m_pnl');
  pnlEl.textContent = (cb.net_profit>=0?'+':'') + cb.net_profit + ' USDT';
  pnlEl.className = 'val ' + (cb.net_profit>=0?'pos':'neg');
  $('m_fees').textContent = cb.fees + ' USDT';
  $('m_equity').textContent = cb.equity + ' USDT';

  renderChart(data);

  const ob = $('ordersBody');
  ob.innerHTML = data.trades.map(t=>`
    <tr>
      <td>${pairTag(t.pair)}</td>
      <td>${esc(t.time)}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${esc(t.type)}</td>
      <td>${esc(t.entry)}</td><td>${esc(t.tp)}</td><td>${esc(t.sl)}</td>
      <td>${t.pnl==null?'':esc(t.pnl)}</td>
      <td>${statusBadge(t)}</td>
      <td>${t.status==='OPEN' ? `<button class="close-btn" data-p="${esc(t.pair)}">CLOSE</button>` : ''}</td>
    </tr>`).join('');
  ob.querySelectorAll('.close-btn').forEach(btn=>btn.addEventListener('click', async ()=>{
    btn.disabled=true; btn.textContent='...';
    const res = await fetch('/api/close-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pair: btn.dataset.p})});
    const j = await res.json();
    if(!j.ok) alert('Close failed: '+(j.message||j.error||'unknown error'));
  }));

  maybeShowToast(data.trades);
  $('logBody').innerHTML = data.logs.slice().reverse().map(l=>`<div class="logline"><span class="t">${esc(l.time)}</span><span>${esc(l.text)}</span></div>`).join('');

  if(!cfgLoadedOnce){ populateForm(data.config); cfgLoadedOnce = true; }
}

$('runBtn').addEventListener('click', async ()=>{
  const willRun = $('runBtn').textContent.includes('RUN');
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({running: willRun})});
});

$('saveCfgBtn').addEventListener('click', async ()=>{
  const j = await postConfig(collectPatch());
  if(j && j.ok){ cfgLoadedOnce = false; }   // reload the form with the values the server actually accepted
});

buildForms(); buildCards(); buildChartSelect();
poll();
</script>
</body>
</html>
"""


def clean_values(raw: dict, schema: dict) -> dict:
    out = {}
    for k, v in raw.items():
        t = schema.get(k)
        if t is None:
            continue
        try:
            if t == "int":
                f = float(v)
                if not math.isfinite(f):
                    continue
                out[k] = int(f)
            elif t == "float":
                f = float(v)
                if not math.isfinite(f):
                    continue
                out[k] = f
            elif t == "bool":
                out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes", "on")
            else:
                out[k] = str(v).strip()
        except (TypeError, ValueError):
            continue
    return out


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "connected": state.connected, "running": state.running})


@app.route("/api/status")
def api_status():
    pair = request.args.get("pair", "P1")
    if pair not in PAIR_IDS:
        pair = "P1"
    return jsonify(state.snapshot(pair))


@app.route("/api/config", methods=["POST"])
def api_config():
    body = request.get_json(force=True) or {}
    warnings = []
    cur = state.get_config()

    g_clean = clean_values(body.get("global") or {}, GLOBAL_SCHEMA)
    if "position_mode" in g_clean and g_clean["position_mode"] not in ("hedge", "oneway"):
        del g_clean["position_mode"]
    if "poll_seconds" in g_clean:
        g_clean["poll_seconds"] = max(3, g_clean["poll_seconds"])
    if "bot_start_date" in g_clean:
        try:
            datetime.strptime(g_clean["bot_start_date"], "%Y-%m-%d")
        except ValueError:
            del g_clean["bot_start_date"]

    pair_clean = {}
    for pid, raw in (body.get("pairs") or {}).items():
        if pid not in PAIR_IDS or not isinstance(raw, dict):
            continue
        c = clean_values(raw, PAIR_SCHEMA)
        ps = state.pair_state[pid]

        if "symbol" in c:
            new_sym = normalize_symbol(c["symbol"])
            if new_sym == cur["pairs"][pid]["symbol"]:
                c["symbol"] = new_sym
            elif ps["in_position"]:
                warnings.append(f"[{pid}] Symbol not changed: close the open ticket first.")
                del c["symbol"]
            else:
                ok, msg = verify_symbol(new_sym)
                if ok is False:
                    warnings.append(f"[{pid}] {msg}")
                    del c["symbol"]
                else:
                    c["symbol"] = new_sym
                    if ok is None:
                        warnings.append(f"[{pid}] {msg}")
        if "timeframe" in c and c["timeframe"] not in TIMEFRAMES:
            del c["timeframe"]
        for k, mn in PAIR_MIN.items():
            if k in c:
                c[k] = max(c[k], mn)
        for k, mx in PAIR_MAX.items():
            if k in c:
                c[k] = min(c[k], mx)
        merged = {**cur["pairs"][pid], **c}
        if merged["range_upper"] < merged["range_lower"]:
            c["range_upper"] = merged["range_lower"]
        pair_clean[pid] = c

    state.apply_patch(g_clean, pair_clean)

    # re-arm / reset market data where it matters
    for pid, c in pair_clean.items():
        old = cur["pairs"][pid]
        with state.lock:
            if (("symbol" in c and c["symbol"] != old["symbol"]) or
                    ("timeframe" in c and c["timeframe"] != old["timeframe"])):
                state.pair_state[pid]["last_checked_ts"] = None
                state.market.pop(pid, None)
            if c.get("enabled") and not old["enabled"]:
                state.pair_state[pid]["last_checked_ts"] = None
        if c:
            state.add_log(f"[{pid}] settings updated: {c}")

    # duplicate-symbol warning
    new_cfg = state.get_config()
    seen = {}
    for pid in PAIR_IDS:
        pc = new_cfg["pairs"][pid]
        if pc["enabled"]:
            seen.setdefault(pc["symbol"], []).append(pid)
    for sym, pids in seen.items():
        if len(pids) > 1:
            warnings.append(f"Slots {', '.join(pids)} are all enabled on {sym}. BingX merges them into ONE real position — use different symbols.")

    return jsonify({"ok": True, "warnings": warnings, "config": new_cfg})


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    set_running(bool(payload.get("running")))
    return jsonify({"running": state.running})


@app.route("/api/manual-order", methods=["POST"])
def api_manual_order():
    payload = request.get_json(force=True) or {}
    pid = payload.get("pair")
    side = (payload.get("side") or "").upper()
    if pid not in PAIR_IDS or side not in ("LONG", "SHORT"):
        return jsonify({"ok": False, "error": "invalid pair/side"}), 400
    ok, msg = open_trade_manual(pid, side)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/close-order", methods=["POST"])
def api_close_order():
    payload = request.get_json(force=True) or {}
    pid = payload.get("pair")
    if pid not in PAIR_IDS:
        return jsonify({"ok": False, "error": "invalid pair"}), 400
    ok, msg = close_trade_manual(pid)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/reset-pair", methods=["POST"])
def api_reset_pair():
    payload = request.get_json(force=True) or {}
    pid = payload.get("pair")
    if pid not in PAIR_IDS:
        return jsonify({"ok": False, "message": "invalid pair"}), 400
    with state.lock:
        if state.pair_state[pid]["in_position"]:
            return jsonify({"ok": False, "message": "Close the open ticket first."})
        cap = state.config["pairs"][pid]["initial_cap"]
        state.pair_state[pid] = empty_pair_state(cap)
        state._save()
    state.add_log(f"[{pid}] stats / equity reset to {cap} USDT")
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
