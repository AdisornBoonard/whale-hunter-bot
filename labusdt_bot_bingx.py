"""
labusdt_bot_bingx.py
--------------------
Single-file drop-in bot for BingX Futures LABUSDT (LAB-USDT:USDT) trading the
RSI Momentum Divergence Zones signal, with a built-in web dashboard.

Same deploy pattern as your old Whale Hunter bot: one file, one process,
PORT from the environment, a background thread doing the work while a web
server keeps the service alive/health-checkable online.

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python labusdt_bot_bingx.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT labusdt_bot_bingx:app

⚠️ MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders
firing on the same signal.

INDICATOR NOTES (see chat for full detail)
-------------------------------------------
- No classic repaint (no lookahead / no future-data leak).
- BUT a pivot is only confirmed `right_bars` bars after it happens - that
  confirmation lag is real, not a bug. Low right_bars / short rsi_length =
  fast but noisier signals. Tune from the dashboard (Indicator Settings).
"""

import os
import json
import threading
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
# 1) INDICATOR - Python port of the Pine Script "RSI Momentum Divergence Zones"
# ============================================================================

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


def rsi_of_momentum(close: np.ndarray, mom_length: int, rsi_length: int):
    momentum = np.full(len(close), np.nan)
    momentum[mom_length:] = close[mom_length:] - close[:-mom_length]

    change = np.diff(momentum, prepend=np.nan)
    up = np.where(change > 0, change, 0.0)
    down = np.where(change < 0, -change, 0.0)
    up[np.isnan(change)] = np.nan
    down[np.isnan(change)] = np.nan

    roll_up = rma(np.nan_to_num(up, nan=0.0), rsi_length)
    roll_down = rma(np.nan_to_num(down, nan=0.0), rsi_length)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = roll_up / roll_down
        rsi = 100 - 100 / (1 + rs)
        rsi[roll_down == 0] = 100.0
        rsi[(roll_up == 0) & (roll_down == 0)] = np.nan

    return momentum, rsi


def find_pivots(series: np.ndarray, left: int, right: int):
    """Pivot low/high at their own index j, confirmable once j+right <= n-1."""
    n = len(series)
    is_low = np.zeros(n, dtype=bool)
    is_high = np.zeros(n, dtype=bool)
    for j in range(left, n - right):
        window = series[j - left: j + right + 1]
        if np.all(np.isnan(window)):
            continue
        if series[j] == np.nanmin(window):
            is_low[j] = True
        if series[j] == np.nanmax(window):
            is_high[j] = True
    return is_low, is_high


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


def bars_since(cond: np.ndarray):
    n = len(cond)
    out = np.full(n, np.nan)
    last_true = None
    for i in range(n):
        if cond[i]:
            last_true = i
        if last_true is not None:
            out[i] = i - last_true
    return out


def value_when(cond: np.ndarray, source: np.ndarray, occurrence: int):
    n = len(cond)
    out = np.full(n, np.nan)
    hist = []
    for i in range(n):
        if cond[i]:
            hist.append(i)
        if len(hist) > occurrence:
            out[i] = source[hist[-1 - occurrence]]
    return out


def compute_divergence(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    _momentum, rsi = rsi_of_momentum(close, cfg["mom_length"], cfg["rsi_length"])
    df["rsi"] = rsi

    is_low, is_high = find_pivots(rsi, cfg["left_bars"], cfg["right_bars"])
    right = cfg["right_bars"]

    found_pl = shift_by(is_low, right, fill=False)
    found_ph = shift_by(is_high, right, fill=False)
    rsi_right = shift_by(rsi, right, fill=np.nan)
    high_level = shift_by(high, right, fill=np.nan)
    low_level = shift_by(low, right, fill=np.nan)

    bars_since_pl = bars_since(found_pl)
    bars_since_ph = bars_since(found_ph)
    min_r, max_r = cfg["min_bars_range"], cfg["max_bars_range"]
    in_range_pl = (bars_since_pl >= min_r) & (bars_since_pl <= max_r)
    in_range_ph = (bars_since_ph >= min_r) & (bars_since_ph <= max_r)
    in_range_pl_prev = np.roll(in_range_pl, 1); in_range_pl_prev[0] = False
    in_range_ph_prev = np.roll(in_range_ph, 1); in_range_ph_prev[0] = False

    rsi_prev_pivot = value_when(found_pl, rsi_right, 1)
    low_prev_pivot = value_when(found_pl, low_level, 1)
    rsi_hl = (rsi_right > rsi_prev_pivot) & in_range_pl_prev
    price_ll = low_level < low_prev_pivot
    bull_div = found_pl & rsi_hl & price_ll

    rsi_prev_pivot_h = value_when(found_ph, rsi_right, 1)
    high_prev_pivot = value_when(found_ph, high_level, 1)
    rsi_lh = (rsi_right < rsi_prev_pivot_h) & in_range_ph_prev
    price_hh = high_level > high_prev_pivot
    bear_div = found_ph & rsi_lh & price_hh

    df["found_pl"] = found_pl
    df["found_ph"] = found_ph
    df["bull_div"] = np.nan_to_num(bull_div, nan=0).astype(bool)
    df["bear_div"] = np.nan_to_num(bear_div, nan=0).astype(bool)
    return df


def latest_signal(df: pd.DataFrame):
    if len(df) == 0:
        return "HOLD", None
    last = df.iloc[-1]
    if bool(last["bull_div"]):
        return "LONG", int(last["timestamp"])
    if bool(last["bear_div"]):
        return "SHORT", int(last["timestamp"])
    return "HOLD", int(last["timestamp"])


# ============================================================================
# 2) SHARED STATE (config / trades / logs), one lock, thread-safe
# ============================================================================

DEFAULT_CONFIG = {
    "symbol": "LAB-USDT:USDT",
    "timeframe": "1m",
    "initial_bet": 1.0,
    "daily_add": 3.0,
    "bot_start_date": "2026-07-07",
    "leverage": 20,
    "tp_percent": 1.0,
    "sl_percent": 0.5,
    "max_tickets": 10,
    "mom_length": 10,
    "rsi_length": 4,
    "left_bars": 4,
    "right_bars": 1,
    "min_bars_range": 5,
    "max_bars_range": 50,
    "poll_seconds": 10,
}


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(DEFAULT_CONFIG)
        self.running = False
        self.connected = False
        self.trades = []
        self.logs = []
        self.ohlcv = []
        self.indicator = {"rsi": [], "timestamps": [], "bull": [], "bear": []}
        self.live_price = None
        self.balance = 0.0
        self.active_long = 0
        self.active_short = 0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self.config.update(data.get("config", {}))
                self.trades = data.get("trades", [])
                self.logs = data.get("logs", [])
            except Exception:
                pass

    def _save(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"config": self.config, "trades": self.trades[:500],
                           "logs": self.logs[-300:]}, f)
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
# 3) TRADING ENGINE - ccxt / BingX (LABUSDT), TP/SL, ticket counting, margin
# ============================================================================

_exchange = None
_last_fired_ts = {"LONG": 0, "SHORT": 0}
_stop_flag = threading.Event()
_thread = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bingx({
            "apiKey": BINGX_API_KEY,
            "secret": BINGX_SECRET_KEY,
            "enableRateLimit": True,
            "timeout": 15000,  # 15s - fail fast instead of hanging forever on a stuck connection
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


def fire_execution_order(symbol, side, entry_price, margin_size, leverage, tp_pct, sl_pct):
    ex = get_exchange()
    try:
        contract_amount = round((margin_size * leverage) / entry_price, 4)
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_pct / 100), 4)
            sl_price = round(entry_price * (1 - sl_pct / 100), 4)
            order_side = "buy"
        else:
            tp_price = round(entry_price * (1 - tp_pct / 100), 4)
            sl_price = round(entry_price * (1 + sl_pct / 100), 4)
            order_side = "sell"

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

        state.add_trade({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": f"OPEN ({'BUY' if side == 'LONG' else 'SELL'})",
            "entry": entry_price, "tp": tp_price, "sl": sl_price,
            "pnl": None, "status": "OPEN", "side": side,
        })
        state.add_log(f"ENTRY EXECUTED: {side} @ {entry_price} | TP {tp_price} / SL {sl_price}")
    except Exception as e:
        state.add_log(f"⚠️ ORDER FAILED ({side}): {e}")


def _compute_margin(cfg):
    try:
        start_dt = datetime.strptime(cfg["bot_start_date"], "%Y-%m-%d")
        days = max((datetime.now() - start_dt).days, 0)
    except Exception:
        days = 0
    return cfg["initial_bet"] + days * cfg["daily_add"]


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (market data streaming)")
    while not _stop_flag.is_set():
        cfg = state.snapshot()["config"]
        try:
            symbol = cfg["symbol"]
            ex = get_exchange()

            state.add_log(f"Fetching {symbol} {cfg['timeframe']} candles from BingX…")
            bars = ex.fetch_ohlcv(symbol, timeframe=cfg["timeframe"], limit=400)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = compute_divergence(df, cfg)
            live_price = float(df.iloc[-1]["close"])
            state.connected = True
            state.add_log(f"Got {len(df)} candles, price={live_price}")
            print(f"[LOOP] PID={os.getpid()} - got {len(df)} candles, price={live_price}", flush=True)

            indicator_payload = {
                "timestamps": df["timestamp"].tolist(),
                "rsi": [None if pd.isna(v) else round(float(v), 2) for v in df["rsi"].tolist()],
                "bull": df["bull_div"].tolist(),
                "bear": df["bear_div"].tolist(),
            }
            ohlcv_payload = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
            state.update_market(ohlcv_payload, indicator_payload, live_price)

            df_trades = fetch_trades_safe(symbol)
            active_l, active_s = count_active_tickets(symbol, df_trades, cfg["max_tickets"])
            try:
                bal = ex.fetch_balance()
                total_cap = float(bal.get("USDT", {}).get("total", 0.0))
            except Exception:
                total_cap = state.balance
            state.update_positions(active_l, active_s, total_cap)

            if state.running:
                sig, ts = latest_signal(df)
                margin = _compute_margin(cfg)
                if sig == "LONG" and active_l < cfg["max_tickets"] and ts != _last_fired_ts["LONG"]:
                    fire_execution_order(symbol, "LONG", live_price, margin, cfg["leverage"], cfg["tp_percent"], cfg["sl_percent"])
                    _last_fired_ts["LONG"] = ts
                elif sig == "SHORT" and active_s < cfg["max_tickets"] and ts != _last_fired_ts["SHORT"]:
                    fire_execution_order(symbol, "SHORT", live_price, margin, cfg["leverage"], cfg["tp_percent"], cfg["sl_percent"])
                    _last_fired_ts["SHORT"] = ts
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
# 4) WEB DASHBOARD - Flask app + embedded HTML (no separate template file)
# ============================================================================

app = Flask(__name__)
print(f"[BOOT] PID={os.getpid()} - Flask app module loading, about to init engine…", flush=True)

# Started at import time (not just under `if __name__ == "__main__"`) so it
# also runs under gunicorn/production servers, same as `python this_file.py`.
init_engine()
print(f"[BOOT] PID={os.getpid()} - init_engine() called", flush=True)

INDEX_HTML = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LABUSDT · RSI Divergence Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{
    --bg:#0A0C12;
    --panel:#12151E;
    --panel-2:#171B27;
    --line:#232838;
    --text:#E7ECF3;
    --muted:#7C879C;
    --gold:#C9A24A;
    --gold-dim:#8A7130;
    --long:#3ED8A0;
    --short:#FF5C72;
    --radius:10px;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
    font-family:'Space Grotesk',sans-serif; -webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  a{color:inherit;}

  .app{display:grid;grid-template-columns:64px 320px 1fr;grid-template-rows:64px 1fr;
    height:100vh;}
  .brand{grid-column:1/3;grid-row:1;display:flex;align-items:center;gap:12px;
    padding:0 20px;border-bottom:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));
    display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:14px;}
  .brand h1{font-size:15px;letter-spacing:.04em;margin:0;font-weight:600;}
  .brand .sub{color:var(--muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase;}

  .topbar{grid-column:3;grid-row:1;display:flex;align-items:center;justify-content:space-between;
    padding:0 24px;border-bottom:1px solid var(--line);}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .dot.on{background:var(--long);box-shadow:0 0 8px var(--long);}
  .run-switch{display:flex;align-items:center;gap:10px;}
  .run-btn{background:var(--panel-2);border:1px solid var(--line);color:var(--text);
    padding:9px 18px;border-radius:8px;font-family:inherit;font-weight:600;font-size:12px;
    letter-spacing:.04em;cursor:pointer;transition:.15s;}
  .run-btn.active{background:var(--long);color:#062018;border-color:var(--long);}
  .run-btn.stopped{background:var(--short);color:#2a0509;border-color:var(--short);}

  .sidebar{grid-column:1;grid-row:2;border-right:1px solid var(--line);display:flex;
    flex-direction:column;align-items:center;padding-top:16px;gap:18px;color:var(--muted);}
  .sidebar .ic{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;
    justify-content:center;font-size:15px;}
  .sidebar .ic.active{background:var(--panel-2);color:var(--gold);}

  .control{grid-column:2;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:20px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
    margin:22px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .field{margin-bottom:10px;}
  .field label{display:block;font-size:11.5px;color:var(--muted);margin-bottom:5px;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);
    color:var(--text);padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;
    font-size:12.5px;}
  .field input:focus, .field select:focus{outline:none;border-color:var(--gold-dim);}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .save-btn{width:100%;margin-top:14px;background:var(--gold);color:#211a06;border:none;
    padding:11px;border-radius:8px;font-weight:700;font-size:12.5px;letter-spacing:.03em;
    cursor:pointer;font-family:inherit;}
  .warn{margin-top:16px;padding:10px 12px;background:#241412;border:1px solid #4a2620;
    border-radius:8px;font-size:11px;color:#f0a89c;line-height:1.5;}

  .main{grid-column:3;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
    padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
    margin-bottom:10px;display:flex;justify-content:space-between;}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:280px;}
  #rsiChart{height:150px;}

  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
  .metric-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
    padding:14px 16px;}
  .metric-card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .metric-card .val{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;margin-top:6px;}
  .val.pos{color:var(--long);} .val.neg{color:var(--short);}

  .split{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;}
  table{width:100%;border-collapse:collapse;font-size:12px;}
  th{text-align:left;color:var(--muted);font-weight:500;font-size:10.5px;
    text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line);}
  td{padding:7px 8px;border-bottom:1px solid #191d29;font-family:'IBM Plex Mono',monospace;}
  .side-long{color:var(--long);} .side-short{color:var(--short);}
  .status-open{color:var(--gold);} .status-win{color:var(--long);} .status-loss{color:var(--short);}
  .logbox{max-height:230px;overflow-y:auto;font-family:'IBM Plex Mono',monospace;font-size:11.5px;}
  .logline{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #171a24;color:#B7C0D1;}
  .logline .t{color:var(--muted);}
  ::-webkit-scrollbar{width:8px;height:8px;}
  ::-webkit-scrollbar-thumb{background:#232838;border-radius:4px;}
</style>
</head>
<body>
<div class="app">
  <div class="brand">
    <div class="mark">Δ</div>
    <div>
      <h1>WHALE HUNTER · RSI DIVERGENCE</h1>
      <div class="sub" id="symbolLabel">LAB / USDT · SWAP</div>
    </div>
  </div>

  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span></div>
    <div class="run-switch">
      <div class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></div>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>

  <div class="sidebar">
    <div class="ic active">◆</div>
    <div class="ic">▤</div>
    <div class="ic">◷</div>
    <div class="ic">▥</div>
    <div class="ic">⚙</div>
  </div>

  <div class="control">
    <h2>Trading Setup</h2>
    <div class="field"><label>Timeframe</label>
      <select id="cfg_timeframe">
        <option value="1m">1m</option><option value="5m" selected>5m</option>
        <option value="15m">15m</option><option value="1h">1h</option>
      </select>
    </div>
    <div class="row2">
      <div class="field"><label>Initial Bet (USDT)</label><input id="cfg_initial_bet" type="number" step="0.1" /></div>
      <div class="field"><label>Leverage (x)</label><input id="cfg_leverage" type="number" step="1" /></div>
    </div>

    <h2>Capital Management</h2>
    <div class="field"><label>Daily Add (USDT)</label><input id="cfg_daily_add" type="number" step="0.1" /></div>
    <div class="row2">
      <div class="field"><label>SL %</label><input id="cfg_sl_percent" type="number" step="0.1" /></div>
      <div class="field"><label>TP %</label><input id="cfg_tp_percent" type="number" step="0.1" /></div>
    </div>
    <div class="field"><label>Max Tickets / Side</label><input id="cfg_max_tickets" type="number" step="1" /></div>

    <h2>Indicator Settings</h2>
    <div class="row2">
      <div class="field"><label>Momentum Length</label><input id="cfg_mom_length" type="number" step="1" /></div>
      <div class="field"><label>RSI Length</label><input id="cfg_rsi_length" type="number" step="1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Left Bars</label><input id="cfg_left_bars" type="number" step="1" /></div>
      <div class="field"><label>Right Bars</label><input id="cfg_right_bars" type="number" step="1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>Min Bars Range</label><input id="cfg_min_bars_range" type="number" step="1" /></div>
      <div class="field"><label>Max Bars Range</label><input id="cfg_max_bars_range" type="number" step="1" /></div>
    </div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS</button>
    <div class="warn">Right Bars ต่ำ = สัญญาณไวแต่หลอกง่ายขึ้น สัญญาณจะยืนยันช้ากว่ากราฟที่เห็น เท่ากับจำนวน Right Bars เสมอ ทดสอบด้วยทุนน้อยก่อนเพิ่ม Leverage/Bet</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart (LABUSDT)</span><span class="price mono" id="livePrice">--</span></div>
      <div id="priceChart"></div>
    </div>
    <div class="panel">
      <div class="panel-title"><span>RSI (Momentum) · Divergence Zones</span><span id="posBadge" class="mono" style="color:var(--muted)">LONG 0 · SHORT 0</span></div>
      <div id="rsiChart"></div>
    </div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">Win Rate</div><div class="val" id="m_winrate">--</div></div>
      <div class="metric-card"><div class="lbl">Total Trades</div><div class="val" id="m_total">--</div></div>
      <div class="metric-card"><div class="lbl">Profit / Loss</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">Account Balance</div><div class="val" id="m_balance">--</div></div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table>
          <thead><tr><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>Status</th></tr></thead>
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
const CFG_KEYS = ["timeframe","initial_bet","leverage","daily_add","sl_percent","tp_percent",
  "max_tickets","mom_length","rsi_length","left_bars","right_bars","min_bars_range","max_bars_range"];

const priceChart = LightweightCharts.createChart(document.getElementById('priceChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const candleSeries = priceChart.addCandlestickSeries({
  upColor:'#3ED8A0', downColor:'#FF5C72', borderVisible:false,
  wickUpColor:'#3ED8A0', wickDownColor:'#FF5C72',
});

const rsiChart = LightweightCharts.createChart(document.getElementById('rsiChart'), {
  layout:{background:{color:'transparent'}, textColor:'#7C879C', fontFamily:'IBM Plex Mono'},
  grid:{vertLines:{color:'#171B27'}, horzLines:{color:'#171B27'}},
  rightPriceScale:{borderColor:'#232838'}, timeScale:{borderColor:'#232838', timeVisible:true},
});
const rsiSeries = rsiChart.addLineSeries({color:'#C9A24A', lineWidth:2});

function syncTimeScales(a,b){
  a.timeScale().subscribeVisibleLogicalRangeChange(r=>{ if(r) b.timeScale().setVisibleLogicalRange(r); });
}
syncTimeScales(priceChart, rsiChart);
syncTimeScales(rsiChart, priceChart);

let cfgLoadedOnce = false;

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
  document.getElementById('balanceVal').textContent = '$' + (data.balance||0).toFixed(2);
  document.getElementById('livePrice').textContent = data.live_price ? ('$'+data.live_price) : '--';
  document.getElementById('posBadge').textContent = `LONG ${data.active_long} · SHORT ${data.active_short}`;

  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running ? 'active' : 'stopped');

  const m = data.metrics;
  document.getElementById('m_winrate').textContent = m.win_rate + '%';
  document.getElementById('m_total').textContent = m.total_trades;
  const pnlEl = document.getElementById('m_pnl');
  pnlEl.textContent = (m.pnl>=0?'+':'') + m.pnl + ' USDT';
  pnlEl.className = 'val ' + (m.pnl>=0?'pos':'neg');
  document.getElementById('m_balance').textContent = '$' + (data.balance||0).toFixed(2);

  if(data.ohlcv && data.ohlcv.length){
    const candles = data.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]}));
    candleSeries.setData(candles);
    const markers = [];
    data.indicator.timestamps.forEach((ts,i)=>{
      if(data.indicator.bull[i]) markers.push({time:Math.floor(ts/1000), position:'belowBar', color:'#3ED8A0', shape:'arrowUp', text:'BUY'});
      if(data.indicator.bear[i]) markers.push({time:Math.floor(ts/1000), position:'aboveBar', color:'#FF5C72', shape:'arrowDown', text:'SELL'});
    });
    candleSeries.setMarkers(markers);

    const rsiPts = data.indicator.timestamps.map((ts,i)=>({time:Math.floor(ts/1000), value: data.indicator.rsi[i]}))
      .filter(p=>p.value!==null && p.value!==undefined);
    rsiSeries.setData(rsiPts);
  }

  const ordersBody = document.getElementById('ordersBody');
  ordersBody.innerHTML = data.trades.map(t=>`
    <tr>
      <td>${t.time||''}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.type||''}</td>
      <td>${t.entry??''}</td>
      <td>${t.tp??''}</td>
      <td>${t.sl??''}</td>
      <td class="status-${(t.status||'').toLowerCase()}">${t.status||''}</td>
    </tr>`).join('');

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
    """Hit this directly in the browser to test the BingX connection
    synchronously, with a short timeout, independent of the background loop."""
    import time as _time
    result = {"symbol": DEFAULT_CONFIG["symbol"]}
    try:
        test_ex = ccxt.bingx({
            "apiKey": BINGX_API_KEY,
            "secret": BINGX_SECRET_KEY,
            "enableRateLimit": True,
            "timeout": 8000,
            "options": {"defaultType": "swap"},
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
        "initial_bet", "daily_add", "leverage", "tp_percent", "sl_percent",
        "max_tickets", "mom_length", "rsi_length", "left_bars", "right_bars",
        "min_bars_range", "max_bars_range", "poll_seconds",
    }
    clean = {}
    for k, v in patch.items():
        if k in numeric_keys:
            try:
                clean[k] = float(v) if "." in str(v) else int(v)
            except (TypeError, ValueError):
                continue
        elif k in {"symbol", "timeframe", "bot_start_date"}:
            clean[k] = str(v)
    state.update_config(clean)
    state.add_log(f"Config updated: {clean}")
    return jsonify(state.snapshot()["config"])


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    set_running(bool(payload.get("running")))
    return jsonify({"running": state.running})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
