"""
cci_mtf_bot.py
------------------------
Single-file drop-in bot for BingX Futures — trades the "CCI Multi-Timeframe
Strategy (1m Entry + 15m/1h Confirm)" Pine Script:

  - Entry/Exit driven by CCI of the MAIN timeframe (the timeframe the bot
    polls on — default 1 minute, matching "TF หลักสำหรับเข้า/ออกออเดอร์ = TF
    ของกราฟที่เปิดอยู่" in the indicator).
  - Confirmed by CCI of up to 2 secondary timeframes (TF ยืนยัน 1 / TF ยืนยัน 2),
    each independently switchable on/off.
  - Long signal  = main-TF CCI crosses OVER the Oversold level, AND (if
    enabled) each confirm-TF CCI is above the Oversold level.
  - Short signal = main-TF CCI crosses UNDER the Overbought level, AND (if
    enabled) each confirm-TF CCI is below the Overbought level.
  - TP / SL are fixed % of entry price.
  - Kill switch: if account equity <= 0, the bot force-closes everything and
    refuses to open any new order until manually reset.

This keeps the same execution/dashboard/state-management skeleton as the
previous confluence_3cond_bot.py (BingX via ccxt, Flask dashboard, single
worker/thread, real TP/SL bracket orders, manual open/close, state.json
persistence) — only the SIGNAL LOGIC has been swapped out for the CCI-MTF
indicator, and it now manages a SINGLE ticket (the Pine strategy is one
condition, not three).

SETUP
-----
pip install ccxt flask pandas numpy python-dotenv gunicorn requests
.env with: BINGX_API_KEY=... / BINGX_SECRET_KEY=...
Run locally:   python cci_mtf_bot.py
Run online:    gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT cci_mtf_bot:app

⚠️ MUST stay at --workers 1 online. State + the trading loop live in one
process's memory; more than one worker/instance means duplicate real orders
firing on the same signal.

DEFAULT PARAMETERS (from the screenshot supplied by the user)
---------------------------------------------------------------
  ทุนเริ่มต้น (USD): 10          มูลค่าต่อไม้ (USD ต่อออเดอร์): 25
  ค่าธรรมเนียม: 0.05% ต่อฝั่ง
  ใช้ TF ยืนยัน 1: เปิด  -> 1 นาที
  ใช้ TF ยืนยัน 2: เปิด  -> 5 นาที
  TF หลัก (เข้า/ออกออเดอร์) = TF ของกราฟที่รันอยู่ (ตั้งเป็น 1 นาที)
  CCI Length: 14   Overbought: 10   Oversold: -10
  Take Profit: 2%   Stop Loss: 2%
  อนุญาตเปิด Long: เปิด   อนุญาตเปิด Short: เปิด
  ถ้า TP และ SL ถูกแตะในแท่งเดียวกัน ให้นับ SL ก่อน: เปิด
  หยุดรันเมื่อพอร์ตติดลบ (Equity <= 0): เปิด
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
# 1) INDICATOR - Python port of the Pine Script CCI Multi-Timeframe logic
# ============================================================================

def cci_series(df: pd.DataFrame, length: int) -> np.ndarray:
    """Classic CCI: (typical price - SMA(tp,len)) / (0.015 * mean deviation)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (tp - sma) / (0.015 * mad)
    return cci.values.astype(float)


def crossover(prev_val: float, cur_val: float, level: float) -> bool:
    if any(np.isnan(v) for v in (prev_val, cur_val)):
        return False
    return prev_val <= level and cur_val > level


def crossunder(prev_val: float, cur_val: float, level: float) -> bool:
    if any(np.isnan(v) for v in (prev_val, cur_val)):
        return False
    return prev_val >= level and cur_val < level


def fetch_confirm_cci_last(ex, symbol: str, timeframe: str, length: int) -> float:
    """Fetch a secondary timeframe and return its most recently CLOSED CCI
    value — mirrors request.security(..., barmerge.lookahead_off) in Pine."""
    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=length + 50)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    cci = cci_series(df, length)
    idx = len(df) - 2  # last closed candle
    if idx < 0 or idx >= len(cci):
        return float("nan")
    return float(cci[idx])


def compute_signals(cci_main: np.ndarray, idx: int, cfg: dict,
                     cci_tf1_last: float, cci_tf2_last: float):
    """Reproduces longCondition / shortCondition from the Pine script."""
    os_level = cfg["os_level"]
    ob_level = cfg["ob_level"]

    main_long_signal = crossover(cci_main[idx - 1], cci_main[idx], os_level) if idx > 0 else False
    main_short_signal = crossunder(cci_main[idx - 1], cci_main[idx], ob_level) if idx > 0 else False

    tf1_long_ok = (not cfg["tf1_enable"]) or (not np.isnan(cci_tf1_last) and cci_tf1_last > os_level)
    tf1_short_ok = (not cfg["tf1_enable"]) or (not np.isnan(cci_tf1_last) and cci_tf1_last < ob_level)
    tf2_long_ok = (not cfg["tf2_enable"]) or (not np.isnan(cci_tf2_last) and cci_tf2_last > os_level)
    tf2_short_ok = (not cfg["tf2_enable"]) or (not np.isnan(cci_tf2_last) and cci_tf2_last < ob_level)

    long_condition = cfg["allow_long"] and main_long_signal and tf1_long_ok and tf2_long_ok
    short_condition = cfg["allow_short"] and main_short_signal and tf1_short_ok and tf2_short_ok
    return long_condition, short_condition


def calc_tp_sl(cfg: dict, is_long: bool, entry_price: float):
    if is_long:
        tp = entry_price * (1 + cfg["tp_pct"] / 100.0)
        sl = entry_price * (1 - cfg["sl_pct"] / 100.0)
    else:
        tp = entry_price * (1 - cfg["tp_pct"] / 100.0)
        sl = entry_price * (1 + cfg["sl_pct"] / 100.0)
    return tp, sl


# ============================================================================
# 2) SHARED STATE
# ============================================================================

DEFAULT_CONFIG = {
    "symbol": "BEAT/USDT:USDT",
    "main_timeframe": "1m",     # TF หลัก (เข้า/ออกออเดอร์) = TF ของกราฟที่รันอยู่
    "leverage": 25,
    "bot_start_date": datetime.now().strftime("%Y-%m-%d"),

    "initial_cap": 10.0,        # ทุนเริ่มต้น (USD)
    "base_order_usdt": 25.0,    # มูลค่าต่อไม้ (USD ต่อออเดอร์)
    "fee_pct": 0.05,            # ค่าธรรมเนียม (% ต่อการเทรด/ต่อฝั่ง)

    "tf1_enable": True, "tf1": "1m",   # ใช้ TF ยืนยัน 1 / Timeframe ยืนยัน 1
    "tf2_enable": True, "tf2": "5m",   # ใช้ TF ยืนยัน 2 / Timeframe ยืนยัน 2

    "cci_len": 14,
    "ob_level": 10,
    "os_level": -10,

    "tp_pct": 2.0,
    "sl_pct": 2.0,
    "allow_long": True,
    "allow_short": True,
    "sl_first_if_both_hit": True,

    "stop_when_blown": True,    # หยุดรันเมื่อพอร์ตติดลบ (Equity <= 0)

    "poll_seconds": 10,
}


def empty_position_state():
    return {
        "in_position": False, "pos_side": None, "entry_price": None,
        "tp": None, "sl": None, "contract_amount": None,
        "total_trades": 0, "wins": 0, "losses": 0, "net_profit": 0.0,
    }


class BotState:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = dict(DEFAULT_CONFIG)
        self.pos = empty_position_state()
        self.bot_stopped = False   # kill switch latch (equity <= 0)
        self.running = False
        self.connected = False
        self.trades = []
        self.logs = []
        self.ohlcv = []
        self.cci_snapshot = {"main": None, "tf1": None, "tf2": None}
        self.live_price = None
        self.balance = 0.0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self.config.update(data.get("config", {}))
                self.pos.update(data.get("pos", {}))
                self.bot_stopped = data.get("bot_stopped", False)
                self.trades = data.get("trades", [])
                self.logs = data.get("logs", [])
            except Exception:
                pass

    def _save(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"config": self.config, "pos": self.pos, "bot_stopped": self.bot_stopped,
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

    def update_market(self, ohlcv, cci_snapshot, live_price):
        with self.lock:
            self.ohlcv = ohlcv
            self.cci_snapshot = cci_snapshot
            self.live_price = live_price

    def equity(self):
        return round(self.config["initial_cap"] + self.pos["net_profit"], 6)

    def snapshot(self):
        with self.lock:
            equity = self.equity()
            winrate = round((self.pos["wins"] / self.pos["total_trades"]) * 100, 2) if self.pos["total_trades"] else 0.0
            net_pct = round((self.pos["net_profit"] / self.config["initial_cap"]) * 100, 2) if self.config["initial_cap"] else 0.0
            return {
                "config": dict(self.config),
                "pos": dict(self.pos),
                "bot_stopped": self.bot_stopped,
                "running": self.running,
                "connected": self.connected,
                "live_price": self.live_price,
                "balance": self.balance,
                "equity": equity,
                "winrate": winrate,
                "net_pct": net_pct,
                "cci": self.cci_snapshot,
                "trades": self.trades[:60],
                "logs": self.logs[-100:],
                "ohlcv": self.ohlcv,
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


def cfg_fee_notional(amount, entry_price):
    cfg = state.config
    return amount * entry_price * (cfg["fee_pct"] / 100.0) * 2  # entry + exit


def fire_entry(symbol: str, side: str, entry_price: float, cfg: dict, manual=False):
    """Opens a real market order + real TP/SL bracket orders."""
    ex = get_exchange()
    try:
        contract_amount = round(cfg["base_order_usdt"] / entry_price, 4)
        tp, sl = calc_tp_sl(cfg, side == "LONG", entry_price)
        tp = round(float(tp), 6)
        sl = round(float(sl), 6)

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

        with state.lock:
            state.pos["in_position"] = True
            state.pos["pos_side"] = side
            state.pos["entry_price"] = entry_price
            state.pos["tp"] = tp
            state.pos["sl"] = sl
            state.pos["contract_amount"] = contract_amount
            state._save()

        tag = " [MANUAL]" if manual else ""
        state.add_trade({
            "id": uuid.uuid4().hex[:10],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_ms": int(datetime.now().timestamp() * 1000),
            "type": f"OPEN ({'BUY' if side == 'LONG' else 'SELL'}) CCI-MTF{tag}",
            "entry": entry_price, "tp": tp, "sl": sl,
            "contract_amount": contract_amount,
            "pnl": None, "status": "OPEN", "side": side,
        })
        state.add_log(f"ENTRY EXECUTED{tag}: {side} @ {entry_price} | TP {tp} / SL {sl}")
        return True, "order sent"
    except Exception as e:
        state.add_log(f"⚠️ ORDER FAILED ({side}): {e}")
        return False, str(e)


def close_ticket(exit_price: float, reason: str):
    """Software-side bookkeeping close (bracket order on the exchange does
    the REAL closing on a live account — this records the result)."""
    with state.lock:
        if not state.pos["in_position"]:
            return
        entry = state.pos["entry_price"] or 0
        amount = state.pos["contract_amount"] or 0
        side = state.pos["pos_side"]
        pnl = (exit_price - entry) * amount if side == "LONG" else (entry - exit_price) * amount
        pnl -= cfg_fee_notional(amount, entry)

        state.pos["total_trades"] += 1
        if pnl >= 0:
            state.pos["wins"] += 1
        else:
            state.pos["losses"] += 1
        state.pos["net_profit"] = round(state.pos["net_profit"] + pnl, 6)

        for t in state.trades:
            if t.get("status") == "OPEN":
                t["status"] = "WIN" if pnl >= 0 else "LOSS"
                t["exit"] = exit_price
                t["exit_ms"] = int(datetime.now().timestamp() * 1000)
                t["pnl"] = round(pnl, 4)
                break

        state.pos["in_position"] = False
        state.pos["pos_side"] = None
        state.pos["entry_price"] = None
        state.pos["tp"] = None
        state.pos["sl"] = None
        state.pos["contract_amount"] = None
        state._save()

    state.add_log(f"Position closed ({reason}) @ ~{exit_price} | PnL {round(pnl, 4)} USDT")


def force_close_market(symbol: str):
    """Used by the kill switch: actually flattens the real position too."""
    if not state.pos["in_position"]:
        return
    ex = get_exchange()
    side = state.pos["pos_side"]
    amount = state.pos["contract_amount"]
    close_side = "sell" if side == "LONG" else "buy"
    try:
        ex.create_order(symbol=symbol, type="market", side=close_side, amount=amount,
                         params={"positionSide": side, "reduceOnly": True})
    except Exception as e:
        state.add_log(f"⚠️ Kill-switch close failed: {e}")
    exit_price = state.live_price or state.pos["entry_price"]
    close_ticket(exit_price, "ACCOUNT BLOWN - STOPPED")


def open_trade_manual(side: str):
    cfg = state.snapshot()["config"]
    if state.bot_stopped:
        return False, "Bot is stopped (equity <= 0)"
    if not state.live_price:
        return False, "No live price yet"
    if state.pos["in_position"]:
        return False, "A ticket is already open"
    return fire_entry(cfg["symbol"], side, state.live_price, cfg, manual=True)


def close_trade_manual():
    if not state.pos["in_position"]:
        return False, "No open ticket"
    cfg = state.snapshot()["config"]
    symbol = cfg["symbol"]
    side = state.pos["pos_side"]
    amount = state.pos["contract_amount"]
    ex = get_exchange()
    close_side = "sell" if side == "LONG" else "buy"
    try:
        ex.create_order(symbol=symbol, type="market", side=close_side, amount=amount,
                         params={"positionSide": side, "reduceOnly": True})
        exit_price = state.live_price or state.pos["entry_price"]
        close_ticket(exit_price, "MANUAL CLOSE")
        return True, "closed"
    except Exception as e:
        state.add_log(f"⚠️ Manual close failed: {e}")
        return False, str(e)


def _loop():
    print(f"[LOOP] PID={os.getpid()} - engine loop thread starting now", flush=True)
    state.add_log("🟢 Engine started (market data streaming)")
    while not _stop_flag.is_set():
        cfg = state.snapshot()["config"]
        try:
            symbol = cfg["symbol"]
            ex = get_exchange()
            bars = ex.fetch_ohlcv(symbol, timeframe=cfg["main_timeframe"], limit=max(300, cfg["cci_len"] + 100))
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            high_arr = df["high"].values.astype(float)
            low_arr = df["low"].values.astype(float)
            live_price = float(df["close"].values[-1])
            state.connected = True

            cci_main = cci_series(df, cfg["cci_len"])
            idx = len(df) - 2  # last CLOSED candle on the main TF

            cci_tf1_last = fetch_confirm_cci_last(ex, symbol, cfg["tf1"], cfg["cci_len"]) if cfg["tf1_enable"] else float("nan")
            cci_tf2_last = fetch_confirm_cci_last(ex, symbol, cfg["tf2"], cfg["cci_len"]) if cfg["tf2_enable"] else float("nan")

            cci_snapshot = {
                "main": None if idx < 0 or np.isnan(cci_main[idx]) else round(float(cci_main[idx]), 2),
                "tf1": None if np.isnan(cci_tf1_last) else round(cci_tf1_last, 2),
                "tf2": None if np.isnan(cci_tf2_last) else round(cci_tf2_last, 2),
            }
            ohlcv_payload = df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
            state.update_market(ohlcv_payload, cci_snapshot, live_price)

            try:
                bal = ex.fetch_balance()
                state.balance = float(bal.get("USDT", {}).get("total", 0.0))
            except Exception:
                pass

            # 1) Kill switch check FIRST — equity <= 0 halts everything
            if cfg["stop_when_blown"] and not state.bot_stopped and state.equity() <= 0:
                state.bot_stopped = True
                state.running = False
                state.add_log("🛑 ACCOUNT BLOWN (equity <= 0) — closing all + halting new entries")
                force_close_market(symbol)

            # 2) Manage the open ticket (check last closed candle's high/low vs TP/SL)
            if not state.bot_stopped and state.pos["in_position"]:
                hi, lo = high_arr[idx], low_arr[idx]
                side = state.pos["pos_side"]
                tp, sl = state.pos["tp"], state.pos["sl"]
                if side == "LONG":
                    hit_tp, hit_sl = hi >= tp, lo <= sl
                else:
                    hit_tp, hit_sl = lo <= tp, hi >= sl

                if hit_tp and hit_sl:
                    # ถ้า TP และ SL ถูกแตะในแท่งเดียวกัน ให้นับ SL ก่อน (ถ้าเปิดใช้)
                    if cfg["sl_first_if_both_hit"]:
                        close_ticket(sl, "SL")
                    else:
                        close_ticket(tp, "TP")
                elif hit_sl:
                    close_ticket(sl, "SL")
                elif hit_tp:
                    close_ticket(tp, "TP")

            # 3) Look for a new entry if flat, running, and past the start date
            if (not state.bot_stopped and state.running and not state.pos["in_position"]
                    and _is_trading_started(cfg) and idx > 0):
                long_condition, short_condition = compute_signals(cci_main, idx, cfg, cci_tf1_last, cci_tf2_last)
                if long_condition:
                    fire_entry(symbol, "LONG", live_price, cfg)
                elif short_condition:
                    fire_entry(symbol, "SHORT", live_price, cfg)

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
<title>CCI Multi-Timeframe Bot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{ --bg:#0A0C12; --panel:#12151E; --panel-2:#171B27; --line:#232838; --text:#E7ECF3; --muted:#7C879C;
    --gold:#C9A24A; --gold-dim:#8A7130; --long:#3ED8A0; --short:#FF5C72; --radius:10px; }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;-webkit-font-smoothing:antialiased;}
  .mono{font-family:'IBM Plex Mono',monospace;}
  .app{display:grid;grid-template-columns:320px 1fr;grid-template-rows:64px 1fr;height:100vh;}
  .brand{grid-column:1/2;grid-row:1;display:flex;align-items:center;gap:12px;padding:0 20px;border-bottom:1px solid var(--line);}
  .brand .mark{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,var(--gold),var(--gold-dim));display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0C12;font-size:12px;}
  .brand h1{font-size:14px;letter-spacing:.04em;margin:0;font-weight:600;}
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
  .manual-btn{border:1px solid var(--line);background:var(--panel-2);color:var(--text);padding:7px 12px;border-radius:6px;font-family:inherit;font-weight:600;font-size:11px;cursor:pointer;}
  .manual-btn.long{color:var(--long);border-color:var(--long);}
  .manual-btn.short{color:var(--short);border-color:var(--short);}
  .close-btn{background:transparent;border:1px solid var(--short);color:var(--short);padding:3px 9px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;}
  .control{grid-column:1;grid-row:2;border-right:1px solid var(--line);overflow-y:auto;padding:20px;}
  .control h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:22px 0 10px;}
  .control h2:first-child{margin-top:0;}
  .field{margin-bottom:10px;}
  .field label{display:block;font-size:11.5px;color:var(--muted);margin-bottom:5px;}
  .field input, .field select{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:7px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .save-btn{width:100%;margin-top:14px;background:var(--gold);color:#211a06;border:none;padding:11px;border-radius:8px;font-weight:700;font-size:12.5px;letter-spacing:.03em;cursor:pointer;font-family:inherit;}
  .reset-btn{width:100%;margin-top:8px;background:transparent;border:1px solid var(--short);color:var(--short);padding:9px;border-radius:8px;font-weight:700;font-size:11.5px;cursor:pointer;font-family:inherit;}
  .main{grid-column:2;grid-row:2;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;}
  .panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between;}
  .price{font-family:'IBM Plex Mono',monospace;color:var(--gold);}
  #priceChart{height:280px;}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
  .metric-card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;}
  .metric-card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
  .metric-card .val{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;margin-top:6px;}
  .val.pos{color:var(--long);} .val.neg{color:var(--short);}
  .cci-row{display:flex;gap:14px;}
  .cci-card{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:12px 14px;}
  .cci-card .lbl{font-size:10.5px;color:var(--muted);text-transform:uppercase;}
  .cci-card .val{font-family:'IBM Plex Mono',monospace;font-size:17px;font-weight:600;margin-top:4px;}
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
  <div class="brand"><div class="mark">CCI</div><div><h1>CCI MULTI-TIMEFRAME BOT</h1><div class="sub" id="symbolLabel">LOADING…</div></div></div>
  <div class="topbar">
    <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">connecting…</span></div>
    <div class="manual-group">
      <span class="status-pill">Balance <span class="mono price" id="balanceVal" style="margin-left:6px">--</span></span>
      <button class="manual-btn long" id="manualLongBtn">▲ LONG</button>
      <button class="manual-btn short" id="manualShortBtn">▼ SHORT</button>
      <button class="run-btn stopped" id="runBtn">▶ RUN BOT</button>
    </div>
  </div>

  <div class="control">
    <h2>เงินทุน / ค่าธรรมเนียม / ขนาดไม้</h2>
    <div class="field"><label>Symbol</label><input id="cfg_symbol" type="text" /></div>
    <div class="row2">
      <div class="field"><label>ทุนเริ่มต้น (USD)</label><input id="cfg_initial_cap" type="number" step="0.01" /></div>
      <div class="field"><label>มูลค่าต่อไม้ (USD ต่อออเดอร์)</label><input id="cfg_base_order_usdt" type="number" step="0.01" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>ค่าธรรมเนียม (%)</label><input id="cfg_fee_pct" type="number" step="0.01" /></div>
      <div class="field"><label>Leverage (x)</label><input id="cfg_leverage" type="number" step="1" /></div>
    </div>
    <div class="field"><label>วันเริ่มเทรด</label><input id="cfg_bot_start_date" type="date" /></div>

    <h2>ไทม์เฟรม</h2>
    <div class="field"><label>TF หลัก (เข้า/ออกออเดอร์)</label>
      <select id="cfg_main_timeframe"><option value="1m">1 นาที</option><option value="5m">5 นาที</option><option value="15m">15 นาที</option><option value="1h">1 ชั่วโมง</option></select>
    </div>
    <div class="row2">
      <div class="field"><label>ใช้ TF ยืนยัน 1</label><select id="cfg_tf1_enable"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>Timeframe ยืนยัน 1</label><select id="cfg_tf1"><option value="1m">1 นาที</option><option value="5m">5 นาที</option><option value="15m">15 นาที</option><option value="1h">1 ชั่วโมง</option></select></div>
    </div>
    <div class="row2">
      <div class="field"><label>ใช้ TF ยืนยัน 2</label><select id="cfg_tf2_enable"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>Timeframe ยืนยัน 2</label><select id="cfg_tf2"><option value="1m">1 นาที</option><option value="5m">5 นาที</option><option value="15m">15 นาที</option><option value="1h">1 ชั่วโมง</option></select></div>
    </div>

    <h2>CCI</h2>
    <div class="row2">
      <div class="field"><label>CCI Length</label><input id="cfg_cci_len" type="number" step="1" /></div>
      <div class="field"><label>Overbought</label><input id="cfg_ob_level" type="number" step="1" /></div>
    </div>
    <div class="field"><label>Oversold</label><input id="cfg_os_level" type="number" step="1" /></div>

    <h2>Take Profit / Stop Loss (% ของราคา)</h2>
    <div class="row2">
      <div class="field"><label>Take Profit (%)</label><input id="cfg_tp_pct" type="number" step="0.1" /></div>
      <div class="field"><label>Stop Loss (%)</label><input id="cfg_sl_pct" type="number" step="0.1" /></div>
    </div>
    <div class="row2">
      <div class="field"><label>อนุญาตเปิด Long</label><select id="cfg_allow_long"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
      <div class="field"><label>อนุญาตเปิด Short</label><select id="cfg_allow_short"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>
    </div>
    <div class="field"><label>ถ้า TP/SL แตะพร้อมกัน นับ SL ก่อน</label><select id="cfg_sl_first_if_both_hit"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>

    <h2>Kill Switch</h2>
    <div class="field"><label>หยุดรันเมื่อพอร์ตติดลบ (Equity &lt;= 0)</label><select id="cfg_stop_when_blown"><option value="true">เปิด</option><option value="false">ปิด</option></select></div>

    <button class="save-btn" id="saveCfgBtn">SAVE SETTINGS</button>
    <button class="reset-btn" id="resetKillBtn">RESET KILL SWITCH</button>
    <div class="warn">ออเดอร์เข้า/ออกอ้างอิงจาก CCI ของ TF หลัก ยืนยันด้วย CCI ของ TF ยืนยัน 1/2 (ถ้าเปิดใช้) — TP/SL เป็นออเดอร์จริงบน BingX ทุกไม้</div>
  </div>

  <div class="main">
    <div class="panel">
      <div class="panel-title"><span>Price Chart</span><span class="price mono" id="livePrice">--</span></div>
      <div id="priceChart"></div>
    </div>

    <div class="cci-row">
      <div class="cci-card"><div class="lbl">CCI (TF หลัก)</div><div class="val" id="cci_main">--</div></div>
      <div class="cci-card"><div class="lbl">CCI (TF ยืนยัน 1)</div><div class="val" id="cci_tf1">--</div></div>
      <div class="cci-card"><div class="lbl">CCI (TF ยืนยัน 2)</div><div class="val" id="cci_tf2">--</div></div>
    </div>

    <div class="metrics">
      <div class="metric-card"><div class="lbl">สถานะบอท</div><div class="val" id="m_status">--</div></div>
      <div class="metric-card"><div class="lbl">Equity ปัจจุบัน</div><div class="val" id="m_equity">--</div></div>
      <div class="metric-card"><div class="lbl">กำไร/ขาดทุนสุทธิ</div><div class="val" id="m_pnl">--</div></div>
      <div class="metric-card"><div class="lbl">Winrate</div><div class="val" id="m_winrate">--</div></div>
    </div>

    <div class="split">
      <div class="panel">
        <div class="panel-title">List Order</div>
        <table><thead><tr><th>Time</th><th>Type</th><th>Entry</th><th>TP</th><th>SL</th><th>Status</th><th>Action</th></tr></thead><tbody id="ordersBody"></tbody></table>
      </div>
      <div class="panel"><div class="panel-title">Bot Activity Log</div><div class="logbox" id="logBody"></div></div>
    </div>
  </div>
</div>

<script>
const CFG_KEYS = ["symbol","main_timeframe","leverage","bot_start_date","initial_cap","base_order_usdt","fee_pct",
  "tf1_enable","tf1","tf2_enable","tf2","cci_len","ob_level","os_level",
  "tp_pct","sl_pct","allow_long","allow_short","sl_first_if_both_hit","stop_when_blown"];

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
function showToast(t){
  const isWin = t.status==='WIN';
  const el = document.createElement('div');
  el.className = 'toast ' + (isWin?'win':'loss');
  el.innerHTML = `<div class="ttitle">${isWin?'🎯 TP HIT':'🛑 SL HIT'}</div><div class="tbody">${t.side} @ ${t.exit??''} · PnL ${(t.pnl>=0?'+':'')+t.pnl} USDT</div>`;
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

function render(data){
  document.getElementById('connDot').className = 'dot ' + (data.bot_stopped?'stopped':(data.connected?'on':''));
  document.getElementById('connText').textContent = data.bot_stopped ? 'STOPPED (พอร์ตแตก)' : (data.connected ? 'CONNECTED · BingX' : 'CONNECTING…');
  document.getElementById('symbolLabel').textContent = (data.config.symbol||'') + ' · ' + (data.config.main_timeframe||'');
  document.getElementById('livePrice').textContent = data.live_price ? ('$'+data.live_price) : '--';
  document.getElementById('balanceVal').textContent = '$' + (data.balance||0).toFixed(2);

  const runBtn = document.getElementById('runBtn');
  runBtn.textContent = data.running ? '■ STOP BOT' : '▶ RUN BOT';
  runBtn.className = 'run-btn ' + (data.running?'active':'stopped');
  runBtn.disabled = data.bot_stopped;

  document.getElementById('cci_main').textContent = data.cci.main ?? '--';
  document.getElementById('cci_tf1').textContent = data.cci.tf1 ?? '--';
  document.getElementById('cci_tf2').textContent = data.cci.tf2 ?? '--';

  const statusEl = document.getElementById('m_status');
  statusEl.textContent = data.bot_stopped ? 'STOPPED' : (data.pos.in_position ? ('🟢 '+data.pos.pos_side) : (data.running ? 'RUNNING' : 'IDLE'));
  statusEl.className = 'val ' + (data.bot_stopped ? 'neg' : (data.running ? 'pos' : ''));

  document.getElementById('m_equity').textContent = data.equity + ' USDT';
  const pnlEl = document.getElementById('m_pnl');
  pnlEl.textContent = (data.pos.net_profit>=0?'+':'') + data.pos.net_profit.toFixed(4) + ' USDT (' + (data.net_pct>=0?'+':'') + data.net_pct + '%)';
  pnlEl.className = 'val ' + (data.pos.net_profit>=0?'pos':'neg');
  document.getElementById('m_winrate').textContent = data.winrate + '% (' + data.pos.wins + 'W / ' + data.pos.losses + 'L)';

  if(data.ohlcv && data.ohlcv.length){
    candleSeries.setData(data.ohlcv.map(r=>({time:Math.floor(r[0]/1000), open:r[1], high:r[2], low:r[3], close:r[4]})));
    const markers = [];
    data.trades.forEach(t=>{
      markers.push({time: Math.floor(t.entry_ms/1000), position: t.side==='LONG'?'belowBar':'aboveBar', color: t.side==='LONG'?'#3ED8A0':'#FF5C72', shape: t.side==='LONG'?'arrowUp':'arrowDown', text: t.side});
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
      <td>${t.time||''}</td>
      <td class="${t.side==='LONG'?'side-long':'side-short'}">${t.type||''}</td>
      <td>${t.entry??''}</td><td>${t.tp??''}</td><td>${t.sl??''}</td>
      <td>${statusBadge(t)}</td>
      <td>${t.status==='OPEN' ? `<button class="close-btn" id="closeBtn">CLOSE</button>` : ''}</td>
    </tr>`).join('');
  const closeBtn = document.getElementById('closeBtn');
  if(closeBtn){
    closeBtn.addEventListener('click', async ()=>{
      closeBtn.disabled=true; closeBtn.textContent='...';
      const res = await fetch('/api/close-order', {method:'POST'});
      const j = await res.json();
      if(!j.ok) alert('Close failed: '+(j.message||j.error||'unknown error'));
    });
  }

  maybeShowToast(data.trades);
  document.getElementById('logBody').innerHTML = data.logs.slice().reverse().map(l=>`<div class="logline"><span class="t">${l.time}</span><span>${l.text}</span></div>`).join('');

  if(!cfgLoadedOnce){
    CFG_KEYS.forEach(k=>{ const el = document.getElementById('cfg_'+k); if(el && data.config[k]!==undefined) el.value = data.config[k]; });
    cfgLoadedOnce = true;
  }
}

document.getElementById('manualLongBtn').addEventListener('click', async ()=>{
  if(!confirm('เปิดออเดอร์ LONG ด้วยมือ ยิงจริงทันที แน่ใจไหม?')) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({side:'LONG'})});
  const j = await res.json();
  if(!j.ok) alert('เปิดออเดอร์ไม่สำเร็จ: '+(j.message||j.error||'unknown error'));
});
document.getElementById('manualShortBtn').addEventListener('click', async ()=>{
  if(!confirm('เปิดออเดอร์ SHORT ด้วยมือ ยิงจริงทันที แน่ใจไหม?')) return;
  const res = await fetch('/api/manual-order', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({side:'SHORT'})});
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
    return jsonify({"status": "ok", "connected": state.connected, "running": state.running, "bot_stopped": state.bot_stopped})


@app.route("/api/status")
def api_status():
    return jsonify(state.snapshot())


@app.route("/api/config", methods=["POST"])
def api_config():
    patch = request.get_json(force=True) or {}
    numeric_keys = {"leverage", "initial_cap", "base_order_usdt", "fee_pct",
                    "cci_len", "ob_level", "os_level", "tp_pct", "sl_pct"}
    bool_keys = {"tf1_enable", "tf2_enable", "allow_long", "allow_short", "sl_first_if_both_hit", "stop_when_blown"}
    string_keys = {"symbol", "main_timeframe", "tf1", "tf2", "bot_start_date"}

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
    if side not in ("LONG", "SHORT"):
        return jsonify({"ok": False, "error": "invalid side"}), 400
    ok, msg = open_trade_manual(side)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/close-order", methods=["POST"])
def api_close_order():
    ok, msg = close_trade_manual()
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
