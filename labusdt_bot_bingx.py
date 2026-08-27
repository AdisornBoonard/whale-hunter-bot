#!/usr/bin/env python3
# ==============================================================================
# Whale Hunter Multi-Timeframe Bot (3m + 5m, พารามิเตอร์แยกอิสระต่อ TF)
# ------------------------------------------------------------------------------
# แปลงเงื่อนไขจากอินดิเกเตอร์ Pine Script "Whale Hunter V10 - Real Fee & Growth
# (No Repaint)" มาเป็นบอทเทรดจริงด้วย Python + ccxt
#
# ตรรกะสัญญาณ (เหมือนอินดิเกเตอร์ทุกประการ) — คำนวณแยกอิสระในแต่ละ Timeframe:
#   - Whale Volume: volume ของแท่งที่ปิดแล้ว > SMA(volume, VOL_MA_LEN) x VOL_SPIKE_MULT
#   - CCI(CCI_LEN) ตัดขึ้น 100 (Long) / ตัดลง -100 (Short)
#   - ต้องยืนยันทิศทางด้วย EMA(EMA_LEN): close > EMA = อนุญาต Long, close < EMA = อนุญาต Short
#   - ใช้สัญญาณจาก "แท่งที่ปิดแล้ว" เท่านั้น (no repaint) แล้วเข้าออเดอร์ที่ราคา
#     "open" ของแท่งถัดไป เหมือนกับที่อินดิเกเตอร์ใช้ open ของแท่งปัจจุบันตอนคำนวณ TP/SL
#
# จุดสำคัญตามที่ขอ:
#   - รัน 2 Timeframe (ค่าเริ่มต้น 3m และ 5m) พร้อมกัน อิสระต่อกัน
#   - TP / SL ตั้งแยกกันคนละชุดต่อ TF
#   - จำนวนไม้สูงสุดที่เปิดพร้อมกัน (Max Trades) แยกคนละชุดต่อ TF
#   - Equity / เงินทุน เป็นบัญชีเดียวกัน (ใช้เงินจริงก้อนเดียว) แต่นับ Win/Loss
#     และจำนวนไม้ที่เปิดอยู่แยกกันต่อ TF สำหรับแดชบอร์ด
#   - PnL ต่อไม้คำนวณแบบเดียวกับอินดิเกเตอร์ต้นฉบับ คือใช้ % TP/SL คูณ Leverage
#     โดยตรง (ไม่ได้คำนวณจากส่วนต่างราคาจริง) เพื่อให้ผลตรงกับต้นฉบับ
#   - เพิ่ม Kill Switch (หยุดรันเมื่อ Equity รวม <= 0) ตามที่ใช้มาตลอดทั้งบอท
#     (อินดิเกเตอร์ต้นฉบับไม่มีส่วนนี้ แต่เป็นข้อกำหนดหลักของบอทเรา)
#
# ⚠️ คำเตือนความปลอดภัย
#   - ค่าเริ่มต้น DRY_RUN = True (โหมดจำลอง ไม่ส่งออเดอร์จริง)
#   - มี Leverage สูง ความเสี่ยงพอร์ตแตกไวมาก ทดสอบบน Testnet ก่อนใช้เงินจริงเสมอ
#   - ห้ามใส่ API Key/Secret ตรง ๆ ในไฟล์นี้ถ้าจะแชร์โค้ดต่อ ใช้ environment
#     variable แทน (ดูหมายเหตุท้ายไฟล์)
# ==============================================================================

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import ccxt

# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG = {
    # ---------------- Exchange / Connection ----------------
    "EXCHANGE_ID": "binance",
    "API_KEY": os.getenv("EXCHANGE_API_KEY", ""),
    "API_SECRET": os.getenv("EXCHANGE_API_SECRET", ""),
    "USE_TESTNET": True,
    "SYMBOL": "BTC/USDT",
    "MARKET_TYPE": "future",               # ต้องเป็น future ถ้าจะ Short จริง/ใช้ Leverage

    # ---------------- เงินทุน / Leverage / ค่าธรรมเนียม (ใช้ร่วมกันทั้งบัญชี) ----------------
    "START_CAPITAL_USD": 10.0,
    "BASE_MARGIN_USD": 1.0,                # เงินต้นเริ่มต้นต่อไม้ ($)
    "DAILY_ADD_USD": 1.0,                  # เพิ่มเงินต้นต่อไม้ วันละเท่านี้ (โตตามวัน เหมือนต้นฉบับ)
    "LEVERAGE": 20.0,
    "FEE_PERCENT": 0.04,                   # % ต่อฝั่ง (Maker/Taker) ของมูลค่า margin x leverage

    # ---------------- Whale Signal (ค่าที่ใช้ร่วมกันได้ หรือปรับแยกต่อ TF ก็ได้) ----------------
    "CCI_LEN": 20,
    "VOL_MA_LEN": 20,
    "VOL_SPIKE_MULT": 2.0,
    "EMA_LEN": 200,

    # ---------------- Timeframes: ตั้งค่า TP / SL / Max Trades แยกอิสระต่อ TF ----------------
    "TIMEFRAMES": {
        "3m": {
            "ENABLE": True,
            "TP_PERCENT": 3.0,
            "SL_PERCENT": 5.0,
            "MAX_TRADES": 3,
            "OHLCV_LIMIT": 250,
        },
        "5m": {
            "ENABLE": True,
            "TP_PERCENT": 3.0,
            "SL_PERCENT": 5.0,
            "MAX_TRADES": 3,
            "OHLCV_LIMIT": 250,
        },
    },

    # ---------------- Kill Switch ----------------
    "STOP_WHEN_BLOWN": True,

    # ---------------- Runtime ----------------
    "DRY_RUN": True,
    "POLL_SECONDS": 5,
    "LOG_FILE": "whale_hunter_bot.log",
    "TRADE_LOG_CSV": "whale_hunter_trades.csv",
}

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG["LOG_FILE"], encoding="utf-8"),
    ],
)
log = logging.getLogger("WhaleHunter-MTF")


# ==============================================================================
# INDICATORS
# ==============================================================================

def compute_cci(df: pd.DataFrame, length: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(length).mean()
    mean_dev = tp.rolling(length).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    return (tp - sma) / (0.015 * mean_dev)


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


# ==============================================================================
# STATE
# ==============================================================================

@dataclass
class Trade:
    side: int              # 1 = Long, -1 = Short
    margin: float
    tp_price: float
    sl_price: float
    tf: str


@dataclass
class TFStats:
    active_count: int = 0
    win_count: int = 0
    loss_count: int = 0


@dataclass
class AccountState:
    equity: float
    start_time_ms: int
    total_fee_paid: float = 0.0
    stopped: bool = False
    trades: list = field(default_factory=list)          # list[Trade] — ไม้ที่เปิดอยู่ (ทุก TF ปนกัน)
    tf_stats: dict = field(default_factory=dict)         # tf_name -> TFStats
    trade_log: list = field(default_factory=list)

    def stats_for(self, tf: str) -> TFStats:
        if tf not in self.tf_stats:
            self.tf_stats[tf] = TFStats()
        return self.tf_stats[tf]

    @property
    def growth_pct(self) -> float:
        start = CONFIG["START_CAPITAL_USD"]
        return ((self.equity - start) / start * 100.0) if start else 0.0

    def days_passed(self) -> int:
        now_ms = int(time.time() * 1000)
        return int((now_ms - self.start_time_ms) // (24 * 60 * 60 * 1000))

    def current_margin(self) -> float:
        return CONFIG["BASE_MARGIN_USD"] + self.days_passed() * CONFIG["DAILY_ADD_USD"]


# ==============================================================================
# EXCHANGE CLIENT
# ==============================================================================

def build_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, CONFIG["EXCHANGE_ID"])
    exchange = exchange_class({
        "apiKey": CONFIG["API_KEY"],
        "secret": CONFIG["API_SECRET"],
        "enableRateLimit": True,
        "options": {"defaultType": CONFIG["MARKET_TYPE"]},
    })
    if CONFIG["USE_TESTNET"] and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
        log.info("เปิดใช้งาน Testnet/Sandbox mode")

    if not CONFIG["DRY_RUN"] and CONFIG["MARKET_TYPE"] == "future":
        try:
            exchange.set_leverage(int(CONFIG["LEVERAGE"]), CONFIG["SYMBOL"])
            log.info(f"ตั้ง Leverage บน exchange เป็น {CONFIG['LEVERAGE']}x")
        except Exception as e:
            log.warning(f"ตั้ง Leverage บน exchange ไม่สำเร็จ (ตั้งเองในแอปก่อนรันก็ได้): {e}")

    return exchange


def fetch_ohlcv_df(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ==============================================================================
# ORDER EXECUTION
# ==============================================================================

def place_order(exchange: ccxt.Exchange, side: str, notional_usd: float, price: float, symbol: str) -> dict | None:
    qty_units = notional_usd / price
    if CONFIG["DRY_RUN"]:
        log.info(f"[DRY_RUN] ORDER {side.upper()} notional=${notional_usd:.2f} qty={qty_units:.8f} {symbol}")
        return {"dry_run": True, "side": side, "amount": qty_units}
    try:
        order = exchange.create_order(symbol, type="market", side=side, amount=qty_units)
        log.info(f"[LIVE] ORDER {side.upper()} notional=${notional_usd:.2f} qty={qty_units:.8f} {symbol} -> id={order.get('id')}")
        return order
    except Exception as e:
        log.error(f"ส่งออเดอร์ล้มเหลว: {e}")
        return None


def entry_fee_usd(margin: float) -> float:
    return (margin * CONFIG["LEVERAGE"]) * (CONFIG["FEE_PERCENT"] / 100.0)


# ==============================================================================
# SIGNAL EVALUATION (แยกต่อ Timeframe, ไม่ repaint)
# ==============================================================================

def evaluate_tf_signal(exchange: ccxt.Exchange, symbol: str, tf: str, limit: int) -> dict:
    df = fetch_ohlcv_df(exchange, symbol, tf, limit)
    df["cci"] = compute_cci(df, CONFIG["CCI_LEN"])
    df["vol_ma"] = df["volume"].rolling(CONFIG["VOL_MA_LEN"]).mean()
    df["ema"] = compute_ema(df["close"], CONFIG["EMA_LEN"])

    # ใช้ข้อมูลของ "แท่งที่ปิดแล้ว" เท่านั้น (index -2) เพื่อไม่ให้เกิด repaint
    # เทียบเท่ากับการใช้ [1] shift ในโค้ด Pine ต้นฉบับ
    cci_prev = df["cci"].iloc[-3]
    cci_now = df["cci"].iloc[-2]
    close_now = df["close"].iloc[-2]
    ema_now = df["ema"].iloc[-2]
    vol_now = df["volume"].iloc[-2]
    vol_ma_now = df["vol_ma"].iloc[-2]

    is_whale = vol_now > (vol_ma_now * CONFIG["VOL_SPIKE_MULT"])
    long_signal = is_whale and (cci_now > 100) and (close_now > ema_now) and (cci_prev <= 100 < cci_now)
    short_signal = is_whale and (cci_now < -100) and (close_now < ema_now) and (cci_prev >= -100 > cci_now)

    # เข้าออเดอร์ที่ราคา "open" ของแท่งปัจจุบัน (แท่งถัดจากแท่งที่ยืนยันสัญญาณ)
    # เหมือนกับที่อินดิเกเตอร์ต้นฉบับใช้ open ตอนคำนวณ TP/SL
    entry_price = df["open"].iloc[-1]
    last_high = df["high"].iloc[-2]
    last_low = df["low"].iloc[-2]

    return {
        "long_signal": long_signal,
        "short_signal": short_signal,
        "entry_price": entry_price,
        "high": last_high,
        "low": last_low,
        "is_whale": is_whale,
        "cci": cci_now,
    }


# ==============================================================================
# เปิดไม้ใหม่ (ต่อ Timeframe อิสระ)
# ==============================================================================

def try_open_for_tf(state: AccountState, exchange: ccxt.Exchange, symbol: str, tf: str, tf_cfg: dict, sig: dict) -> None:
    if state.stopped or not tf_cfg["ENABLE"]:
        return

    stats = state.stats_for(tf)
    margin = state.current_margin()

    if stats.active_count >= tf_cfg["MAX_TRADES"] or state.equity < margin:
        return

    if not (sig["long_signal"] or sig["short_signal"]):
        return

    side = 1 if sig["long_signal"] else -1
    entry_price = sig["entry_price"]
    notional = margin * CONFIG["LEVERAGE"]

    order_side = "buy" if side == 1 else "sell"
    order = place_order(exchange, order_side, notional, entry_price, symbol)
    if order is None:
        return

    fee = entry_fee_usd(margin)
    state.equity -= fee
    state.total_fee_paid += fee

    tp_price = entry_price * (1 + tf_cfg["TP_PERCENT"] / 100.0) if side == 1 else entry_price * (1 - tf_cfg["TP_PERCENT"] / 100.0)
    sl_price = entry_price * (1 - tf_cfg["SL_PERCENT"] / 100.0) if side == 1 else entry_price * (1 + tf_cfg["SL_PERCENT"] / 100.0)

    state.trades.append(Trade(side=side, margin=margin, tp_price=tp_price, sl_price=sl_price, tf=tf))
    stats.active_count += 1

    log.info(f"[{tf}] เปิด {'LONG' if side == 1 else 'SHORT'} @ open={entry_price:.4f} | "
             f"margin=${margin:.2f} x{CONFIG['LEVERAGE']} | fee=${fee:.4f} | ไม้ {tf} ตอนนี้ {stats.active_count}/{tf_cfg['MAX_TRADES']}")


# ==============================================================================
# ปิดไม้ (เช็ค SL ก่อน TP เสมอ ต่อ Timeframe — เหมือนต้นฉบับ)
# ==============================================================================

def try_close_for_tf(state: AccountState, exchange: ccxt.Exchange, symbol: str, tf: str, tf_cfg: dict, sig: dict) -> None:
    if state.stopped:
        return

    high, low = sig["high"], sig["low"]
    remaining = []

    for tr in state.trades:
        if tr.tf != tf:
            remaining.append(tr)
            continue

        closed, pnl, exit_type = False, 0.0, None
        exit_fee = entry_fee_usd(tr.margin)  # ค่าธรรมเนียมขาออก คิดจากมาร์จิ้นเดิมของไม้นั้น

        if tr.side == 1:  # Long
            if low <= tr.sl_price:
                pnl = -(tr.margin * (tf_cfg["SL_PERCENT"] / 100.0) * CONFIG["LEVERAGE"]) - exit_fee
                exit_type, closed = "SL", True
            elif high >= tr.tp_price:
                pnl = (tr.margin * (tf_cfg["TP_PERCENT"] / 100.0) * CONFIG["LEVERAGE"]) - exit_fee
                exit_type, closed = "TP", True
        else:  # Short
            if high >= tr.sl_price:
                pnl = -(tr.margin * (tf_cfg["SL_PERCENT"] / 100.0) * CONFIG["LEVERAGE"]) - exit_fee
                exit_type, closed = "SL", True
            elif low <= tr.tp_price:
                pnl = (tr.margin * (tf_cfg["TP_PERCENT"] / 100.0) * CONFIG["LEVERAGE"]) - exit_fee
                exit_type, closed = "TP", True

        if not closed:
            remaining.append(tr)
            continue

        close_side = "sell" if tr.side == 1 else "buy"
        exit_price = tr.sl_price if exit_type == "SL" else tr.tp_price
        order = place_order(exchange, close_side, tr.margin * CONFIG["LEVERAGE"], exit_price, symbol)
        if order is None:
            remaining.append(tr)  # ปิดไม่สำเร็จ ถือไว้ต่อ ลองรอบหน้า
            continue

        state.equity += pnl
        state.total_fee_paid += exit_fee
        stats = state.stats_for(tf)
        stats.active_count -= 1
        if pnl > 0:
            stats.win_count += 1
        else:
            stats.loss_count += 1

        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "tf": tf,
            "side": "LONG" if tr.side == 1 else "SHORT",
            "margin": tr.margin,
            "exit_type": exit_type,
            "pnl": pnl,
            "equity_after": state.equity,
        }
        state.trade_log.append(row)
        _append_trade_csv(row)

        log.info(f"[{tf}] ปิดไม้ {row['side']} {exit_type} | PnL = {pnl:+.4f} USD | Equity = {state.equity:.4f} USD")

    state.trades = remaining


def _append_trade_csv(row: dict) -> None:
    path = CONFIG["TRADE_LOG_CSV"]
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


# ==============================================================================
# KILL SWITCH — ปิดทุกไม้ (ทุก TF) เมื่อ Equity รวม <= 0
# ==============================================================================

def check_kill_switch(state: AccountState, exchange: ccxt.Exchange, symbol: str) -> None:
    if not CONFIG["STOP_WHEN_BLOWN"] or state.stopped:
        return
    if state.equity > 0:
        return

    log.warning("⚠ พอร์ตติดลบ (Equity <= 0) — กำลังปิดโพซิชันทั้งหมดทุก TF และหยุดบอท")
    for tr in state.trades:
        close_side = "sell" if tr.side == 1 else "buy"
        place_order(exchange, close_side, tr.margin * CONFIG["LEVERAGE"], tr.sl_price, symbol)
        exit_fee = entry_fee_usd(tr.margin)
        pnl = -(tr.margin * (CONFIG["TIMEFRAMES"][tr.tf]["SL_PERCENT"] / 100.0) * CONFIG["LEVERAGE"]) - exit_fee
        state.equity += pnl
        stats = state.stats_for(tr.tf)
        stats.active_count -= 1
        stats.loss_count += 1

    state.trades = []
    state.stopped = True


# ==============================================================================
# DASHBOARD
# ==============================================================================

def print_dashboard(state: AccountState) -> None:
    status = "STOPPED (พอร์ตแตก)" if state.stopped else "RUNNING"
    lines = [
        "┌── WHALE HUNTER MULTI-TF DASHBOARD ─────────────────────────",
        f"│ สถานะบอท        : {status}",
        f"│ ทุนเริ่มต้น       : {CONFIG['START_CAPITAL_USD']:.2f} USD",
        f"│ Equity ปัจจุบัน   : {state.equity:.4f} USD",
        f"│ Growth %        : {state.growth_pct:+.2f}%",
        f"│ ค่าธรรมเนียมรวม   : {state.total_fee_paid:.4f} USD",
        f"│ วันที่ผ่านมา       : {state.days_passed() + 1} วัน (มาร์จิ้นปัจจุบัน ${state.current_margin():.2f}/ไม้)",
    ]
    for tf, tf_cfg in CONFIG["TIMEFRAMES"].items():
        if not tf_cfg["ENABLE"]:
            continue
        stats = state.stats_for(tf)
        total = stats.win_count + stats.loss_count
        winrate = (stats.win_count / total * 100.0) if total > 0 else 0.0
        lines.append(f"│ --- TF {tf} ---")
        lines.append(f"│   ไม้เปิดอยู่ / สูงสุด : {stats.active_count} / {tf_cfg['MAX_TRADES']}")
        lines.append(f"│   Win / Loss         : {stats.win_count} / {stats.loss_count}  (Winrate {winrate:.2f}%)")
        lines.append(f"│   TP / SL            : {tf_cfg['TP_PERCENT']}% / {tf_cfg['SL_PERCENT']}%")
    lines.append("└─────────────────────────────────────────────────────────────")
    log.info("\n".join(lines))


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main() -> None:
    symbol = CONFIG["SYMBOL"]
    exchange = build_exchange()
    state = AccountState(equity=CONFIG["START_CAPITAL_USD"], start_time_ms=int(time.time() * 1000))

    enabled_tfs = [tf for tf, c in CONFIG["TIMEFRAMES"].items() if c["ENABLE"]]
    log.info(f"เริ่มบอท Whale Hunter MTF | Exchange={CONFIG['EXCHANGE_ID']} | Symbol={symbol} | "
             f"DRY_RUN={CONFIG['DRY_RUN']} | Leverage={CONFIG['LEVERAGE']}x | TFs={enabled_tfs}")

    if not CONFIG["DRY_RUN"] and (not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]):
        log.error("DRY_RUN=False แต่ไม่พบ API_KEY/API_SECRET — หยุดการทำงานเพื่อความปลอดภัย")
        sys.exit(1)

    last_dashboard_ts = 0.0

    try:
        while not state.stopped:
            try:
                for tf, tf_cfg in CONFIG["TIMEFRAMES"].items():
                    if not tf_cfg["ENABLE"]:
                        continue
                    sig = evaluate_tf_signal(exchange, symbol, tf, tf_cfg["OHLCV_LIMIT"])
                    try_close_for_tf(state, exchange, symbol, tf, tf_cfg, sig)
                    try_open_for_tf(state, exchange, symbol, tf, tf_cfg, sig)

                check_kill_switch(state, exchange, symbol)

                now = time.time()
                if now - last_dashboard_ts >= 30:
                    print_dashboard(state)
                    last_dashboard_ts = now

            except ccxt.NetworkError as e:
                log.warning(f"NetworkError: {e} — ลองใหม่รอบถัดไป")
            except ccxt.ExchangeError as e:
                log.error(f"ExchangeError: {e}")
            except Exception as e:
                log.exception(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")

            time.sleep(CONFIG["POLL_SECONDS"])

    except KeyboardInterrupt:
        log.info("ได้รับคำสั่งหยุดจากผู้ใช้ (Ctrl+C)")

    print_dashboard(state)
    if state.stopped:
        log.warning("บอทหยุดทำงานเนื่องจากพอร์ตติดลบ (Kill Switch)")
    log.info("จบการทำงานของบอท")


if __name__ == "__main__":
    main()

# ==============================================================================
# หมายเหตุการใช้งาน
# ==============================================================================
# 1) ติดตั้งไลบรารีก่อนรัน:
#       pip install ccxt pandas
#
# 2) ตั้งค่า API Key/Secret ผ่าน environment variable (ไม่ hardcode ในไฟล์):
#       export EXCHANGE_API_KEY="xxxxx"
#       export EXCHANGE_API_SECRET="yyyyy"
#    แล้วรัน:
#       python whale_hunter_bot.py
#
# 3) ปรับ TP/SL/Max Trades ของแต่ละ Timeframe ได้อิสระที่ CONFIG["TIMEFRAMES"]
#    เช่น ถ้าต้องการปิด TF ไหนชั่วคราว ตั้ง "ENABLE": False ของ TF นั้น
#    ถ้าต้องการเพิ่ม TF ที่ 3 (เช่น 15m) ก็เพิ่ม key ใหม่ในดิกชันนารีนี้ได้เลย
#
# 4) การคำนวณ PnL ต่อไม้ใช้สูตรเดียวกับอินดิเกเตอร์ต้นฉบับ คือคิดจาก
#    margin x (TP%/SL% ของ TF นั้น) x Leverage แบบตรงไปตรงมา (ไม่ได้คำนวณจาก
#    ส่วนต่างราคาจริงที่ TP/SL แตะ) เพื่อให้ผลลัพธ์ตรงกับอินดิเกเตอร์ตัวต้นแบบ
#    ข้อดีคือ backtest/paper-trade ตรงกับที่เห็นบนกราฟ ข้อเสียคือถ้าราคา slippage
#    ผ่านจุด TP/SL ไปมาก ตัวเลขจริงบน exchange อาจต่างจากตัวเลขในแดชบอร์ดเล็กน้อย
#
# 5) เพราะมี Leverage และ Short จริง ต้องตั้ง MARKET_TYPE = "future" และใช้
#    symbol ที่ตรงกับ futures market ของ exchange นั้น ๆ
#
# 6) ทดสอบด้วย DRY_RUN=True และ/หรือ USE_TESTNET=True ก่อนเสมอ จนกว่าจะมั่นใจ
#    ผลลัพธ์ที่ได้ ก่อนเปลี่ยนเป็นเทรดเงินจริง
#
# 7) Kill Switch: ถ้า Equity รวม (ของทั้งบัญชี ไม่ใช่แยกต่อ TF) <= 0 บอทจะปิด
#    ไม้ที่เปิดอยู่ทั้งหมดในทุก TF แล้วออกจากลูปทันที (จบการทำงานจริง)
# ==============================================================================
