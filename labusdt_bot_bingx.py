#!/usr/bin/env python3
# ==============================================================================
# Whale Hunter Bot — Dual Timeframe (3m + 5m independent engines)
# ------------------------------------------------------------------------------
# แปลงเงื่อนไขจากอินดิเกเตอร์ Pine Script "Whale Hunter V10 - Real Fee & Growth
# (No Repaint)" มาเป็นบอทเทรดจริงด้วย Python + ccxt
#
# เงื่อนไขเข้าไม้ (เหมือนอินดิเกเตอร์ทุกประการ):
#   - Volume Spike: volume ปัจจุบัน > SMA(volume,20) x 2.0
#   - Long : CCI(20) ตัดขึ้นผ่าน 100  และ close > EMA(200)  และ CCI ปัจจุบัน > 100
#   - Short: CCI(20) ตัดลงผ่าน -100 และ close < EMA(200)  และ CCI ปัจจุบัน < -100
#   - เช็คสัญญาณจากแท่งที่ "ปิดแล้ว" เท่านั้น (No Repaint เหมือนต้นฉบับที่ใช้ [1])
#
# จุดสำคัญตามที่ขอ: บอทนี้ "เฝ้ามอง 2 timeframe พร้อมกัน" คือ 3 นาที และ 5 นาที
# โดยใช้เงื่อนไขเข้าไม้ชุดเดียวกันทั้งคู่ (ไม่ใช่การ confirm ข้าม TF แบบบอทก่อนหน้า)
# แต่ทำงานเป็น "เอนจินอิสระ 2 ชุด" — แต่ละ TF มีของตัวเองแยกกันโดยสิ้นเชิง:
#   - TP / SL (%) แยกกัน
#   - จำนวนไม้เปิดพร้อมกันสูงสุด (max concurrent trades) แยกกัน
#   - มาร์จิ้นต่อไม้ + อัตราการเพิ่มมาร์จิ้นรายวัน + Leverage แยกกัน
#   - รายการโพซิชันที่เปิดอยู่ (positions list) แยกกันคนละ pool
# ส่วน Equity/ทุนรวม และ Kill Switch ใช้บัญชีเดียวกันร่วมกันทั้งสอง TF
# (เพราะเป็นเงินทุนก้อนเดียวกันจริงในบัญชีเทรด)
#
# ⚠️ คำเตือนความปลอดภัย
#   - ค่าเริ่มต้น DRY_RUN = True (โหมดจำลอง ไม่ส่งออเดอร์จริง)
#   - มี Leverage สูง เสี่ยงพอร์ตแตกไว ทดสอบบน Testnet ก่อนใช้เงินจริงเสมอ
#   - ห้าม hardcode API Key/Secret ถ้าจะแชร์โค้ดต่อ ใช้ environment variable แทน
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

SHARED_CONFIG = {
    # ---------------- Exchange / Connection ----------------
    "EXCHANGE_ID": "binance",
    "API_KEY": os.getenv("EXCHANGE_API_KEY", ""),
    "API_SECRET": os.getenv("EXCHANGE_API_SECRET", ""),
    "USE_TESTNET": True,
    "SYMBOL": "BTC/USDT",                  # ถ้าใช้ futures อาจต้องเป็น 'BTC/USDT:USDT'
    "MARKET_TYPE": "future",               # ต้องเป็น 'future' ถึงจะ Short/ใช้ Leverage จริงได้

    # ---------------- Whale Hunter Signal Params (ใช้ร่วมกันทั้ง 2 TF) ----------------
    "CCI_LEN": 20,
    "CCI_LEVEL": 100,                      # OB = +100, OS = -100
    "VOLUME_MA_LEN": 20,
    "VOLUME_SPIKE_MULT": 2.0,
    "EMA_LEN": 200,

    # ---------------- เงินทุนรวม / Kill Switch ----------------
    "START_CAPITAL_USD": 10.0,
    "STOP_WHEN_BLOWN": True,

    # ---------------- Runtime ----------------
    "DRY_RUN": True,
    "POLL_SECONDS": 5,
    "OHLCV_LIMIT": 300,                    # ต้องเยอะพอสำหรับ EMA200
    "LOG_FILE": "whale_hunter_bot.log",
    "TRADE_LOG_CSV": "whale_hunter_trades.csv",
}

# พารามิเตอร์ที่ "แยกกันคนละ TF" ตามที่ขอ: TP/SL, Leverage, มาร์จิ้น, จำนวนไม้สูงสุด
TF_CONFIGS = {
    "3m": {
        "TIMEFRAME": "3m",
        "BASE_MARGIN_USD": 1.0,            # เงินต้นเริ่มต้นต่อไม้
        "DAILY_ADD_USD": 1.0,              # เพิ่มเงินต้นต่อไม้วันละเท่านี้ (ไล่ตามต้นฉบับ)
        "LEVERAGE": 20.0,
        "FEE_PERCENT": 0.04,               # % ของ Notional ต่อฝั่ง (เข้า/ออก)
        "MAX_TRADES": 3,                   # จำนวนไม้เปิดพร้อมกันสูงสุดของ TF นี้
        "TP_PERCENT": 3.0,
        "SL_PERCENT": 5.0,
    },
    "5m": {
        "TIMEFRAME": "5m",
        "BASE_MARGIN_USD": 1.0,
        "DAILY_ADD_USD": 1.0,
        "LEVERAGE": 20.0,
        "FEE_PERCENT": 0.04,
        "MAX_TRADES": 3,
        "TP_PERCENT": 3.0,
        "SL_PERCENT": 5.0,
    },
}

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(SHARED_CONFIG["LOG_FILE"], encoding="utf-8"),
    ],
)
log = logging.getLogger("WhaleHunter")


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


def compute_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


# ==============================================================================
# STATE
# ==============================================================================

@dataclass
class Position:
    side: int          # 1 = Long, -1 = Short
    entry_price: float
    qty_units: float
    margin: float
    notional: float
    tf_name: str


@dataclass
class TFEngine:
    """เอนจินอิสระของแต่ละ Timeframe: เก็บโพซิชัน/สถิติ/พารามิเตอร์ของตัวเอง"""
    name: str
    cfg: dict
    positions: list = field(default_factory=list)
    total_trades: int = 0
    win_trades: int = 0
    lose_trades: int = 0

    @property
    def winrate_pct(self) -> float:
        return (self.win_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0

    def current_margin(self, days_passed: int) -> float:
        return self.cfg["BASE_MARGIN_USD"] + (days_passed * self.cfg["DAILY_ADD_USD"])

    def open_mark_value(self, last_price: float) -> float:
        total = 0.0
        for p in self.positions:
            total += (last_price - p.entry_price) * p.qty_units if p.side == 1 else (p.entry_price - last_price) * p.qty_units
        return total


@dataclass
class AccountState:
    equity: float
    start_time: datetime
    total_fee_paid: float = 0.0
    stopped: bool = False
    trade_log: list = field(default_factory=list)

    @property
    def net_profit(self) -> float:
        return self.equity - SHARED_CONFIG["START_CAPITAL_USD"]

    @property
    def net_profit_pct(self) -> float:
        start = SHARED_CONFIG["START_CAPITAL_USD"]
        return (self.net_profit / start * 100.0) if start else 0.0

    @property
    def days_passed(self) -> int:
        delta = datetime.now(timezone.utc) - self.start_time
        return int(delta.total_seconds() // 86400)


# ==============================================================================
# EXCHANGE CLIENT
# ==============================================================================

def build_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, SHARED_CONFIG["EXCHANGE_ID"])
    exchange = exchange_class({
        "apiKey": SHARED_CONFIG["API_KEY"],
        "secret": SHARED_CONFIG["API_SECRET"],
        "enableRateLimit": True,
        "options": {"defaultType": SHARED_CONFIG["MARKET_TYPE"]},
    })
    if SHARED_CONFIG["USE_TESTNET"] and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
        log.info("เปิดใช้งาน Testnet/Sandbox mode")

    if not SHARED_CONFIG["DRY_RUN"] and SHARED_CONFIG["MARKET_TYPE"] == "future":
        for tf_name, cfg in TF_CONFIGS.items():
            try:
                exchange.set_leverage(int(cfg["LEVERAGE"]), SHARED_CONFIG["SYMBOL"])
            except Exception as e:
                log.warning(f"ตั้ง Leverage ({tf_name}) บน exchange ไม่สำเร็จ: {e}")

    return exchange


def fetch_ohlcv_df(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ==============================================================================
# ORDER EXECUTION
# ==============================================================================

def fee_usd(notional_usd: float, fee_percent: float) -> float:
    return notional_usd * fee_percent / 100.0


def place_order(exchange: ccxt.Exchange, side: str, qty_units: float, symbol: str) -> dict | None:
    if SHARED_CONFIG["DRY_RUN"]:
        log.info(f"[DRY_RUN] ORDER {side.upper()} qty={qty_units:.8f} {symbol}")
        return {"dry_run": True, "side": side, "amount": qty_units}
    try:
        order = exchange.create_order(symbol, type="market", side=side, amount=qty_units)
        log.info(f"[LIVE] ORDER {side.upper()} qty={qty_units:.8f} {symbol} -> id={order.get('id')}")
        return order
    except Exception as e:
        log.error(f"ส่งออเดอร์ล้มเหลว: {e}")
        return None


# ==============================================================================
# SIGNAL EVALUATION (Whale Hunter logic, no-repaint) — ใช้ร่วมกันทั้ง 2 TF
# ==============================================================================

def evaluate_whale_signals(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> dict:
    df = fetch_ohlcv_df(exchange, symbol, timeframe, SHARED_CONFIG["OHLCV_LIMIT"])
    df["cci"] = compute_cci(df, SHARED_CONFIG["CCI_LEN"])
    df["ema"] = compute_ema(df["close"], SHARED_CONFIG["EMA_LEN"])
    df["vol_ma"] = compute_sma(df["volume"], SHARED_CONFIG["VOLUME_MA_LEN"])

    lvl = SHARED_CONFIG["CCI_LEVEL"]
    mult = SHARED_CONFIG["VOLUME_SPIKE_MULT"]

    # ใช้แท่งที่ "ปิดแล้ว" 2 แท่งล่าสุด (index -2 ปัจจุบัน, -3 ก่อนหน้า) เพื่อคำนวณ crossover
    # โดยไม่แตะแท่งที่กำลังวิ่งอยู่ (index -1) -> เทียบเท่าการใช้ nz(sig[1]) ในต้นฉบับ
    cci_prev, cci_now = df["cci"].iloc[-3], df["cci"].iloc[-2]
    close_now = df["close"].iloc[-2]
    ema_now = df["ema"].iloc[-2]
    vol_now = df["volume"].iloc[-2]
    vol_ma_now = df["vol_ma"].iloc[-2]

    is_whale = vol_now > (vol_ma_now * mult)
    cross_up = (cci_prev <= lvl) and (cci_now > lvl)
    cross_down = (cci_prev >= -lvl) and (cci_now < -lvl)

    long_signal = is_whale and (cci_now > lvl) and (close_now > ema_now) and cross_up
    short_signal = is_whale and (cci_now < -lvl) and (close_now < ema_now) and cross_down

    return {
        "close": close_now,
        "high": df["high"].iloc[-2],
        "low": df["low"].iloc[-2],
        "open": df["open"].iloc[-2],
        "is_whale": bool(is_whale),
        "cci": cci_now,
        "long_signal": bool(long_signal),
        "short_signal": bool(short_signal),
    }


# ==============================================================================
# ENTRY / EXIT ต่อเอนจิน (แต่ละ TF อิสระจากกัน)
# ==============================================================================

def try_close_positions(engine: TFEngine, state: AccountState, exchange: ccxt.Exchange, symbol: str, signals: dict) -> None:
    if state.stopped or not engine.positions:
        return

    high, low = signals["high"], signals["low"]
    tp_p, sl_p = engine.cfg["TP_PERCENT"], engine.cfg["SL_PERCENT"]
    fee_p = engine.cfg["FEE_PERCENT"]
    still_open = []

    for pos in engine.positions:
        # เช็ค SL ก่อนเสมอ (เหมือนต้นฉบับ: ป้องกันกรณีชนพร้อมกันในแท่งเดียว)
        exit_price, exit_type = None, None
        if pos.side == 1:
            sl = pos.entry_price * (1 - sl_p / 100.0)
            tp = pos.entry_price * (1 + tp_p / 100.0)
            if low <= sl:
                exit_price, exit_type = sl, "SL"
            elif high >= tp:
                exit_price, exit_type = tp, "TP"
        else:
            sl = pos.entry_price * (1 + sl_p / 100.0)
            tp = pos.entry_price * (1 - tp_p / 100.0)
            if high >= sl:
                exit_price, exit_type = sl, "SL"
            elif low <= tp:
                exit_price, exit_type = tp, "TP"

        if exit_price is None:
            still_open.append(pos)
            continue

        close_side = "sell" if pos.side == 1 else "buy"
        order = place_order(exchange, close_side, pos.qty_units, symbol)
        if order is None:
            still_open.append(pos)
            continue

        exit_fee = fee_usd(pos.notional, fee_p)
        # PnL คำนวณจาก % ที่โดนจริง x notional (เหมือนสูตรต้นฉบับ margin * pct * leverage)
        pnl_pct = -sl_p if exit_type == "SL" else tp_p
        gross_pnl = pos.margin * (pnl_pct / 100.0) * engine.cfg["LEVERAGE"]
        net_pnl = gross_pnl - exit_fee

        state.equity += net_pnl
        state.total_fee_paid += exit_fee
        engine.total_trades += 1
        if net_pnl > 0:
            engine.win_trades += 1
        else:
            engine.lose_trades += 1

        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "tf": engine.name,
            "side": "LONG" if pos.side == 1 else "SHORT",
            "entry": pos.entry_price,
            "exit": exit_price,
            "exit_type": exit_type,
            "margin": pos.margin,
            "net_pnl": net_pnl,
            "equity_after": state.equity,
        }
        state.trade_log.append(row)
        _append_trade_csv(row)

        log.info(
            f"[{engine.name}] ปิดไม้ {row['side']} {exit_type} @ {exit_price:.4f} | "
            f"PnL สุทธิ = {net_pnl:+.4f} USD | Equity รวม = {state.equity:.4f} USD"
        )

    engine.positions = still_open


def try_open_positions(engine: TFEngine, state: AccountState, exchange: ccxt.Exchange, symbol: str, signals: dict) -> None:
    if state.stopped:
        return
    if len(engine.positions) >= engine.cfg["MAX_TRADES"]:
        return
    if not (signals["long_signal"] or signals["short_signal"]):
        return

    margin = engine.current_margin(state.days_passed)
    if state.equity < margin:
        log.warning(f"[{engine.name}] Equity ไม่พอเปิดไม้ใหม่ (ต้องการ ${margin}, มี ${state.equity:.4f})")
        return

    leverage = engine.cfg["LEVERAGE"]
    notional = margin * leverage
    price = signals["open"] if "open" in signals else signals["close"]
    qty_units = notional / price
    side = 1 if signals["long_signal"] else -1

    order = place_order(exchange, "buy" if side == 1 else "sell", qty_units, symbol)
    if order is None:
        return

    entry_fee = fee_usd(notional, engine.cfg["FEE_PERCENT"])
    state.equity -= entry_fee
    state.total_fee_paid += entry_fee

    engine.positions.append(Position(
        side=side, entry_price=price, qty_units=qty_units,
        margin=margin, notional=notional, tf_name=engine.name,
    ))
    log.info(
        f"[{engine.name}] เปิด {'LONG' if side == 1 else 'SHORT'} ที่ {price:.4f} | "
        f"margin=${margin:.2f} x{leverage} = ${notional:.2f} | Fee เปิด=${entry_fee:.4f} | "
        f"ไม้เปิดอยู่ตอนนี้ {len(engine.positions)}/{engine.cfg['MAX_TRADES']}"
    )


def _append_trade_csv(row: dict) -> None:
    path = SHARED_CONFIG["TRADE_LOG_CSV"]
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


# ==============================================================================
# KILL SWITCH — ปิดทุกไม้ในทุกเอนจินพร้อมกันเมื่อ Equity <= 0
# ==============================================================================

def check_kill_switch(engines: dict, state: AccountState, exchange: ccxt.Exchange, symbol: str, last_prices: dict) -> None:
    if not SHARED_CONFIG["STOP_WHEN_BLOWN"] or state.stopped:
        return
    if state.equity > 0:
        return

    log.warning("⚠ พอร์ตติดลบ (Equity <= 0) — กำลังปิดทุกไม้ในทุก Timeframe และหยุดบอท")
    for tf_name, engine in engines.items():
        last_price = last_prices.get(tf_name)
        for pos in engine.positions:
            close_side = "sell" if pos.side == 1 else "buy"
            place_order(exchange, close_side, pos.qty_units, symbol)
            if last_price is not None:
                mark_pnl = (last_price - pos.entry_price) * pos.qty_units if pos.side == 1 else (pos.entry_price - last_price) * pos.qty_units
            else:
                mark_pnl = 0.0
            fee = fee_usd(pos.notional, engine.cfg["FEE_PERCENT"])
            net_pnl = mark_pnl - fee
            state.equity += net_pnl
            state.total_fee_paid += fee
            engine.total_trades += 1
            if net_pnl > 0:
                engine.win_trades += 1
            else:
                engine.lose_trades += 1
        engine.positions = []

    state.stopped = True


# ==============================================================================
# DASHBOARD
# ==============================================================================

def print_dashboard(engines: dict, state: AccountState, last_prices: dict) -> None:
    unrealized = state.equity
    for tf_name, engine in engines.items():
        lp = last_prices.get(tf_name)
        if lp is not None:
            unrealized += engine.open_mark_value(lp)

    status = "STOPPED (พอร์ตแตก)" if state.stopped else "RUNNING"

    lines = [
        "┌── WHALE HUNTER DASHBOARD (3m + 5m) ─────────────────────────",
        f"│ สถานะบอท          : {status}",
        f"│ ทุนเริ่มต้น         : {SHARED_CONFIG['START_CAPITAL_USD']:.2f} USD",
        f"│ Equity รวม (Realized): {state.equity:.4f} USD",
        f"│ Equity รวม (Mark-to-Market): {unrealized:.4f} USD",
        f"│ กำไร/ขาดทุนสุทธิ    : {state.net_profit:+.4f} USD ({state.net_profit_pct:+.2f}%)",
        f"│ ค่าธรรมเนียมสะสม     : {state.total_fee_paid:.4f} USD",
        f"│ วันที่ผ่านมา         : {state.days_passed + 1} วัน",
        "├──────────────────────────────────────────────────────────",
    ]
    for tf_name, engine in engines.items():
        margin_now = engine.current_margin(state.days_passed)
        lines.append(
            f"│ [{tf_name}] ไม้เปิดอยู่ {len(engine.positions)}/{engine.cfg['MAX_TRADES']} | "
            f"ปิดแล้ว {engine.total_trades} (W{engine.win_trades}/L{engine.lose_trades}) | "
            f"Winrate {engine.winrate_pct:.2f}% | TP {engine.cfg['TP_PERCENT']}% / SL {engine.cfg['SL_PERCENT']}% | "
            f"Margin ตอนนี้ ${margin_now:.2f} x{engine.cfg['LEVERAGE']}"
        )
    lines.append("└──────────────────────────────────────────────────────────")
    log.info("\n".join(lines))


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main() -> None:
    symbol = SHARED_CONFIG["SYMBOL"]
    exchange = build_exchange()
    state = AccountState(equity=SHARED_CONFIG["START_CAPITAL_USD"], start_time=datetime.now(timezone.utc))
    engines = {name: TFEngine(name=name, cfg=cfg) for name, cfg in TF_CONFIGS.items()}

    log.info(f"เริ่มบอท Whale Hunter | Exchange={SHARED_CONFIG['EXCHANGE_ID']} | Symbol={symbol} | "
             f"DRY_RUN={SHARED_CONFIG['DRY_RUN']} | เฝ้ามอง TF: {list(TF_CONFIGS.keys())}")

    if not SHARED_CONFIG["DRY_RUN"] and (not SHARED_CONFIG["API_KEY"] or not SHARED_CONFIG["API_SECRET"]):
        log.error("DRY_RUN=False แต่ไม่พบ API_KEY/API_SECRET — หยุดการทำงานเพื่อความปลอดภัย")
        sys.exit(1)

    last_dashboard_ts = 0.0

    try:
        while not state.stopped:
            last_prices = {}
            try:
                for tf_name, engine in engines.items():
                    signals = evaluate_whale_signals(exchange, symbol, engine.cfg["TIMEFRAME"])
                    last_prices[tf_name] = signals["close"]

                    try_close_positions(engine, state, exchange, symbol, signals)
                    if not state.stopped:
                        try_open_positions(engine, state, exchange, symbol, signals)

                    check_kill_switch(engines, state, exchange, symbol, last_prices)
                    if state.stopped:
                        break

                now = time.time()
                if now - last_dashboard_ts >= 30:
                    print_dashboard(engines, state, last_prices)
                    last_dashboard_ts = now

            except ccxt.NetworkError as e:
                log.warning(f"NetworkError: {e} — ลองใหม่รอบถัดไป")
            except ccxt.ExchangeError as e:
                log.error(f"ExchangeError: {e}")
            except Exception as e:
                log.exception(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")

            time.sleep(SHARED_CONFIG["POLL_SECONDS"])

    except KeyboardInterrupt:
        log.info("ได้รับคำสั่งหยุดจากผู้ใช้ (Ctrl+C)")

    print_dashboard(engines, state, {})
    if state.stopped:
        log.warning("บอทหยุดทำงานเนื่องจากพอร์ตติดลบ (Kill Switch)")
    log.info("จบการทำงานของบอท")


if __name__ == "__main__":
    main()

# ==============================================================================
# หมายเหตุการใช้งาน
# ==============================================================================
# 1) ติดตั้งไลบรารีก่อนรัน:  pip install ccxt pandas
#
# 2) ตั้งค่า API Key/Secret ผ่าน environment variable:
#       export EXCHANGE_API_KEY="xxxxx"
#       export EXCHANGE_API_SECRET="yyyyy"
#       python whale_hunter_bot.py
#
# 3) โครงสร้างสำคัญที่ตอบโจทย์ที่ขอ:
#    - TF_CONFIGS["3m"] และ TF_CONFIGS["5m"] คือพารามิเตอร์ที่แยกกันอิสระต่อ TF
#      (TP_PERCENT, SL_PERCENT, MAX_TRADES, LEVERAGE, BASE_MARGIN_USD, ...)
#    - engines["3m"].positions และ engines["5m"].positions คือ pool ไม้ที่แยก
#      กันคนละ TF โดยสิ้นเชิง ไม่ปนกัน
#    - เงื่อนไขสัญญาณ (evaluate_whale_signals) เป็นฟังก์ชันเดียวกัน เรียกซ้ำ
#      แยกรอบสำหรับแต่ละ TF ในทุกรอบ loop
#    - Equity/เงินทุนรวม และ Kill Switch ใช้ร่วมกันทั้ง 2 TF (บัญชีเดียวกัน)
#
# 4) ทดสอบด้วย DRY_RUN=True และ/หรือ USE_TESTNET=True ก่อนเสมอ โดยเฉพาะกับ
#    Leverage สูงแบบนี้ (20x) ความเสี่ยงพอร์ตแตกไวมากถ้าตั้งค่าผิดพลาด
#
# 5) ข้อจำกัดสำคัญ:
#    - ต้องตั้ง MARKET_TYPE = "future" และใช้ symbol ที่ตรงกับ futures market
#      ของ exchange นั้น ๆ ถึงจะ Short จริงและใช้ Leverage ได้
#    - ราคาที่ใช้เช็ค TP/SL อ้างอิงจาก high/low ของแท่งเทียนที่ปิดแล้ว เป็นการ
#      ประมาณ ไม่ใช่การจับคู่ order แบบ tick-by-tick จริงเหมือนตลาด
#    - โค้ดนี้ไม่ได้เช็ค available margin ของบัญชีจริงก่อนเปิดไม้ใหม่ ควรเพิ่มการ
#      เช็ค exchange.fetch_balance() ก่อนใช้เงินจริง
#    - เมื่อ Kill Switch ทำงาน บอทจะปิดทุกไม้ในทุก TF และออกจาก while loop ทันที
# ==============================================================================
