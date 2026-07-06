import os
import time
import math
import ccxt
import requests
import numpy as np  
import pandas as pd
from dotenv import load_dotenv
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

# --- CONFIGURATION FROM USER DASHBOARD ---
load_dotenv()
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

exchange = ccxt.bingx({
    'apiKey': BINGX_API_KEY,
    'secret': BINGX_SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'} 
})

# 🎯 เปลี่ยนคู่เหรียญและไทม์เฟรมเป็น LAB/USDT และ 1m
SYMBOL = 'LAB/USDT:USDT'
TIMEFRAME = '1m'
LEVERAGE = 150                   # Leverage 150x

# --- ตั้งค่าระบบคำนวณ Margin ทบรายวัน ---
BOT_START_DATE = "2026-07-07"   
BASE_MARGIN = 1.0               
DAILY_ADD = 3.0               

# ====================================================
# ⚙️ PARAMETERS ชุดที่ 1: Dual-ChoCh Setup
# ====================================================
FAST_LEFT_BARS = 1
FAST_RIGHT_BARS = 1
SLOW_LEFT_BARS = 20
SLOW_RIGHT_BARS = 20
DC_EMA_LEN = 200
DC_RSI_LEN = 14
DC_RSI_TRIGGER = 50
DC_STOCH_K_LEN = 5
DC_STOCH_D_LEN = 4
DC_STOCH_OS = 20
DC_STOCH_OB = 80
DC_TP_PERCENT = 5.0
DC_SL_PERCENT = 5.0

# ====================================================
# ⚙️ PARAMETERS ชุดที่ 2: Trend Pullback Setup
# ====================================================
TPB_EMA_LEN = 200
TPB_RSI_LEN = 14
TPB_RSI_TRIGGER = 50
TPB_STOCH_K_LEN = 5
TPB_STOCH_D_LEN = 3              
TPB_STOCH_OS = 30                
TPB_STOCH_OB = 70                
TPB_TP_PERCENT = 5.0
TPB_SL_PERCENT = 5.0

# --- GLOBAL VARIABLES FOR ANTI-DOUBLE FIRE & TRACKING ---
last_shot_dc_long = 0
last_shot_dc_short = 0
last_shot_tpb_long = 0
last_shot_tpb_short = 0

fast_choch_dir = 0
slow_choch_dir = 0

def init_exchange_settings():
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"⚙️ Set Leverage to {LEVERAGE}x for {SYMBOL} successfully.")
    except Exception as e:
        print(f"⚠️ Warning setting leverage: {e}")

class SimpleHTTPServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"LAB Dual-Strategy (Dual-ChoCh + Trend Pullback) 1m Bot is running online!")
    def log_message(self, format, *args): return

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPServer)
    server.serve_forever()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
    except: pass

# --- TECHNICAL INDICATOR FUNCTIONS ---
def calculate_ema(df, length):
    return df['close'].ewm(span=length, adjust=False).mean()

def calculate_rsi(df, length):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic_kd(df, k_len, d_len):
    low_min = df['low'].rolling(window=k_len).min()
    high_max = df['high'].rolling(window=k_len).max()
    stoch_raw = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_k = stoch_raw.rolling(window=d_len).mean()
    stoch_d = stoch_k.rolling(window=d_len).mean()
    return stoch_k, stoch_d

def calculate_pivots(df, left, right):
    pivothigh = [np.nan] * len(df)
    pivotlow = [np.nan] * len(df)
    for i in range(left, len(df) - right):
        center_high = df['high'].iloc[i]
        center_low = df['low'].iloc[i]
        if all(center_high > df['high'].iloc[i - j] for j in range(1, left + 1)) and \
           all(center_high >= df['high'].iloc[i + j] for j in range(1, right + 1)):
            pivothigh[i + right] = center_high
        if all(center_low < df['low'].iloc[i - j] for j in range(1, left + 1)) and \
           all(center_low <= df['low'].iloc[i + j] for j in range(1, right + 1)):
            pivotlow[i + right] = center_low
    return pd.Series(pivothigh), pd.Series(pivotlow)

def fire_execution_order(strategy_name, side, estimated_price, margin_size, tp_pct, sl_pct):
    try:
        # คำนวณขนาดสัญญารองรับทศนิยมของเหรียญ LAB
        contract_amount = round((margin_size * LEVERAGE) / estimated_price, 2)
        if contract_amount < 0.1: contract_amount = 0.1 
            
        order_side = 'buy' if side == "LONG" else 'sell'
        emoji = "🟩🚀" if side == "LONG" else "🟥💥"

        main_order = exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        entry_price = float(main_order.get('price', 0)) or float(main_order.get('average', 0)) or estimated_price
        
        tp_factor = tp_pct / 100
        sl_factor = sl_pct / 100
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), 4)
            sl_price = round(entry_price * (1 - sl_factor), 4)
        else:
            tp_price = round(entry_price * (1 - tp_factor), 4)
            sl_price = round(entry_price * (1 + sl_factor), 4)

        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = f"{emoji} *[{strategy_name}]* LAB ยิงสำเร็จ!\n• *ฝั่ง:* {side}\n• *ราคาทุน:* ${entry_price}\n• *Margin:* ${margin_size:.2f}\n🎯 *TP ({tp_pct}%):* ${tp_price} | 🛑 *SL ({sl_pct}%):* ${sl_price}"
        send_telegram_message(tg_msg)
    except Exception as e: 
        print(f"Error {strategy_name} LAB order: {e}")
        send_telegram_message(f"⚠️ *[{strategy_name} - LAB ยิงพลาด!]*\n• *สาเหตุ:* `{str(e)}`")

# --- MAIN LOOP ENGINE ---
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    init_exchange_settings()

    print("🟢 LAB/USDT Futures 1m Engine is Active...")
    send_telegram_message("🤖 *[LAB/USDT 1m]* บอทปรับเข้าสู่โหมดรันเหรียญ LAB ที่ไทม์เฟรม 1 นาที เรียบร้อยครับพี่!")

    last_heartbeat_time = 0.0

    while True:
        try:
            start_dt = datetime.strptime(BOT_START_DATE, "%Y-%m-%d")
            now_dt = datetime.now()
            days_passed = (now_dt - start_dt).days
            if days_passed < 0: days_passed = 0
            current_mgn_active = BASE_MARGIN + (days_passed * DAILY_ADD)

            # ดึงข้อมูลแท่งเทียน 1m
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=500)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            live_price = df.iloc[-1]['close']
            
            idx = len(df) - 2 
            candle_timestamp = df.iloc[idx]['timestamp']
            readable_candle_time = pd.to_datetime(candle_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M')

            # ====================================================
            # 🛡️ ENGINE 1: Dual-ChoCh Logic
            # ====================================================
            df['dc_ema'] = calculate_ema(df, DC_EMA_LEN)
            df['dc_rsi'] = calculate_rsi(df, DC_RSI_LEN)
            df['dc_stoch_k'], df['dc_stoch_d'] = calculate_stochastic_kd(df, DC_STOCH_K_LEN, DC_STOCH_D_LEN)
            df['f_p_high'], df['f_p_low'] = calculate_pivots(df, FAST_LEFT_BARS, FAST_RIGHT_BARS)
            df['s_p_high'], df['s_p_low'] = calculate_pivots(df, SLOW_LEFT_BARS, SLOW_RIGHT_BARS)

            for i in range(len(df)):
                if not pd.isna(df['f_p_high'].iloc[i]): f_ph_val = df['f_p_high'].iloc[i]
                if not pd.isna(df['f_p_low'].iloc[i]):  f_pl_val = df['low'].iloc[i - FAST_RIGHT_BARS]
                if not pd.isna(f_ph_val) and df['close'].iloc[i] > f_ph_val and fast_choch_dir != 1: fast_choch_dir = 1
                if not pd.isna(f_pl_val) and df['close'].iloc[i] < f_pl_val and fast_choch_dir != -1: fast_choch_dir = -1

                if not pd.isna(df['s_p_high'].iloc[i]): s_ph_val = df['s_p_high'].iloc[i]
                if not pd.isna(df['s_p_low'].iloc[i]):  s_pl_val = df['low'].iloc[i - SLOW_RIGHT_BARS]
                if not pd.isna(s_ph_val) and df['close'].iloc[i] > s_ph_val and slow_choch_dir != 1: slow_choch_dir = 1
                if not pd.isna(s_pl_val) and df['close'].iloc[i] < s_pl_val and slow_choch_dir != -1: slow_choch_dir = -1

            dc_trend_bullish = df.iloc[idx]['close'] > df.iloc[idx]['dc_ema']
            dc_pullback_long = (df.iloc[idx]['dc_rsi'] <= DC_RSI_TRIGGER) and (df.iloc[idx]['dc_stoch_k'] < DC_STOCH_OS)
            dc_stoch_cross_up = (df.iloc[idx-1]['dc_stoch_k'] <= df.iloc[idx-1]['dc_stoch_d']) and (df.iloc[idx]['dc_stoch_k'] > df.iloc[idx]['dc_stoch_d'])
            dc_base_long = dc_trend_bullish and dc_pullback_long and dc_stoch_cross_up

            dc_trend_bearish = df.iloc[idx]['close'] < df.iloc[idx]['dc_ema']
            dc_pullback_short = (df.iloc[idx]['dc_rsi'] >= (100 - DC_RSI_TRIGGER)) and (df.iloc[idx]['dc_stoch_k'] > DC_STOCH_OB)
            dc_stoch_cross_down = (df.iloc[idx-1]['dc_stoch_k'] >= df.iloc[idx-1]['dc_stoch_d']) and (df.iloc[idx]['dc_stoch_k'] < df.iloc[idx]['dc_stoch_d'])
            dc_base_short = dc_trend_bearish and dc_pullback_short and dc_stoch_cross_down

            dc_long_sig, dc_short_sig = False, False
            if dc_base_long:
                if slow_choch_dir == 1: dc_long_sig = True
                elif slow_choch_dir == -1: dc_short_sig = True
            if dc_base_short:
                if slow_choch_dir == -1: dc_short_sig = True
                elif slow_choch_dir == 1: dc_long_sig = True

            # ====================================================
            # 🛡️ ENGINE 2: Trend Pullback Logic
            # ====================================================
            df['tpb_ema'] = calculate_ema(df, TPB_EMA_LEN)
            df['tpb_rsi'] = calculate_rsi(df, TPB_RSI_LEN)
            df['tpb_stoch_k'], df['tpb_stoch_d'] = calculate_stochastic_kd(df, TPB_STOCH_K_LEN, TPB_STOCH_D_LEN)

            tpb_trend_bullish = df.iloc[idx]['close'] > df.iloc[idx]['tpb_ema']
            tpb_pullback_long = (df.iloc[idx]['tpb_rsi'] <= TPB_RSI_TRIGGER) and (df.iloc[idx]['tpb_stoch_k'] < TPB_STOCH_OS)
            tpb_stoch_cross_up = (df.iloc[idx-1]['tpb_stoch_k'] <= df.iloc[idx-1]['tpb_stoch_d']) and (df.iloc[idx]['tpb_stoch_k'] > df.iloc[idx]['tpb_stoch_d'])
            tpb_long_sig = tpb_trend_bullish and tpb_pullback_long and tpb_stoch_cross_up

            tpb_trend_bearish = df.iloc[idx]['close'] < df.iloc[idx]['tpb_ema']
            tpb_pullback_short = (df.iloc[idx]['tpb_rsi'] >= (100 - TPB_RSI_TRIGGER)) and (df.iloc[idx]['tpb_stoch_k'] > TPB_STOCH_OB)
            tpb_stoch_cross_down = (df.iloc[idx-1]['tpb_stoch_k'] >= df.iloc[idx-1]['tpb_stoch_d']) and (df.iloc[idx]['tpb_stoch_k'] < df.iloc[idx]['tpb_stoch_d'])
            tpb_short_sig = tpb_trend_bearish and tpb_pullback_short and tpb_stoch_cross_down

            # ====================================================
            # 🚀 EXECUTION & ANTI-DOUBLE FIRE CONTROL (1m Scale)
            # ====================================================
            # --- กลยุทธ์ที่ 1: Dual-ChoCh ---
            if dc_long_sig and candle_timestamp != last_shot_dc_long:
                fire_execution_order("Dual-ChoCh", "LONG", live_price, current_mgn_active, DC_TP_PERCENT, DC_SL_PERCENT)
                last_shot_dc_long = candle_timestamp
            elif dc_short_sig and candle_timestamp != last_shot_dc_short:
                fire_execution_order("Dual-ChoCh", "SHORT", live_price, current_mgn_active, DC_TP_PERCENT, DC_SL_PERCENT)
                last_shot_dc_short = candle_timestamp

            # --- กลยุทธ์ที่ 2: Trend Pullback ---
            if tpb_long_sig and candle_timestamp != last_shot_tpb_long:
                fire_execution_order("Trend-Pullback", "LONG", live_price, current_mgn_active, TPB_TP_PERCENT, TPB_SL_PERCENT)
                last_shot_tpb_long = candle_timestamp
            elif tpb_short_sig and candle_timestamp != last_shot_tpb_short:
                fire_execution_order("Trend-Pullback", "SHORT", live_price, current_mgn_active, TPB_TP_PERCENT, TPB_SL_PERCENT)
                last_shot_tpb_short = candle_timestamp

            # --- Heartbeat ทุกๆ 1 ชม. ---
            current_time = time.time()
            if current_time - last_heartbeat_time >= 3600:
                try: bal = exchange.fetch_balance(); total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
                except: total_cap = 0.0
                heartbeat_msg = (
                    f"🤖 *[LAB 1m Bot - รายงานตัว]*\n"
                    f"• *คู่เหรียญ:* `LAB/USDT` (ไทม์เฟรม 1m)\n"
                    f"• *ทุนสุทธิพอร์ตหลัก:* ${total_cap}\n"
                    f"• *Margin ไม้ถัดไป:* ${current_mgn_active:.2f}"
                )
                send_telegram_message(heartbeat_msg)
                last_heartbeat_time = current_time

            print(f"🔄 [1m Loop] LAB Price: {live_price} | Time: {readable_candle_time}")

        except Exception as ex:
            print(f"Loop Error: {ex}")
            
        # ขยับเวลามาลูปเร็วขึ้นทุกๆ 5 วินาที เพื่อให้ทันความเร็วของแท่ง 1m
        time.sleep(5)
