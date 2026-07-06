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

# --- CONFIGURATION ---
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

# เปลี่ยนตัวแปรสำหรับคริปโตคู่ LABUSDT
SYMBOL = 'LAB/USDT:USDT'

# ==========================================
# ⚙️ SETTING 1: สำหรับไทม์เฟรม 1m
# ==========================================
TIMEFRAME_M1 = '1m'
P_LENGTH_M1 = 23            
SRC_THRESHOLD_M1 = 0.2     
TP_PERCENT_M1 = 3.0        
SL_PERCENT_M1 = 3.0        

# ==========================================
# ⚙️ SETTING 2: สำหรับไทม์เฟรม 5m
# ==========================================
TIMEFRAME_M5 = '5m'
P_LENGTH_M5 = 27            
SRC_THRESHOLD_M5 = 1.6     
TP_PERCENT_M5 = 3.0        
SL_PERCENT_M5 = 3.0        

# --- ตั้งค่าระบบคำนวณ Margin ทบรายวัน ---
BOT_START_DATE = "2026-07-07"  
BASE_MARGIN = 1.00    
DAILY_ADD = 3.00
LEVERAGE = 20            
MAX_TICKETS = 10

# --- GLOBAL VARIABLES FOR ANTI-DOUBLE FIRE (แยกคีย์ตัดสัญญาณซ้ำอิสระ) ---
last_shot_m1_long = 0
last_shot_m1_short = 0
last_shot_m5_long = 0
last_shot_m5_short = 0

class SimpleHTTPServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Whale Hunter V9.5 24/7 Engine is active!")
    def log_message(self, format, *args): return

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPServer)
    server.serve_forever()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: 
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return res.json()
    except Exception as e: 
        print(f"Telegram Send Error: {e}")
        return None

# ฟังก์ชันหาตำแหน่งยอดคลื่น Local Pivot (ถอดลอจิกจาก Pine Script)
def find_pivots(df, length):
    highs = df['high'].values
    lows = df['low'].values
    p_highs = [np.nan] * len(df)
    p_lows = [np.nan] * len(df)
    
    for i in range(length, len(df) - length):
        if highs[i] == max(highs[i - length : i + length + 1]):
            p_highs[i] = highs[i]
        if lows[i] == min(lows[i - length : i + length + 1]):
            p_lows[i] = lows[i]
    return p_highs, p_lows

# ปลดล็อกให้รัน 24/7 ตลอดทั้งวันเสาร์-อาทิตย์
def check_is_weekend():
    return False  # Crypto Never Sleeps

def fetch_trades_safe():
    try:
        trades = exchange.fetch_my_trades(symbol=SYMBOL, limit=30)
        order_markers = []
        for t in trades:
            order_markers.append({
                "timestamp": t['timestamp'],
                "side": t.get('info', {}).get('positionSide', '').upper(),
                "price": float(t['price']),
                "amount": float(t['amount']),
                "trade_side": t.get('side', '').lower()
            })
        df = pd.DataFrame(order_markers)
        if not df.empty:
            df = df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
        return df
    except: return pd.DataFrame()

def count_active_tickets(df_trades):
    l_count, s_count = 0, 0
    try:
        positions = exchange.fetch_positions()
        active_positions = {}
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if 'LAB' in pos_symbol:
                    side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                    active_positions[side] = abs(size)

        if active_positions and not df_trades.empty:
            for side, current_actual_size in active_positions.items():
                target_trade_side = 'buy' if side == 'LONG' else 'sell'
                df_filtered = df_trades[(df_trades['side'] == side) & (df_trades['trade_side'] == target_trade_side)]
                accumulated_size = 0.0
                for _, row in df_filtered.iterrows():
                    if accumulated_size >= current_actual_size: break
                    amt = row['amount']
                    if accumulated_size + amt > current_actual_size: amt = current_actual_size - accumulated_size
                    if amt > 0.0001:
                        if side == 'LONG' and l_count < MAX_TICKETS: l_count += 1
                        elif side == 'SHORT' and s_count < MAX_TICKETS: s_count += 1
                    accumulated_size += row['amount']
    except: pass
    return l_count, s_count

def fire_execution_order(side, entry_price, margin_size, timeframe, tp_pct, sl_pct):
    try:
        contract_amount = round((margin_size * LEVERAGE) / entry_price, 4)
        tp_factor = tp_pct / 100
        sl_factor = sl_pct / 100
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), 4)
            sl_price = round(entry_price * (1 - sl_factor), 4)
            order_side = 'buy'
            emoji = f"🚀 [Double Bottom ({timeframe})]"
        else:
            tp_price = round(entry_price * (1 - tp_factor), 4)
            sl_price = round(entry_price * (1 + sl_factor), 4)
            order_side = 'sell'
            emoji = f"💥 [Double Top ({timeframe})]"

        try: exchange.set_leverage(LEVERAGE, SYMBOL)
        except: pass

        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = f"{emoji} ยิง LABUSDT สำเร็จ!\n• *ฝั่ง:* {side}\n• *ราคาเข้า:* ${entry_price}\n• *Margin:* ${margin_size:.4f} ({LEVERAGE}x)\n🎯 *TP ({tp_pct}%):* ${tp_price} | 🛑 *SL ({sl_pct}%):* ${sl_price}"
        send_telegram_message(tg_msg)
    except Exception as e: 
        print(f"Error executing order ({timeframe}): {e}")
        send_telegram_message(f"⚠️ *[Whale Hunter]* สัญญาณ {timeframe} ยิงพลาดเนื่องจาก: `{str(e)}`")

# ตรรกะแกะรอยหา Pattern จากชุดข้อมูล DataFrame
def analyze_pattern_signals(df, p_length, threshold):
    df = df.copy()
    df['p_high'], df['p_low'] = find_pivots(df, p_length)
    
    idx = len(df) - 1 - p_length
    candle_ts = df.iloc[idx]['timestamp']
    readable_time = pd.to_datetime(candle_ts, unit='ms').strftime('%Y-%m-%d %H:%M')
    
    double_bottom = False
    double_top = False
    
    # ดักสัญญาณขาขึ้น (Double Bottom)
    if not pd.isna(df.iloc[idx]['p_low']):
        pl1 = df.iloc[idx]['p_low']
        df_past_lows = df.iloc[:idx].dropna(subset=['p_low'])
        if not df_past_lows.empty:
            pl2 = df_past_lows.iloc[-1]['p_low']
            if (abs(pl1 - pl2) / pl2 * 100) <= threshold:
                double_bottom = True

    # ดักสัญญาณขาลง (Double Top)
    if not pd.isna(df.iloc[idx]['p_high']):
        ph1 = df.iloc[idx]['p_high']
        df_past_highs = df.iloc[:idx].dropna(subset=['p_high'])
        if not df_past_highs.empty:
            ph2 = df_past_highs.iloc[-1]['p_high']
            if (abs(ph1 - ph2) / ph2 * 100) <= threshold:
                double_top = True
                
    signal = "HOLD"
    if double_bottom: signal = "LONG"
    elif double_top: signal = "SHORT"
    
    return signal, candle_ts, readable_time

# --- MAIN LOOP ENGINE ---
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    print("🟢 Whale Hunter V9.5 Dual-Engine (LABUSDT 24/7) Active...")
    send_telegram_message("🟢 *[Whale Hunter V9.5]* ระบบ 24/7 คริปโตคู่ LABUSDT เฝ้าควบ 1m และ 5m ทำงานเต็มระบบแล้วครับ!")

    last_heartbeat_time = 0.0 

    while True:
        try:
            # คำนวณขยาย Margin ทบรายวันตามเวลาปัจจุบัน
            start_dt = datetime.strptime(BOT_START_DATE, "%Y-%m-%d")
            now_dt = datetime.now()
            days_passed = (now_dt - start_dt).days
            if days_passed < 0: days_passed = 0
            current_mgn_active = BASE_MARGIN + (days_passed * DAILY_ADD)

            # ดึงสถานะออเดอร์ในพอร์ต
            df_trades = fetch_trades_safe()
            active_l, active_s = count_active_tickets(df_trades)
            
            # --------------------------------==================
            # 🔎 ENGINE PART 1: แกะรอยสัญญาณ ไทม์เฟรม 1m
            # --------------------------------==================
            bars_m1 = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_M1, limit=400)
            df_m1 = pd.DataFrame(bars_m1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            sig_m1, ts_m1, time_m1 = analyze_pattern_signals(df_m1, P_LENGTH_M1, SRC_THRESHOLD_M1)
            live_price = df_m1.iloc[-1]['close'] 
            
            if sig_m1 == "LONG" and active_l < MAX_TICKETS:
                if ts_m1 != last_shot_m1_long:
                    fire_execution_order("LONG", live_price, current_mgn_active, "1m", TP_PERCENT_M1, SL_PERCENT_M1)
                    last_shot_m1_long = ts_m1
                    active_l += 1 
            elif sig_m1 == "SHORT" and active_s < MAX_TICKETS:
                if ts_m1 != last_shot_m1_short:
                    fire_execution_order("SHORT", live_price, current_mgn_active, "1m", TP_PERCENT_M1, SL_PERCENT_M1)
                    last_shot_m1_short = ts_m1
                    active_s += 1

            # --------------------------------==================
            # 🔎 ENGINE PART 2: แกะรอยสัญญาณ ไทม์เฟรม 5m
            # --------------------------------==================
            bars_m5 = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME_M5, limit=400)
            df_m5 = pd.DataFrame(bars_m5, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            sig_m5, ts_m5, time_m5 = analyze_pattern_signals(df_m5, P_LENGTH_M5, SRC_THRESHOLD_M5)
            
            if sig_m5 == "LONG" and active_l < MAX_TICKETS:
                if ts_m5 != last_shot_m5_long:
                    fire_execution_order("LONG", live_price, current_mgn_active, "5m", TP_PERCENT_M5, SL_PERCENT_M5)
                    last_shot_m5_long = ts_m5
                    active_l += 1
            elif sig_m5 == "SHORT" and active_s < MAX_TICKETS:
                if ts_m5 != last_shot_m5_short:
                    fire_execution_order("SHORT", live_price, current_mgn_active, "5m", TP_PERCENT_M5, SL_PERCENT_M5)
                    last_shot_m5_short = ts_m5
                    active_s += 1

            # --- Telegram Heartbeat รายงานสถานะบอททุกต้นชั่วโมง ---
            current_time = time.time()
            if last_heartbeat_time == 0.0 or (current_time - last_heartbeat_time >= 3600):
                try:
                    bal = exchange.fetch_balance()
                    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
                except: total_cap = 0.0
                
                heartbeat_msg = (
                    f"🤖 *[Whale Hunter V9.5 - รายงาน LABUSDT]*\n"
                    f"• *โหมดการทำงาน:* 🟢 ออนไลน์ตลอด 24 ชั่วโมง (คริปโต)\n"
                    f"• *ราคาปัจจุบัน:* ${live_price}\n"
                    f"• *ตั๋วค้างในระบบ:* LONG [{active_l}/{MAX_TICKETS}] | SHORT [{active_s}/{MAX_TICKETS}]\n"
                    f"• *Margin ไม้ถัดไป:* ${current_mgn_active:.4f}\n"
                    f"• *ทุนสุทธิในพอร์ต:* ${total_cap} USDT"
                )
                send_telegram_message(heartbeat_msg)
                last_heartbeat_time = current_time

            print(f"🔄 [1m Engine] Sig: {sig_m1} @ {time_m1} | [5m Engine] Sig: {sig_m5} @ {time_m5} | Price: {live_price}")

        except Exception as ex:
            print(f"⚠️ Dual-Engine Loop Error: {ex}")
            
        # ลูปตรวจเช็คทุก ๆ 10 วินาที เพื่อความแม่นยำสูงสุดในไทม์เฟรมย่อย
        time.sleep(10)
