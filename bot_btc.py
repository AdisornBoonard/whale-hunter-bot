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

SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '5m'
MFI_LENGTH = 14
VOL_MULTIPLIER = 0.7

# --- ตั้งค่าระบบคำนวณ Margin ทบรายวัน (ตรงตามหน้าจอคอนฟิก) ---
BOT_START_DATE = "2026-07-01"   # รูปแบบ ปปปป-ดด-วว
BASE_MARGIN = 0.50               # Margin ไม้แรก $1 (ปรับตามภาพของพี่)
DAILY_ADD = 1.50                 # เพิ่ม Margin วันละ $3 (ปรับตามภาพของพี่)
LEVERAGE = 150                  # Leverage 150x
MAX_TICKETS = 3                 # เปิดสูงสุดต่อฝั่ง 3 ไม้
TP_PERCENT = 2.0                # TP 2% (ปรับตามภาพของพี่)
SL_PERCENT = 2.5                # SL 2.5% (ปรับตามภาพของพี่)

# --- GLOBAL VARIABLES FOR ANTI-DOUBLE FIRE ---
last_shot_timestamp_long = 0
last_shot_timestamp_short = 0

def init_exchange_settings():
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"⚙️ Set Leverage to {LEVERAGE}x for {SYMBOL} successfully.")
    except Exception as e:
        print(f"⚠️ Warning setting leverage: {e} (ระบบจะรันต่อด้วยค่าตั้งต้นบนเว็บ)")

class SimpleHTTPServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"BTC Whale Hunter V8.9 (Pure Divergence Sync) is running online!")
    def log_message(self, format, *args): return

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPServer)
    server.serve_forever()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
    except: pass

def calculate_mfi(df, length=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.copy()
    negative_flow = money_flow.copy()
    price_change = typical_price.diff()
    positive_flow[price_change <= 0] = 0
    negative_flow[price_change >= 0] = 0
    pos_mf = positive_flow.rolling(window=length).sum()
    neg_mf = negative_flow.rolling(window=length).sum()
    return 100 - (100 / (1 + (pos_mf / neg_mf.abs())))

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
            if size != 0 and pos.get('symbol', '').upper() == SYMBOL:
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

def fire_execution_order(side, estimated_price, margin_size):
    try:
        # 1. คำนวณขนาดสัญญาเบื้องต้นสำหรับการยิง Market Order
        contract_amount = round((margin_size * LEVERAGE) / estimated_price, 4)
        if contract_amount < 0.0001: contract_amount = 0.0001 
            
        order_side = 'buy' if side == "LONG" else 'sell'
        emoji = "🟠🚀" if side == "LONG" else "🟠💥"

        # 2. ส่งคำสั่งหลัก Market Order และรับข้อมูลราคาแมตช์จริงเพื่อป้องกัน Slippage
        main_order = exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        
        # ดึงราคาที่แมตช์จริงจากตลาด (ถ้าแลกเปลี่ยนไม่คืนค่า ให้ใช้ราคาประเมินสำรอง)
        entry_price = float(main_order.get('price', 0)) or float(main_order.get('average', 0)) or estimated_price
        
        # 3. คำนวณราคา TP/SL จากราคาต้นทุนจริงบนกระดาน
        tp_factor = TP_PERCENT / 100
        sl_factor = SL_PERCENT / 100
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), 1)
            sl_price = round(entry_price * (1 - sl_factor), 1)
        else:
            tp_price = round(entry_price * (1 - tp_factor), 1)
            sl_price = round(entry_price * (1 + sl_factor), 1)

        try:
            # 4. ส่งคำสั่งตั้งเงื่อนไขพ่วง TP/SL เฝ้าบนกระดานกระชั้นชิด
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = f"{emoji} *[Whale Hunter BTC - Pure Sync]* ยิงสำเร็จ!\n• *ฝั่ง:* {side}\n• *ราคาทุนจริง:* ${entry_price}\n• *Margin:* ${margin_size:.2f}\n🎯 *TP:* ${tp_price} | 🛑 *SL:* ${sl_price}"
        send_telegram_message(tg_msg)
    except Exception as e: 
        print(f"Error BTC order: {e}")
        error_msg = f"⚠️ *[Whale Hunter BTC - ยิงพลาด!]*\n• เกิดข้อผิดพลาดในการส่งคำสั่ง\n• *ฝั่ง:* {side}\n• *สาเหตุ:* `{str(e)}`"
        send_telegram_message(error_msg)

# --- MAIN LOOP ENGINE ---
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    init_exchange_settings()

    print("🟢 BTC Whale Hunter (Pure Mode Sync V8.9) is Active...")
    send_telegram_message("🟠 *[Whale Hunter BTC]* เริ่มระบบเฝ้ากราฟแบบ Pure Divergence (ปิด EMA & CCI ทั้งกระดาน) เรียบร้อยครับพี่!")

    last_heartbeat_time = 0.0

    while True:
        try:
            # คำนวณวันทบต้นทุน Margin รายวัน
            start_dt = datetime.strptime(BOT_START_DATE, "%Y-%m-%d")
            now_dt = datetime.now()
            days_passed = (now_dt - start_dt).days
            if days_passed < 0: days_passed = 0
            
            current_mgn_active = BASE_MARGIN + (days_passed * DAILY_ADD)

            # ดึงข้อมูลแท่งเทียนแท่งล่าสุด
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # คำนวณ Indicator พื้นฐาน
            df['mfi'] = calculate_mfi(df, length=MFI_LENGTH)
            df['v_ma'] = df['volume'].rolling(window=20).mean()
            
            # ดึงข้อมูล "แท่งที่เพิ่งปิดตัวลงสมบูรณ์" (Index -2) มาวิเคราะห์เพื่อไม่ให้สัญญาณเพ้นท์ย้อนหลัง
            idx = len(df) - 2
            candle_timestamp = df.iloc[idx]['timestamp'] 
            readable_candle_time = pd.to_datetime(candle_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M')
            
            c_high, c_low, c_vol = df.iloc[idx]['high'], df.iloc[idx]['low'], df.iloc[idx]['volume']
            c_mfi, c_vma = df.iloc[idx]['mfi'], df.iloc[idx]['v_ma']
            
            # ดึงข้อมูลแท่งก่อนหน้าตัวมันเองอีกหนึ่งชั้น (Index -3) ถอดลอจิกตามสูตร [1] ของ TradingView เป๊ะๆ
            p_high, p_low, p_mfi = df.iloc[idx-1]['high'], df.iloc[idx-1]['low'], df.iloc[idx-1]['mfi']
            
            # ====================================================
            # 📌 PURE LOGIC CONFIGURATION (งดใช้ EMA & CCI ไร้ตัวกรอง)
            # ====================================================
            bull_div = (c_low < p_low) and (c_mfi > p_mfi) and (c_mfi < 40)
            bear_div = (c_high > p_high) and (c_mfi < p_mfi) and (c_mfi > 60)
            is_w = c_vol > (c_vma * VOL_MULTIPLIER)
            
            signal = "HOLD"
            if bull_div and is_w:
                signal = "LONG"
            elif bear_div and is_w:
                signal = "SHORT"
                    
            live_price = df.iloc[-1]['close']

            df_trades = fetch_trades_safe()
            active_l, active_s = count_active_tickets(df_trades)

            # --- ตรวจสอบเงื่อนไขการส่งคำสั่ง + ล็อกเวลาป้องกันยิงเบิ้ลแท่งเดิม ---
            if signal == "LONG" and active_l < MAX_TICKETS:
                if candle_timestamp != last_shot_timestamp_long:
                    fire_execution_order("LONG", live_price, current_mgn_active)
                    last_shot_timestamp_long = candle_timestamp  
                else:
                    print(f"⏳ [ANTI-DOUBLE LONG] สัญญาณซ้ำแท่งเดิม ({readable_candle_time}) บอทล็อกข้ามให้ครับพี่")

            elif signal == "SHORT" and active_s < MAX_TICKETS:
                if candle_timestamp != last_shot_timestamp_short:
                    fire_execution_order("SHORT", live_price, current_mgn_active)
                    last_shot_timestamp_short = candle_timestamp 
                else:
                    print(f"⏳ [ANTI-DOUBLE SHORT] สัญญาณซ้ำแท่งเดิม ({readable_candle_time}) บอทล็อกข้ามให้ครับพี่")

            # --- Heartbeat ทุกๆ 1 ชม. ---
            current_time = time.time()
            if current_time - last_heartbeat_time >= 3600:
                try:
                    bal = exchange.fetch_balance()
                    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
                    avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
                except: total_cap, avail_cap = 0.0, 0.0
                
                heartbeat_msg = (
                    f"🤖 *[Whale Hunter BTC - รายงานตัว]*\n"
                    f"• *สถานะบอท:* 🟢 ออนไลน์โหมด Pure (TF 5m)\n"
                    f"• *วันที่คำนวณบอท:* รันมาแล้ว {days_passed} วัน\n"
                    f"• *ราคาปัจจุบัน:* ${live_price}\n"
                    f"• *ตั๋วค้าง:* LONG [{active_l}/{MAX_TICKETS}] | SHORT [{active_s}/{MAX_TICKETS}]\n"
                    f"• *ทุนคงเหลือ:* ${avail_cap} / สุทธิ ${total_cap}\n"
                    f"• *Margin ไม้ถัดไป:* ${current_mgn_active:.2f}"
                )
                send_telegram_message(heartbeat_msg)
                last_heartbeat_time = current_time

            print(f"🔄 BTC Loop Finished. Signal: {signal} | Price: {live_price} | Candle Time: {readable_candle_time} | Day Count: {days_passed}")

        except Exception as ex:
            print(f"Loop Error: {ex}")
            
        # ปรับความเร็วลูปให้เหมาะสม: บอทจะหลับคราวละ 10 วินาที เพื่อรีบตื่นขึ้นมาตรวจเช็คต้นแท่งใหม่อย่างรวดเร็ว ไม่พลาดวินาทีแรกที่สัญญาณเกิด
        time.sleep(10)
