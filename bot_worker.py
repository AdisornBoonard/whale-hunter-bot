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

SYMBOL = 'NCCOGOLD2USD/USDT:USDT'
TIMEFRAME = '1m'
MFI_LENGTH = 14
VOL_MULTIPLIER = 0.7

# --- ตั้งค่าระบบคำนวณ Margin ทบรายวัน ---
BOT_START_DATE = "2026-06-31"  
BASE_MARGIN = 1.00    
DAILY_ADD = 3.00
LEVERAGE = 250
MAX_TICKETS = 10
TP_PERCENT = 0.50
SL_PERCENT = 0.30

USE_EMA = True
EMA_LENGTH = 10
EMA_REVERSE_DIST = 1.5

USE_CCI = True
CCI_LENGTH = 100
CCI_OB = 40.0
CCI_OS = -150.0

# --- GLOBAL VARIABLES FOR ANTI-DOUBLE FIRE ---
last_shot_timestamp_long = 0
last_shot_timestamp_short = 0

class SimpleHTTPServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Whale Hunter V8.9.1 Gold Edition is active and fixed!")
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

# 🔥 FIX สูตรคำนวณ MFI (จำลอง ta.rma ของ TradingView ให้ถูกต้องและไม่แคราช)
def calculate_mfi_pinescript(df, length=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.copy()
    negative_flow = money_flow.copy()
    
    price_change = typical_price.diff()
    positive_flow[price_change <= 0] = 0.0
    negative_flow[price_change >= 0] = 0.0
    
    # คำนวณแบบ RMA (ดัดแปลงจากสูตร Alpha ของ Pine Script)
    alpha = 1.0 / length
    pos_mfi_rma = positive_flow.ewm(alpha=alpha, adjust=False).mean()
    neg_mfi_rma = negative_flow.abs().ewm(alpha=alpha, adjust=False).mean()
    
    # ป้องกันอาการหารด้วยศูนย์ (Division by Zero) ด้วยการบวกค่าเล็กๆ (1e-10) ดักไว้
    mfi_val = 100.0 - (100.0 / (1.0 + (pos_mfi_rma / (neg_mfi_rma + 1e-10))))
    return mfi_val

def calculate_cci(df, length=100):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=length).mean()
    mad = typical_price.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical_price - sma) / (0.015 * (mad + 1e-10))

def check_is_weekend():
    now = datetime.now()
    day = now.weekday() # 5 = เสาร์, 6 = อาทิตย์, 0 = จันทร์
    hour = now.hour
    
    if day == 5: 
        return True
    if day == 6: 
        return True
    if day == 0 and hour < 5: 
        return True
    return False

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
                if 'GOLD' in pos_symbol or 'NCCO' in pos_symbol:
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

def fire_execution_order(side, entry_price, margin_size):
    try:
        contract_amount = round((margin_size * LEVERAGE) / entry_price, 4)
        tp_factor = TP_PERCENT / 100
        sl_factor = SL_PERCENT / 100
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), 2)
            sl_price = round(entry_price * (1 - sl_factor), 2)
            order_side = 'buy'
            emoji = "🚀"
        else:
            tp_price = round(entry_price * (1 - tp_factor), 2)
            sl_price = round(entry_price * (1 + sl_factor), 2)
            order_side = 'sell'
            emoji = "💥"

        try: exchange.set_leverage(LEVERAGE, SYMBOL)
        except: pass

        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = f"{emoji} *[Whale Hunter V8.9.1]* ยิงออโต้สำเร็จ!\n• *ฝั่ง:* {side}\n• *ราคาเข้า:* ${entry_price}\n• *Margin:* ${margin_size:.4f}\n🎯 *TP:* ${tp_price} | 🛑 *SL:* ${sl_price}"
        send_telegram_message(tg_msg)
    except Exception as e: 
        print(f"Error executing order: {e}")
        send_telegram_message(f"⚠️ *[Whale Hunter]* ยิงพลาดเนื่องจาก: `{str(e)}`")

# --- MAIN LOOP ENGINE ---
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    print("🟢 Whale Hunter V8.9.1 Fixed Engine Active...")
    # บังคับส่งสัญญาณแจ้งเปิดบอททันทีเพื่อยืนยันว่า Token Telegram ถูกต้องและทำงานได้จริง
    send_telegram_message("🟢 *[Whale Hunter V8.9.1]* บอทเวอร์ชันแก้ไขจุดบกพร่องสูตรคำนวณ เริ่มทำงานออนไลน์แล้วคราบบบ!")

    # ตั้งค่าเริ่มต้นให้รายงานตัวทันทีในลูปแรก (ไม่ต้องรอนาน)
    last_heartbeat_time = 0.0 

    while True:
        try:
            if check_is_weekend():
                print("⏳ [WEEKEND FILTER] ตลาดทองคำปิดทำการช่วงวันหยุด บอทระงับการทำงานชั่วคราว...")
                time.sleep(60)
                continue

            start_dt = datetime.strptime(BOT_START_DATE, "%Y-%m-%d")
            now_dt = datetime.now()
            days_passed = (now_dt - start_dt).days
            if days_passed < 0: days_passed = 0
            current_mgn_active = BASE_MARGIN + (days_passed * DAILY_ADD)

            # ใช้ดึงข้อมูลย้อนหลัง 300 แท่งเพื่อให้ฐานคำนวณ EMA 10 อิ่มตัวและนิ่งเท่ากราฟจริง
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=300)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['mfi'] = calculate_mfi_pinescript(df, length=MFI_LENGTH)
            df['ema'] = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean() 
            df['v_ma'] = df['volume'].rolling(window=20).mean()
            df['cci'] = calculate_cci(df, length=CCI_LENGTH)
            
            idx = len(df) - 2
            candle_timestamp = df.iloc[idx]['timestamp'] 
            readable_candle_time = pd.to_datetime(candle_timestamp, unit='ms').strftime('%Y-%m-%d %H:%M')
            
            c_close, c_high, c_low = df.iloc[idx]['close'], df.iloc[idx]['high'], df.iloc[idx]['low']
            c_vol, c_mfi, c_vma = df.iloc[idx]['volume'], df.iloc[idx]['mfi'], df.iloc[idx]['v_ma']
            c_ema, c_cci = df.iloc[idx]['ema'], df.iloc[idx]['cci']
            
            p_high, p_low, p_mfi = df.iloc[idx-1]['high'], df.iloc[idx-1]['low'], df.iloc[idx-1]['mfi']
            
            bull_div = (c_low < p_low) and (c_mfi > p_mfi) and (c_mfi < 40)
            bear_div = (c_high > p_high) and (c_mfi < p_mfi) and (c_mfi > 60)
            is_w = c_vol > (c_vma * VOL_MULTIPLIER)
            
            long_base = bull_div and is_w
            short_base = bear_div and is_w
            
            ema_bull = c_close > c_ema
            ema_bear = c_close < c_ema
            ema_distance = abs(c_close - c_ema) / c_ema * 100
            far_from_ema = ema_distance >= EMA_REVERSE_DIST
            
            l_sig, s_sig = False, False
            
            if not USE_EMA:
                l_sig = long_base
                s_sig = short_base
            else:
                if long_base:
                    if far_from_ema:
                        if ema_bull: s_sig = True
                        else: l_sig = True
                    else:
                        if ema_bull: l_sig = True
                        else: s_sig = True

                if short_base:
                    if far_from_ema:
                        if ema_bull: s_sig = True
                        else: l_sig = True
                    else:
                        if ema_bear: s_sig = True 
                        else: l_sig = True

            if USE_CCI and (l_sig or s_sig):
                if c_cci > CCI_OB:
                    s_sig = True
                    l_sig = False
                elif c_cci < CCI_OS:
                    l_sig = True
                    s_sig = False

            signal = "HOLD"
            if l_sig: signal = "LONG"
            elif s_sig: signal = "SHORT"
                        
            live_price = df.iloc[-1]['close']
            df_trades = fetch_trades_safe()
            active_l, active_s = count_active_tickets(df_trades)

            if signal == "LONG" and active_l < MAX_TICKETS:
                if candle_timestamp != last_shot_timestamp_long:
                    fire_execution_order("LONG", live_price, current_mgn_active)
                    last_shot_timestamp_long = candle_timestamp  
                else:
                    print(f"⏳ [ANTI-DOUBLE LONG] สัญญาณซ้ำในแท่งเดิม ({readable_candle_time})")

            elif signal == "SHORT" and active_s < MAX_TICKETS:
                if candle_timestamp != last_shot_timestamp_short:
                    fire_execution_order("SHORT", live_price, current_mgn_active)
                    last_shot_timestamp_short = candle_timestamp 
                else:
                    print(f"⏳ [ANTI-DOUBLE SHORT] สัญญาณซ้ำในแท่งเดิม ({readable_candle_time})")

            # --- Heartbeat ประจำชั่วโมง (ส่งทันทีในรอบแรก) ---
            current_time = time.time()
            if last_heartbeat_time == 0.0 or (current_time - last_heartbeat_time >= 3600):
                try:
                    bal = exchange.fetch_balance()
                    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
                    avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
                except: total_cap, avail_cap = 0.0, 0.0
                
                heartbeat_msg = (
                    f"🤖 *[Whale Hunter V8.9.1 - รายงานตัว]*\n"
                    f"• *สถานะบอท:* 🟢 ออนไลน์ปกติ (Fixed Engine)\n"
                    f"• *ราคาปัจจุบัน:* ${live_price}\n"
                    f"• *ตั๋วค้าง:* LONG [{active_l}/{MAX_TICKETS}] | SHORT [{active_s}/{MAX_TICKETS}]\n"
                    f"• *Margin ไม้ถัดไป:* ${current_mgn_active:.4f}\n"
                    f"• *ทุนสุทธิในพอร์ต:* ${total_cap} USDT"
                )
                send_telegram_message(heartbeat_msg)
                last_heartbeat_time = current_time

            print(f"🔄 Loop Check Finished. Signal: {signal} | Price: {live_price} | Candle: {readable_candle_time}")

        except Exception as ex:
            print(f"⚠️ Loop Error Detected: {ex}")
            
        time.sleep(10)
