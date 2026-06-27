import os
import time
import math
import ccxt
import requests
import numpy as np  
import pandas as pd
from dotenv import load_dotenv

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

# --- ตั้งค่ากลยุทธ์ตามโค้ดล่าสุดของพี่ ---
BASE_MARGIN = 0.02  
DAILY_ADD = 0.06
LEVERAGE = 250
MAX_TICKETS = 10
TP_PERCENT = 0.50
SL_PERCENT = 0.30

# ตั้งค่าอินดิเคเตอร์ฟิลเตอร์
USE_EMA = True
EMA_LENGTH = 10
EMA_REVERSE_DIST = 1.5

USE_CCI = True
CCI_LENGTH = 100
CCI_OB = 40.0
CCI_OS = -150.0

# ตัวแปรควบคุมระบบภายในคอร์
bot_start_time = time.time()
last_order_time = 0
last_heartbeat_time = 0

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

def calculate_cci(df, length=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=length).mean()
    mad = typical_price.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical_price - sma) / (0.015 * mad)

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
    except:
        return pd.DataFrame()

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

        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = f"{emoji} *[Whale Hunter V8.9 - Worker]* ยิงออโต้สำเร็จ!\n• *ฝั่ง:* {side}\n• *ราคาเข้า:* ${entry_price}\n• *Margin:* ${margin_size:.4f}\n🎯 *TP:* ${tp_price} | 🛑 *SL:* ${sl_price}"
        send_telegram_message(tg_msg)
    except Exception as e:
        print(f"Error executing order: {e}")

# --- START ENGINE LOOP ---
print("🟢 Whale Hunter V8.9 Background Worker is Active...")
send_telegram_message("🟢 *[Whale Hunter V8.9]* บอทหลักเริ่มทำงานเงียบหลังบ้านรัน 24 ชั่วโมงเรียบร้อย!")

while True:
    try:
        # 1. คำนวณขยับมาร์จิ้นรายวัน
        days_passed = math.floor((time.time() - bot_start_time) / (24 * 60 * 60))
        current_mgn_active = BASE_MARGIN + (days_passed * DAILY_ADD)

        # 2. ดึงราคากราฟคำนวณสัญญาณตามสูตรล่าสุดของพี่
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['mfi'] = calculate_mfi(df, length=MFI_LENGTH)
        df['ema'] = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean() 
        df['v_ma'] = df['volume'].rolling(window=20).mean()
        df['cci'] = calculate_cci(df, length=CCI_LENGTH)
        
        idx = len(df) - 2
        c_close = df.iloc[idx]['close']
        c_high  = df.iloc[idx]['high']
        c_low   = df.iloc[idx]['low']
        c_vol   = df.iloc[idx]['volume']
        c_mfi   = df.iloc[idx]['mfi']
        c_vma   = df.iloc[idx]['v_ma']
        c_ema   = df.iloc[idx]['ema']
        c_cci   = df.iloc[idx]['cci']
        
        p_high  = df.iloc[idx-1]['high']
        p_low   = df.iloc[idx-1]['low']
        p_mfi   = df.iloc[idx-1]['mfi']
        
        bull_div = (c_low < p_low) and (c_mfi > p_mfi) and (c_mfi < 40)
        bear_div = (c_high > p_high) and (c_mfi < p_mfi) and (c_mfi > 60)
        is_w = c_vol > (c_vma * VOL_MULTIPLIER)
        
        long_base  = bull_div and is_w
        short_base = bear_div and is_w
        
        ema_bull = c_close > c_ema
        ema_bear = c_close < c_ema
        ema_distance = abs(c_close - c_ema) / c_ema * 100
        far_from_ema = ema_distance >= EMA_REVERSE_DIST
        
        signal = "HOLD"
        if not USE_EMA:
            if long_base: signal = "LONG"
            if short_base: signal = "SHORT"
        else:
            if long_base:
                signal = "SHORT" if (far_from_ema and ema_bull) else "LONG"
            if short_base:
                signal = "SHORT" if not (far_from_ema and not ema_bear) else "LONG"
                    
        if USE_CCI and signal in ["LONG", "SHORT"]:
            if c_cci > CCI_OB: signal = "SHORT"
            elif c_cci < CCI_OS: signal = "LONG"
                
        live_price = df.iloc[-1]['close']

        # 3. ตรวจสอบจำนวนไม้ย่อยค้างกระดาน
        df_trades = fetch_trades_safe()
        active_l, active_s = count_active_tickets(df_trades)

        # 4. ลอจิกส่งออเดอร์ออโต้ (Cooldown 60 วิ)
        current_time_sec = time.time()
        if signal in ["LONG", "SHORT"] and (current_time_sec - last_order_time) > 60:
            if signal == "LONG" and active_l < MAX_TICKETS:
                fire_execution_order("LONG", live_price, current_mgn_active)
                last_order_time = current_time_sec
            elif signal == "SHORT" and active_s < MAX_TICKETS:
                fire_execution_order("SHORT", live_price, current_mgn_active)
                last_order_time = current_time_sec

        # 5. สแตมป์เวลาส่งรายงานตัวประจำชั่วโมง (3600 วิ)
        if (current_time_sec - last_heartbeat_time) >= 3600:
            try:
                bal = exchange.fetch_balance()
                total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
                avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
            except: total_cap, avail_cap = 0.0, 0.0
            
            heartbeat_msg = (
                f"🤖 *[Whale Hunter V8.9 - รายงานตัวประจำชั่วโมง]*\n"
                f"• *สถานะบอท:* 🟢 บอทรันปกติ (รันเงียบคลาวด์)\n"
                f"• *ราคาปัจจุบัน:* ${live_price}\n"
                f"• *จำนวนไม้ค้าง:* LONG [{active_l}/{MAX_TICKETS}] | SHORT [{active_s}/{MAX_TICKETS}]\n"
                f"• *ทุนคงเหลือ:* ${avail_cap} / สุทธิ ${total_cap}\n"
                f"• *Margin ปัจจุบัน:* ${current_mgn_active:.4f}\n"
                f"📟 _บอทยังทำงานอยู่ดี ระบบคลาวด์เปิดต่อเนื่องครับ_"
            )
            send_telegram_message(heartbeat_msg)
            last_heartbeat_time = current_time_sec

    except Exception as ex:
        print(f"Error in running engine loop: {ex}")
        
    time.sleep(10)
