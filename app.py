import os
import time
import ccxt
import requests
import pandas as pd
import plotly.graph_objects as px
import streamlit as st
from dotenv import load_dotenv

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Whale Hunter V7.3 - Cloud Steady", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        body, .main, .block-container { background-color: #0b0e14 !important; color: white !important; }
        div[data-testid="stMetricValue"] { color: #00e676 !important; font-family: monospace; font-size: 24px; }
        .stTable { background-color: #12161f !important; border: 1px solid #1e2533 !important; border-radius: 6px; }
        h3 { color: #90a4ae !important; font-size: 14px !important; text-transform: uppercase; letter-spacing: 0.5px; }
        .stCaption { font-family: monospace; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

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
exchange.set_sandbox_mode(False) 

SYMBOL = 'NCCOGOLD2USD/USDT:USDT'
TIMEFRAME = '1m'
MFI_LENGTH = 14
LOOKBACK = 100
VOL_MULTIPLIER = 0.7
EMA_LENGTH = 10
USE_EMA = True

default_margin = 0.06   
default_lev = 250       
default_max_t = 5       
default_tp = 0.50       
default_sl = 0.30       

if 'bot_active' not in st.session_state:
    st.session_state.bot_active = True

# --- 2. HELPERS FUNCTIONS ---
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

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

# --- 3. CORE TRADING ENGINE ---
def get_balance():
    try:
        # ดึงเงินในพอร์ตฟิวเจอร์สจริงจาก BingX มาโชว์
        bal = exchange.fetch_balance()
        usdt_bal = bal.get('USDT', {})
        total_balance = float(usdt_bal.get('total', 0.0))
        available_balance = float(usdt_bal.get('free', 0.0))
        return round(total_balance, 2), round(available_balance, 2)
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return 0.0, 0.0

def get_market_and_signal():
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=150)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['mfi'] = calculate_mfi(df, length=MFI_LENGTH)
        df['ema'] = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean()
        df['v_ma'] = df['volume'].rolling(window=20).mean()
        
        idx = len(df) - 2
        c_close, c_high, c_low, c_vol, c_mfi, c_vma, c_ema = (
            df.iloc[idx]['close'], df.iloc[idx]['high'], df.iloc[idx]['low'],
            df.iloc[idx]['volume'], df.iloc[idx]['mfi'], df.iloc[idx]['v_ma'], df.iloc[idx]['ema']
        )
        
        prev_low = df.iloc[idx-LOOKBACK:idx]['low'].min()
        prev_high = df.iloc[idx-LOOKBACK:idx]['high'].max()
        prev_mfi_min = df.iloc[idx-LOOKBACK:idx]['mfi'].min()
        prev_mfi_max = df.iloc[idx-LOOKBACK:idx]['mfi'].max()
        
        bull_div = (c_low < prev_low) and (c_mfi > prev_mfi_min) and (c_mfi < 40)
        bear_div = (c_high > prev_high) and (c_mfi < prev_mfi_max) and (c_mfi > 60)
        is_whale_vol = c_vol > (c_vma * VOL_MULTIPLIER)
        
        long_base = bull_div and is_whale_vol
        short_base = bear_div and is_whale_vol
        
        l_sig, s_sig = False, False
        if not USE_EMA:
            l_sig, s_sig = long_base, short_base
        else:
            if long_base:
                if c_close > c_ema: l_sig = True
                else: s_sig = True
            if short_base:
                if c_close < c_ema: s_sig = True
                else: l_sig = True
                
        live_price = df.iloc[-1]['close']
        bar_time = df.iloc[-1]['timestamp']
        
        if l_sig: return "LONG", live_price, bar_time, df
        if s_sig: return "SHORT", live_price, bar_time, df
        return "HOLD", live_price, bar_time, df
    except:
        return "ERROR", 0, None, pd.DataFrame()

def check_account_positions(live_price, base_margin, leverage):
    active_positions = []
    l_count, s_count = 0, 0
    try:
        # ดึงพอร์ตจริงหลังบ้านจาก BingX โดยตรง คลาวด์รีเฟรชยังไงข้อมูลก็ไม่มีวันหาย!
        positions = exchange.fetch_positions() 
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if 'GOLD' in pos_symbol or 'NCCO' in pos_symbol:
                    side = pos.get('side', '').upper()
                    
                    # 💡 คัดแยกจำนวนตั๋วจำลองด้วยสูตรคณิตศาสตร์: ขนาดสัญญารวม / ขนาดสัญญาต่อ 1 ไม้
                    one_ticket_size = round((base_margin * leverage) / live_price, 4) if live_price > 0 else 0.0001
                    calculated_tickets = max(1, round(abs(size) / one_ticket_size))
                    
                    if side == 'LONG' or size > 0: 
                        l_count = calculated_tickets
                        side = 'LONG'
                    else: 
                        s_count = calculated_tickets
                        side = 'SHORT'
                    
                    active_positions.append({
                        "ID": pos.get('id', 'N/A'), 
                        "Side": side, 
                        "Total Size (Amt)": abs(size),
                        "Est. ไม้ที่เปิด": calculated_tickets,
                        "Entry Price": pos.get('entryPrice'),
                        "Margin ($)": round(float(pos.get('initialMargin', 0)), 4),
                        "Leverage": f"{pos.get('leverage')}x",
                        "P/L ($)": round(float(pos.get('unrealizedPnl', 0)), 4),
                        "P/L (%)": f"{round(float(pos.get('percentage', 0)), 2)}%"
                    })
    except Exception as e:
        print(f"Error fetching positions: {e}")
    return active_positions, l_count, s_count

def fire_execution_order(side, entry_price, margin_size, leverage, tp_p, sl_p, is_manual=False):
    try:
        side = side.upper()
        contract_amount = round((margin_size * leverage) / entry_price, 4)
        mode_text = "Manual (กดมือ)" if is_manual else "Auto (บอทยิง)"
        emoji_side = "🚀" if side == "LONG" else "💥"
        
        tp_percent = tp_p / 100
        sl_percent = sl_p / 100
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_percent), 2)
            sl_price = round(entry_price * (1 - sl_percent), 2)
            order_side, pos_side = 'buy', 'LONG'
        elif side == "SHORT":
            tp_price = round(entry_price * (1 - tp_percent), 2)
            sl_price = round(entry_price * (1 + sl_percent), 2)
            order_side, pos_side = 'sell', 'SHORT'
        else:
            return f"❌ ไม่รู้จักฝั่ง {side}"

        main_params = {'positionSide': pos_side}
        order = exchange.create_order(
            symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params=main_params
        )
        
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(
                symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount,
                params={'positionSide': pos_side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'}
            )
            exchange.create_order(
                symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount,
                params={'positionSide': pos_side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'}
            )
        except Exception as tp_sl_err:
            print(f"⚠️ ตั้ง TP/SL พ่วงไม่สำเร็จ: {tp_sl_err}")

        tg_msg = (
            f"{emoji_side} *Whale Hunter สั่งเปิดไม้เงินจริงสำเร็จ!*\n"
            f"• *โหมดสั่งการ:* {mode_text}\n"
            f"• *ฝั่ง:* {pos_side}\n"
            f"• *ราคาเข้าซื้อ:* ${entry_price}\n"
            f"🎯 *เป้า (TP):* ${tp_price} | 🛑 *คัต (SL):* ${sl_price}"
        )
        send_telegram_message(tg_msg)
        return f"เปิด {pos_side} สำเร็จ ({mode_text})"
    except Exception as e:
        return f"❌ สั่งซื้อล้มเหลว: {e}"

def close_all_positions():
    try:
        positions = exchange.fetch_positions() 
        closed_count = 0
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if 'GOLD' in pos_symbol or 'NCCO' in pos_symbol:
                    side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                    close_side = 'sell' if side == 'LONG' else 'buy'
                    exchange.create_order(
                        symbol=SYMBOL, type='market', side=close_side, amount=abs(size), params={'positionSide': side}
                    )
                    closed_count += 1
        return f"✅ เคลียร์พอร์ตสำเร็จ (ปิดไป {closed_count} ไม้)" if closed_count > 0 else "ℹ️ ไม่มีออเดอร์ค้างให้ปิด"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

# --- 4. STREAMLIT FRONT-END ---
bot_status_indicator = "● LIVE AUTOTRADING ACTIVE" if st.session_state.bot_active else "○ AUTOTRADING DISABLED"
bot_status_color = "#00e676" if st.session_state.bot_active else "#ff1744"

st.markdown(f"""
<div style="background-color: #12161f; border: 1px solid #1e2533; padding: 10px 20px; border-radius: 6px; margin-bottom: 20px;">
    <span style="font-size: 20px; font-weight: bold; color: white;">WHALE HUNTER V7.3 - CLOUD RUNNER</span>
    <span style="float: right; color: {bot_status_color}; font-weight: bold; padding-top: 5px;">{bot_status_indicator}</span>
</div>
""", unsafe_allow_html=True)

signal, live_price, bar_time, df_market = get_market_and_signal()
total_capital, available_capital = get_balance() # ดึงยอดเงินพอร์ตจริง
positions_list, active_l, active_s = check_account_positions(live_price, default_margin, default_lev)

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.metric("เงินทุนทั้งหมดในพอร์ต", f"${total_capital}")
        base_mgn = st.number_input("Margin ไม้แรก ($)", value=default_margin, format="%.2f")  
        lev = st.number_input("Leverage (x)", value=default_lev, min_value=1, max_value=250)
        max_t = st.number_input("เปิดสูงสุด (ต่อฝั่ง)", value=default_max_t)
        
        tp_p = st.slider("TP (%)", 0.1, 5.0, default_tp)
        sl_p = st.slider("SL (%)", 0.1, 5.0, default_sl)
        
        c_status_1, c_status_2 = st.columns(2)
        with c_status_1:
            if st.button("🟢 เปิดรันบอท", use_container_width=True):
                st.session_state.bot_active = True
                send_telegram_message("🟢 *Whale Hunter:* ระบบเปิดรันบอทออโต้แล้ว")
                st.rerun()
        with c_status_2:
            if st.button("🛑 ปิดระบบบอท", use_container_width=True):
                st.session_state.bot_active = False
                send_telegram_message("🛑 *Whale Hunter:* ระบบบอทออโต้ถูกปิดใช้งาน")
                st.rerun()

    st.markdown("<h3>Manual Control (กดมือ)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        c_buy, c_sell = st.columns(2)
        with c_buy:
            if st.button("🚀 OPEN LONG", use_container_width=True):
                res = fire_execution_order("LONG", live_price, base_mgn, lev, tp_p, sl_p, is_manual=True)
                st.rerun()
        with c_sell:
            if st.button("💥 OPEN SHORT", use_container_width=True):
                res = fire_execution_order("SHORT", live_price, base_mgn, lev, tp_p, sl_p, is_manual=True)
                st.rerun()
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🛑 CLOSE ALL POSITIONS (ปิดพอร์ตจริง)", use_container_width=True, type="secondary"):
            res = close_all_positions()
            send_telegram_message(f"🚨 *Whale Hunter:* กดมือสั่งปิดทุกไม้เรียบร้อย: {res}")
            st.rerun()

# --- ระบบออโต้สแกนยิงสัญญาพึ่งพาข้อมูลจริงบนกระดาน ---
if st.session_state.bot_active and signal in ["LONG", "SHORT"]:
    # เช็คประวัติกันเหนียวในความจำชั่วคราว
    if 'last_order_time' not in st.session_state:
        st.session_state.last_order_time = 0
        
    current_time_sec = time.time()
    # ดักให้ยิงได้ 1 ครั้งต่อ 1 แท่งเทียนนาที (ห่างกันอย่างน้อย 50 วินาที)
    if (current_time_sec - st.session_state.last_order_time) > 50:
        if signal == "LONG" and active_l < max_t:
            res = fire_execution_order("LONG", live_price, base_mgn, lev, tp_p, sl_p)
            st.session_state.last_order_time = current_time_sec
            st.rerun()
        elif signal == "SHORT" and active_s < max_t:
            res = fire_execution_order("SHORT", live_price, base_mgn, lev, tp_p, sl_p)
            st.session_state.last_order_time = current_time_sec
            st.rerun()

with col_center:
    st.markdown(f"<h3>{SYMBOL} Real-Time Chart</h3>", unsafe_allow_html=True)
    if not df_market.empty:
        fig = px.Figure()
        fig.add_trace(px.Candlestick(
            x=df_market['datetime'], open=df_market['open'], high=df_market['high'],
            low=df_market['low'], close=df_market['close'], name="ราคา"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#12161f', plot_bgcolor='#12161f',
            margin=dict(l=5, r=5, t=5, b=5), height=250, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3>Positions (ดึงจาก BingX 100% ไม่กลัวคลาวด์ดับ)</h3>", unsafe_allow_html=True)
    if positions_list:
        st.dataframe(pd.DataFrame(positions_list), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ พอร์ตว่างเปล่า (ไม่มีออเดอร์ค้างบนหน้ากระดานเทรด BingX)")

with col_right:
    st.markdown("<h3>Performance Dashboard</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Net Equity", f"${total_capital}")
        m2.metric("เงินที่ใช้ได้", f"${available_capital}")
        m3, m4 = st.columns(2)
        m3.metric("คำนวณจำนวนไม้ (L/S)", f"{active_l} / {active_s}")
        m4.metric("ราคาตลาดปัจจุบัน", f"${live_price}")

    st.markdown("<h3>Signal Log (บันทึกสแกนเรียงลำดับเวลา)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(f"⏱️ อัปเดตล่าสุดเมื่อ: {time.strftime('%H:%M:%S')}")
        st.caption(f"• ระบบวิเคราะห์แท่งเทียนนาทีปัจจุบัน: *{signal}*")
        st.caption(f"• การตั้งค่าบอทออโต้ตอนนี้: {'🟢 เปิดทำงาน' if st.session_state.bot_active else '🛑 ปิดทำงาน'}")
        if signal in ["LONG", "SHORT"]:
            st.markdown(f"<span style='color:#00e676;'>🎯 พบสัญญาณ {signal} ณ ราคา ${live_price}</span>", unsafe_allow_html=True)
        else:
            st.caption("• สถานะตลาด: กำลังสแกนหาจุด Divergence...")

time.sleep(3)
st.rerun()
