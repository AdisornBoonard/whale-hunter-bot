import os
import time
import ccxt
import requests
import pandas as pd
import plotly.graph_objects as px
import streamlit as st
from dotenv import load_dotenv

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Whale Hunter V7.2 - Live Trader", layout="wide", initial_sidebar_state="collapsed")

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
LOOKBACK = 20
VOL_MULTIPLIER = 0.7
EMA_LENGTH = 20
USE_EMA = True

init_cap = 0.6          
default_margin = 0.06   
default_daily = 0.03    
default_lev = 250       
default_max_t = 5       
default_tp = 0.50       
default_sl = 0.30       

# --- 2. INITIALIZE SESSION STATES FOR VIRTUAL ORDERS ---
if 'virtual_orders' not in st.session_state:
    st.session_state.virtual_orders = [] # เก็บตั๋วออเดอร์จำลองแยกไม้
if 'order_counter' not in st.session_state:
    st.session_state.order_counter = 1
if 'logs' not in st.session_state: 
    st.session_state.logs = [f"🟢 บอทเริ่มต้นระบบเงินจริง (มาร์จิ้นไม้แรก ${default_margin})"]
if 'last_processed_bar' not in st.session_state: 
    st.session_state.last_processed_bar = None
if 'bot_active' not in st.session_state:
    st.session_state.bot_active = True

# --- 3. HELPERS FUNCTIONS ---
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

# --- 4. CORE TRADING ENGINE ---
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

def fire_execution_order(side, entry_price, margin_size, leverage, tp_p, sl_p, is_manual=False):
    try:
        side = side.upper()
        contract_amount = round((margin_size * leverage) / entry_price, 4)
        mode_text = "Manual (กดมือ)" if is_manual else "Auto (บอทยิง)"
        emoji_side = "🚀" if side == "LONG" else "💥"
        
        main_params = {'positionSide': side}
        order = exchange.create_order(
            symbol=SYMBOL, type='market', side='buy' if side == 'LONG' else 'sell', amount=contract_amount, params=main_params
        )
        
        # สร้างตั๋วจำลองแยกไม้เก็บไว้ในระบบบอท
        new_order = {
            "Ticket": f"#{st.session_state.order_counter}",
            "Side": side,
            "Amount": contract_amount,
            "Entry": entry_price,
            "Margin": margin_size,
            "Leverage": leverage,
            "Time": time.strftime('%H:%M:%S')
        }
        st.session_state.virtual_orders.append(new_order)
        st.session_state.order_counter += 1

        tg_msg = (
            f"{emoji_side} *Whale Hunter เข้าออเดอร์จำลองแยกไม้แล้ว!*\n"
            f"• *ตั๋วเลขที่:* #{st.session_state.order_counter-1}\n"
            f"• *ฝั่ง:* {side} ({mode_text})\n"
            f"• *ราคาเข้า:* ${entry_price}\n"
            f"• *ขนาดสัญญา:* {contract_amount} สัญญา"
        )
        send_telegram_message(tg_msg)
        return f"เปิด {side} ตั๋ว #{st.session_state.order_counter-1} สำเร็จ"
    except Exception as e:
        return f"❌ สั่งซื้อล้มเหลว: {e}"

def close_specific_virtual_order(ticket_id):
    try:
        # ค้นหาตั๋วจำลองเพื่อดึงขนาดสัญญาที่จะทำการเคลียร์ออก
        for idx, order in enumerate(st.session_state.virtual_orders):
            if order['Ticket'] == ticket_id:
                side = order['Side']
                amount = order['Amount']
                close_side = 'sell' if side == 'LONG' else 'buy'
                
                # ส่งคำสั่ง Partial Close หักขนาดสัญญาไม้นั้นออกจากพอร์ตรวม Cross
                exchange.create_order(
                    symbol=SYMBOL, type='market', side=close_side, amount=amount, params={'positionSide': side}
                )
                
                st.session_state.virtual_orders.pop(idx)
                return f"✅ ปิดตั๋ว {ticket_id} ขนาด {amount} สำเร็จ"
        return "❌ ไม่พบตั๋วออเดอร์นี้ในระบบ"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาดในการสั่งปิดตั๋ว: {e}"

def close_all_positions():
    try:
        # สั่งเครียร์พอร์ตหลังบ้านจริงทั้งหมด
        positions = exchange.fetch_positions() 
        closed_count = 0
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                close_side = 'sell' if side == 'LONG' else 'buy'
                exchange.create_order(
                    symbol=SYMBOL, type='market', side=close_side, amount=abs(size), params={'positionSide': side}
                )
                closed_count += 1
        # ล้างตั๋วในระบบบอททั้งหมดออก
        st.session_state.virtual_orders = []
        if closed_count > 0:
            return f"✅ เคลียร์พอร์ตหลังบ้านและล้างตั๋วจำลองทั้งหมดเรียบร้อย"
        return "ℹ️ ไม่มีออเดอร์ค้างในระบบ"
    except Exception as e:
        return f"❌ ล้มเหลว: {e}"

# --- 5. STREAMLIT FRONT-END ---
bot_status_indicator = "● LIVE AUTOTRADING ACTIVE" if st.session_state.bot_active else "○ AUTOTRADING DISABLED"
bot_status_color = "#00e676" if st.session_state.bot_active else "#ff1744"

st.markdown(f"""
<div style="background-color: #12161f; border: 1px solid #1e2533; padding: 10px 20px; border-radius: 6px; margin-bottom: 20px;">
    <span style="font-size: 20px; font-weight: bold; color: white;">WHALE HUNTER V7.2 - VIRTUAL TRACKING</span>
    <span style="float: right; color: {bot_status_color}; font-weight: bold; padding-top: 5px;">{bot_status_indicator}</span>
</div>
""", unsafe_allow_html=True)

signal, live_price, bar_time, df_market = get_market_and_signal()

# นับจำนวนฝั่งจากตั๋วเสมือน
active_l = sum(1 for o in st.session_state.virtual_orders if o['Side'] == 'LONG')
active_s = sum(1 for o in st.session_state.virtual_orders if o['Side'] == 'SHORT')

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.number_input("ทุนเริ่มต้น ($)", value=init_cap, disabled=True)
        base_mgn = st.number_input("Margin ไม้แรก ($)", value=default_margin, format="%.2f")  
        lev = st.number_input("Leverage (x)", value=default_lev, min_value=1, max_value=250)
        max_t = st.number_input("เปิดสูงสุด (ต่อฝั่ง)", value=default_max_t)
        
        tp_p = st.slider("TP (%)", 0.1, 5.0, default_tp)
        sl_p = st.slider("SL (%)", 0.1, 5.0, default_sl)
        
        c_status_1, c_status_2 = st.columns(2)
        with c_status_1:
            if st.button("🟢 เปิดรันบอท", use_container_width=True, disabled=st.session_state.bot_active):
                st.session_state.bot_active = True
                st.rerun()
        with c_status_2:
            if st.button("🛑 ปิดระบบบอท", use_container_width=True, disabled=not st.session_state.bot_active):
                st.session_state.bot_active = False
                st.rerun()

    st.markdown("<h3>Manual Control (กดมือ)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        c_buy, c_sell = st.columns(2)
        with c_buy:
            if st.button("🚀 OPEN LONG", use_container_width=True):
                res = fire_execution_order("LONG", live_price, base_mgn, lev, tp_p, sl_p, is_manual=True)
                st.session_state.logs.append(f"🖐️ [{time.strftime('%H:%M:%S')}] {res}")
                st.rerun()
        with c_sell:
            if st.button("💥 OPEN SHORT", use_container_width=True):
                res = fire_execution_order("SHORT", live_price, base_mgn, lev, tp_p, sl_p, is_manual=True)
                st.session_state.logs.append(f"🖐️ [{time.strftime('%H:%M:%S')}] {res}")
                st.rerun()
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🛑 CLOSE ALL POSITIONS (ล้างพอร์ตทั้งหมด)", use_container_width=True, type="secondary"):
            res = close_all_positions()
            st.session_state.logs.append(f"🚨 [{time.strftime('%H:%M:%S')}] {res}")
            st.rerun()

# --- บอทออโต้สแกนยิงตามสัญญาณ ---
if st.session_state.bot_active and signal in ["LONG", "SHORT"] and bar_time != st.session_state.last_processed_bar:
    if signal == "LONG" and active_l < max_t:
        res = fire_execution_order("LONG", live_price, base_mgn, lev, tp_p, sl_p)
        st.session_state.logs.append(f"🟢 [{time.strftime('%H:%M:%S')}] {res}")
        st.session_state.last_processed_bar = bar_time
        st.rerun()
    elif signal == "SHORT" and active_s < max_t:
        res = fire_execution_order("SHORT", live_price, base_mgn, lev, tp_p, sl_p)
        st.session_state.logs.append(f"🔴 [{time.strftime('%H:%M:%S')}] {res}")
        st.session_state.last_processed_bar = bar_time
        st.rerun()

with col_center:
    st.markdown(f"<h3>{SYMBOL} Real-Time Chart</h3>", unsafe_allow_html=True)
    if not df_market.empty:
        fig = px.Figure()
        fig.add_trace(px.Candlestick(
            x=df_market['datetime'], open=df_market['open'], high=df_market['high'],
            low=df_market['low'], close=df_market['close'], name="ราคาตลาด"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='#12161f', plot_bgcolor='#12161f',
            margin=dict(l=5, r=5, t=5, b=5), height=240, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- ตารางแสดงออเดอร์จำลองแยกไม้แบบแกะกล่องสแกนรายตัว ---
    st.markdown("<h3>Virtual Positions (ระบบจำลองแยกไม้)</h3>", unsafe_allow_html=True)
    if st.session_state.virtual_orders:
        for order in st.session_state.virtual_orders:
            # คำนวณ P/L ของตั๋วใบนี้เทียบกับราคาตลาดปัจจุบัน ณ วินาทีนี้แบบแมนนวล
            entry = order['Entry']
            amt = order['Amount']
            side = order['Side']
            
            if side == "LONG":
                pnl_usd = (live_price - entry) * amt
                pnl_pct = ((live_price - entry) / entry) * 100 * order['Leverage']
            else:
                pnl_usd = (entry - live_price) * amt
                pnl_pct = ((entry - live_price) / entry) * 100 * order['Leverage']
            
            pnl_color = "#00e676" if pnl_usd >= 0 else "#ff1744"
            
            # วาดแถวข้อมูลและปุ่มสั่ง Partial Close แยกทีละตั๋ว
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1, 1.5, 2, 2, 1.5])
                c1.markdown(f"**{order['Ticket']}** \n <span style='color:#7a8599;'>{order['Time']}</span>", unsafe_allow_html=True)
                
                badge_side = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
                c2.markdown(f"{badge_side}\n`Amt: {amt}`", unsafe_allow_html=True)
                c3.markdown(f"Entry: `${entry}`\nMargin: `${order['Margin']}`", unsafe_allow_html=True)
                c4.markdown(f"<span style='color:{pnl_color}; font-weight:bold;'>P/L: ${pnl_usd:.4f}<br>({pnl_pct:.2f}%)</span>", unsafe_allow_html=True)
                
                if c5.button("❌ ปิดไม้นี้", key=f"close_{order['Ticket']}", use_container_width=True):
                    res = close_specific_virtual_order(order['Ticket'])
                    st.session_state.logs.append(f"🚨 [{time.strftime('%H:%M:%S')}] {res}")
                    st.rerun()
    else:
        st.info("ℹ️ พอร์ตจำลองว่างเปล่า (รอสัญญาณหรือกดเปิดมือเพื่อสร้างตั๋วแยกไม้)")

with col_right:
    st.markdown("<h3>Performance Dashboard</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Net Equity", f"${init_cap}")
        m2.metric("Margin ไม้แรก", f"${base_mgn}")
        m3, m4 = st.columns(2)
        m3.metric("Active L/S (ตั๋ว)", f"{active_l} / {active_s}")
        m4.metric("Live Price", f"${live_price}")
        st.markdown("<hr style='margin:10px 0; border-color:#1e2533;'>", unsafe_allow_html=True)
        st.caption(f"• สถานะระบบวิเคราะห์: {signal}")

    st.markdown("<h3>Signal Log (บันทึกสแกน)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        for log in reversed(st.session_state.logs[-8:]):
            st.caption(log)

time.sleep(2)
st.rerun()
