import os
import time
import math
import ccxt
import requests
import pandas as pd
import plotly.graph_objects as px
import streamlit as st
from dotenv import load_dotenv

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Whale Hunter V8.2 - Dynamic Matrix", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        body, .main, .block-container { background-color: #0b0e14 !important; color: white !important; }
        div[data-testid="stMetricValue"] { color: #00e676 !important; font-family: monospace; font-size: 24px; }
        .stTable { background-color: #12161f !important; border: 1px solid #1e2533 !important; border-radius: 6px; }
        h3 { color: #90a4ae !important; font-size: 14px !important; text-transform: uppercase; letter-spacing: 0.5px; }
        .stCaption { font-family: monospace; font-size: 12px; }
        .margin-box { background-color: #0d47a1 !important; border: 1px solid #1565c0 !important; padding: 10px; border-radius: 4px; text-align: center; margin-bottom: 15px; }
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
VOL_MULTIPLIER = 0.7

# --- ⚙️ PERSISTENT ENGINE STATES (ระบบจดจำรอบบอท) ---
if 'bot_start_time' not in st.session_state:
    st.session_state.bot_start_time = time.time()
if 'tp_percent' not in st.session_state:
    st.session_state.tp_percent = 0.50
if 'sl_percent' not in st.session_state:
    st.session_state.sl_percent = 0.30
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

def calculate_cci(df, length=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=length).mean()
    mad = typical_price.rolling(window=length).apply(lambda x: pd.Series(x).mad() if hasattr(pd.Series(x), 'mad') else np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def get_balance():
    try:
        bal = exchange.fetch_balance()
        usdt_bal = bal.get('USDT', {})
        total_balance = float(usdt_bal.get('total', 0.0))
        available_balance = float(usdt_bal.get('free', 0.0))
        return round(total_balance, 2), round(available_balance, 2)
    except:
        return 0.0, 0.0

# ฟังก์ชันดึงประวัติเปิดออเดอร์จริงมาวางพิกัดบนกราฟป้องกันจุดเพี้ยนเวลาบอทรันข้ามรอบ
def get_executed_orders_from_exchange():
    order_markers = []
    try:
        trades = exchange.fetch_my_trades(symbol=SYMBOL, limit=40)
        for t in trades:
            if t.get('info', {}).get('positionSide') in ['LONG', 'SHORT'] and t.get('side') in ['buy', 'sell']:
                pos_side = t['info']['positionSide']
                if (pos_side == 'LONG' and t['side'] == 'buy') or (pos_side == 'SHORT' and t['side'] == 'sell'):
                    order_markers.append({
                        "datetime": pd.to_datetime(t['timestamp'], unit='ms'),
                        "side": pos_side,
                        "price": float(t['price'])
                    })
    except:
        pass
    return pd.DataFrame(order_markers)

# --- 3. CORE TRADING ENGINE (CCI + Dynamic EMA Reversal) ---
def get_market_and_signal(use_ema, ema_length, ema_reverse_dist, use_cci, cci_length, cci_ob, cci_os):
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=150)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['mfi'] = calculate_mfi(df, length=MFI_LENGTH)
        df['ema'] = df['close'].ewm(span=ema_length, adjust=False).mean() # 🛠️ ปรับเป็น Dynamic ความยาวเส้นตามต้องการ
        df['v_ma'] = df['volume'].rolling(window=20).mean()
        df['cci'] = calculate_cci(df, length=cci_length)
        
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
        far_from_ema = ema_distance >= ema_reverse_dist
        
        l_sig, s_sig = False, False
        
        if not use_ema:
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
                    
        if use_cci and (l_sig or s_sig):
            if c_cci > cci_ob:
                s_sig = True
                l_sig = False
            elif c_cci < cci_os:
                l_sig = True
                s_sig = False
                
        live_price = df.iloc[-1]['close']
        live_cci = df.iloc[-1]['cci']
        bar_time = df.iloc[-1]['timestamp']
        
        debug_txt = f"Dist: {ema_distance:.2f}% | CCI: {live_cci:.2f}"
        
        if l_sig: return "LONG", live_price, bar_time, df, "SIGNAL_TRIGGERED", live_cci
        if s_sig: return "SHORT", live_price, bar_time, df, "SIGNAL_TRIGGERED", live_cci
        return "HOLD", live_price, bar_time, df, debug_txt, live_cci
    except Exception as ex:
        return "ERROR", 0, None, pd.DataFrame(), str(ex), 0.0

# 🛠️ ระบบคัดกรองข้อมูลไม้จริงแบบแตกราคาเข้าอิสระ ป้องกันเศษทศนิยมปูดไม้เกินจริง
def rebuild_virtual_orders(live_price, active_margin, leverage, max_tickets_allowed):
    virtual_orders = []
    l_count, s_count = 0, 0
    try:
        positions = exchange.fetch_positions()
        # คำนวณหาขนาดมาตรฐานของสัญญาต่อ 1 ตั๋ว
        one_ticket_amt = round((active_margin * leverage) / live_price, 4) if live_price > 0 else 0.0001
        
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if 'GOLD' in pos_symbol or 'NCCO' in pos_symbol:
                    side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                    real_entry_price = float(pos.get('entryPrice', live_price))
                    total_amt = abs(size)
                    
                    # 🎯 ใช้ math.floor ดักปัดเศษลง และใช้ min ครอบล็อกไว้ไม่ให้จำนวนไม้ในตารางปูดเกินจริง
                    raw_tickets = math.floor(total_amt / one_ticket_amt)
                    num_tickets = max(1, min(raw_tickets, max_tickets_allowed))
                    
                    if side == 'LONG': l_count = num_tickets
                    else: s_count = num_tickets
                    
                    for i in range(num_tickets):
                        virtual_orders.append({
                            "Ticket": f"ไม้ที่ #{i+1}",
                            "Index": i+1,
                            "Side": side,
                            "Amount": round(total_amt / num_tickets, 4),
                            "Entry": real_entry_price, # ใช้ Logic ประมวลผลแยกราคา TP/SL ขาดจากกันแบบเดิมของพี่
                            "Margin": round(active_margin, 2),
                            "Leverage": leverage
                        })
    except Exception as e:
        print(f"Error rebuilding orders: {e}")
    return virtual_orders, l_count, s_count

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
            return "❌ ไม่รู้จักฝั่ง"

        main_params = {'positionSide': pos_side}
        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params=main_params)
        
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': pos_side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': pos_side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        tg_msg = (f"{emoji_side} *Whale Hunter V8.2 ยิงสำเร็จ!*\n• *โหมด:* {mode_text}\n• *ฝั่ง:* {pos_side}\n• *ราคาเข้าจริง:* ${entry_price}\n• *ใช้ Margin จริง:* ${margin_size:.4f}\n🎯 *TP:* ${tp_price} | 🛑 *SL:* ${sl_price}")
        send_telegram_message(tg_msg)
        return f"เปิด {pos_side} สำเร็จ"
    except Exception as e:
        return f"❌ สั่งซื้อล้มเหลว: {e}"

def close_specific_virtual_order(side, amount):
    try:
        close_side = 'sell' if side == 'LONG' else 'buy'
        target_amount = abs(float(amount))
        exchange.create_order(symbol=SYMBOL, type='market', side=close_side, amount=target_amount, params={'positionSide': side})
        return True
    except Exception as e:
        st.error(f"API Error: {e}")
        return False

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
                    exchange.create_order(symbol=SYMBOL, type='market', side=close_side, amount=abs(size), params={'positionSide': side})
                    closed_count += 1
        return f"✅ เคลียร์พอร์ตสำเร็จ" if closed_count > 0 else "ℹ️ ไม่มีออเดอร์ค้าง"
    except Exception as e:
        return f"❌ ข้อผิดพลาด: {e}"

# --- 4. STREAMLIT FRONT-END PANEL ---
bot_status_indicator = "● LIVE AUTOTRADING ACTIVE" if st.session_state.bot_active else "○ AUTOTRADING DISABLED"
bot_status_color = "#00e676" if st.session_state.bot_active else "#ff1744"

st.markdown(f"""
<div style="background-color: #12161f; border: 1px solid #1e2533; padding: 10px 20px; border-radius: 6px; margin-bottom: 20px;">
    <span style="font-size: 20px; font-weight: bold; color: white;">WHALE HUNTER V8.2 - DYNAMIC MATRIX ENGINE</span>
    <span style="float: right; color: {bot_status_color}; font-weight: bold; padding-top: 5px;">{bot_status_indicator}</span>
</div>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        total_capital, available_capital = get_balance()
        st.metric("เงินทุนสุทธิในกระดาน", f"${total_capital}")
        
        base_mgn = st.number_input("Margin ไม้แรก ($)", value=1.00, format="%.4f", step=0.1)  
        daily_add = st.number_input("เพิ่ม Margin วันละ ($)", value=3.00, format="%.4f", step=0.01)
        
        lev = st.number_input("Leverage (x)", value=250, min_value=1, max_value=250)
        max_t = st.number_input("เปิดสูงสุด (ต่อฝั่ง)", value=3)
        
        st.session_state.tp_percent = st.slider("TP (%)", 0.1, 5.0, st.session_state.tp_percent)
        st.session_state.sl_percent = st.slider("SL (%)", 0.1, 5.0, st.session_state.sl_percent)
        
        # --- 🛠️ EMA Filter Options (มีช่องปรับตั้งค่าความยาวและระยะห่างครบถ้วน) ---
        st.markdown("<h4 style='color:#90a4ae; font-size:12px; margin-top:10px;'>EMA FILTER CONFIG</h4>", unsafe_allow_html=True)
        u_ema = st.checkbox("เปิดใช้งาน EMA Filter", value=True)
        e_len = st.number_input("EMA Length", value=200, min_value=1, step=10) # 🎯 ช่องกรอกความยาวเส้นอิสระตามคำสั่งซื้อ
        e_dist = st.number_input("Reverse Distance From EMA (%)", value=1.5, step=0.1)
        
        # --- CCI Reversal Parameters ---
        st.markdown("<h4 style='color:#90a4ae; font-size:12px; margin-top:10px;'>CCI REVERSAL FILTER</h4>", unsafe_allow_html=True)
        u_cci = st.checkbox("เปิดใช้งาน CCI Reversal", value=True)
        c_len = st.number_input("CCI Length", value=100, min_value=1)
        c_ob = st.number_input("CCI Overbought (Sell)", value=40.0, step=10.0)
        c_os = st.number_input("CCI Oversold (Long)", value=-150.0, step=10.0)
        
        days_passed = math.floor((time.time() - st.session_state.bot_start_time) / (24 * 60 * 60))
        current_mgn_active = base_mgn + (days_passed * daily_add)
        
        st.markdown(f"""
            <div class='margin-box'>
                <span style='font-size: 11px; color: #bbdefb;'>รันมาแล้ว {days_passed} วัน</span><br>
                <span style='font-size: 13px; font-weight: bold; color: white;'>CURRENT ORDER MARGIN</span><br>
                <span style='font-size: 22px; font-weight: bold; color: #64b5f6; font-family: monospace;'>${current_mgn_active:.4f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        c_status_1, c_status_2 = st.columns(2)
        with c_status_1:
            if st.button("🟢 เปิดรันบอท", use_container_width=True):
                st.session_state.bot_active = True
                st.session_state.bot_start_time = time.time()
                send_telegram_message(f"🟢 *Whale Hunter V8.2:* เริ่มรันระบบแยกคำนวณไม้แบบอิสระ")
                st.rerun()
        with c_status_2:
            if st.button("🛑 ปิดระบบบอท", use_container_width=True):
                st.session_state.bot_active = False
                send_telegram_message("🛑 *Whale Hunter:* ปิดระบบบอทออโต้ชั่วคราว")
                st.rerun()

    st.markdown("<h3>Manual Control (กดมือ)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        c_buy, c_sell = st.columns(2)
        with c_buy:
            if st.button("🚀 OPEN LONG", use_container_width=True):
                res = fire_execution_order("LONG", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent, is_manual=True)
                st.rerun()
        with c_sell:
            if st.button("💥 OPEN SHORT", use_container_width=True):
                res = fire_execution_order("SHORT", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent, is_manual=True)
                st.rerun()
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🛑 CLOSE ALL POSITIONS (เคลียร์พอร์ต)", use_container_width=True, type="secondary"):
            res = close_all_positions()
            st.rerun()

# ประมวลผลและดึงข้อมูลตลาด
signal, live_price, bar_time, df_market, log_debug, current_cci_val = get_market_and_signal(u_ema, e_len, e_dist, u_cci, c_len, c_ob, c_os)
# 🛠️ ส่งค่า max_t เข้าไปล็อกเพดานจำลองไม้ตาราง Virtual Positions
virtual_orders_list, active_l, active_s = rebuild_virtual_orders(live_price, current_mgn_active, lev, max_t)

# --- ระบบออโต้สแกนยิงคำสั่ง ---
if st.session_state.bot_active and signal in ["LONG", "SHORT"]:
    if 'last_order_time' not in st.session_state:
        st.session_state.last_order_time = 0
        
    current_time_sec = time.time()
    if (current_time_sec - st.session_state.last_order_time) > 50:
        if signal == "LONG" and active_l < max_t:
            res = fire_execution_order("LONG", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent)
            st.session_state.last_order_time = current_time_sec
            st.rerun()
        elif signal == "SHORT" and active_s < max_t:
            res = fire_execution_order("SHORT", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent)
            st.session_state.last_order_time = current_time_sec
            st.rerun()

with col_center:
    st.markdown(f"<h3>{SYMBOL} Real-Time Chart</h3>", unsafe_allow_html=True)
    if not df_market.empty:
        fig = px.Figure()
        fig.add_trace(px.Candlestick(x=df_market['datetime'], open=df_market['open'], high=df_market['high'], low=df_market['low'], close=df_market['close'], name="ราคา"))
        
        # ดึงประวัติเข้าออเดอร์หลักตรงจาก exchange มาแสดงจุดพลอตแบบเสถียร 
        hist_df = get_executed_orders_from_exchange()
        if not hist_df.empty:
            long_marks = hist_df[hist_df['side'] == 'LONG']
            short_marks = hist_df[hist_df['side'] == 'SHORT']
            
            if not long_marks.empty:
                fig.add_trace(px.scatter(long_marks, x='datetime', y='price').data[0])
                fig.data[-1].update(mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676', line=dict(width=1, color='white')), name='LONG ENTRY')
            if not short_marks.empty:
                fig.add_trace(px.scatter(short_marks, x='datetime', y='price').data[0])
                fig.data[-1].update(mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff1744', line=dict(width=1, color='white')), name='SHORT ENTRY')

        fig.update_layout(template="plotly_dark", paper_bgcolor='#12161f', plot_bgcolor='#12161f', margin=dict(l=5, r=5, t=5, b=5), height=240, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 🛠️ แสดงตารางไม้เสมือนตามเงื่อนไข Advance Virtualizer แยกตรวจจับรายตั๋ว
    st.markdown("<h3>Virtual Positions (ตารางแยกคำนวณรายไม้แบบอิสระ)</h3>", unsafe_allow_html=True)
    if virtual_orders_list:
        for order in virtual_orders_list:
            entry = order['Entry'] 
            amt = order['Amount']
            side = order['Side']
            idx_num = order['Index']
            
            tp_factor = st.session_state.tp_percent / 100
            sl_factor = st.session_state.sl_percent / 100
            
            if side == "LONG":
                pnl_usd = (live_price - entry) * amt
                pnl_pct = ((live_price - entry) / entry) * 100 * order['Leverage']
                target_tp = entry * (1 + tp_factor)
                target_sl = entry * (1 - sl_factor)
            else:
                pnl_usd = (entry - live_price) * amt
                pnl_pct = ((entry - live_price) / entry) * 100 * order['Leverage']
                target_tp = entry * (1 - tp_factor)
                target_sl = entry * (1 + sl_factor)
            
            pnl_color = "#00e676" if pnl_usd >= 0 else "#ff1744"
            
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1, 1.2, 2.3, 2, 1.5])
                c1.markdown(f"**{order['Ticket']}**", unsafe_allow_html=True)
                badge_side = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
                c2.markdown(f"{badge_side}\n`Amt: {amt}`", unsafe_allow_html=True)
                
                c3.html(f"""
                    <div style='font-family: monospace; font-size: 13px; color: white; line-height: 1.4;'>
                        Real Entry: <span style='color: #fff176; font-weight:bold;'>${entry:.2f}</span><br>
                        <span style='color: #00e676; font-weight: bold;'>🎯 TP: ${target_tp:.2f}</span><br>
                        <span style='color: #ff1744; font-weight: bold;'>🛑 SL: ${target_sl:.2f}</span>
                    </div>
                """)
                
                c4.markdown(f"<span style='color:{pnl_color}; font-weight:bold;'>P/L: ${pnl_usd:.4f}<br>({pnl_pct:.2f}%)</span>", unsafe_allow_html=True)
                
                if c5.button("❌ ปิดไม้นี้", key=f"btn_close_{side}_{idx_num}", use_container_width=True):
                    success = close_specific_virtual_order(side, amt)
                    if success:
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.info("ℹ️ พอร์ตว่างเปล่า (ไม่มีออเดอร์ค้างในกระดานเทรด)")

with col_right:
    st.markdown("<h3>Performance Dashboard</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Net Equity", f"${total_capital}")
        m2.metric("Available", f"${available_capital}")
        m3, m4 = st.columns(2)
        m3.metric("สรุปไม้จริง (L/S)", f"{active_l} / {active_s}")
        m4.metric("Live Price", f"${live_price}")
        st.metric("Live CCI Value", f"{current_cci_val:.2f}")

    st.markdown("<h3>Signal Log (บันทึกสแกนเรียลไทม์)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(f"⏱️ เวลาคลาวด์: {time.strftime('%H:%M:%S')}")
        st.caption(f"• โมดูล CCI Filter: **{'ON' if u_cci else 'OFF'}**")
        st.caption(f"• ตัวเลขแกะรอยระบบ: `{log_debug}`")
        if signal in ["LONG", "SHORT"]:
            st.markdown(f"<span style='color:#00e676;'>🎯 สแกนเจอและยิงสัญญาณฝั่ง {signal} สมบูรณ์!</span>", unsafe_allow_html=True)

time.sleep(3)
st.rerun()
