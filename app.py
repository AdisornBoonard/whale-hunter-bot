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
st.set_page_config(page_title="Whale Hunter V8.5 - Ultimate Fixed", layout="wide", initial_sidebar_state="collapsed")

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
if 'cached_trades' not in st.session_state:
    st.session_state.cached_trades = pd.DataFrame()

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
    # ดักฟังก์ชันคำนวณเบี่ยงเบน
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
        return round(float(usdt_bal.get('total', 0.0)), 2), round(float(usdt_bal.get('free', 0.0)), 2)
    except Exception as e:
        return 0.0, 0.0

# --- 3. CORE TRADING ENGINE ---
def get_market_and_signal(use_ema, ema_length, ema_reverse_dist, use_cci, cci_length, cci_ob, cci_os):
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=150)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['mfi'] = calculate_mfi(df, length=MFI_LENGTH)
        df['ema'] = df['close'].ewm(span=ema_length, adjust=False).mean() 
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
            if c_cci > cci_ob: s_sig = True; l_sig = False
            elif c_cci < cci_os: l_sig = True; s_sig = False
                
        live_price = df.iloc[-1]['close']
        live_cci = df.iloc[-1]['cci']
        bar_time = df.iloc[-1]['timestamp']
        
        debug_txt = f"Dist: {ema_distance:.2f}% | CCI: {live_cci:.2f}"
        
        if l_sig: return "LONG", live_price, bar_time, df, "SIGNAL_TRIGGERED", live_cci
        if s_sig: return "SHORT", live_price, bar_time, df, "SIGNAL_TRIGGERED", live_cci
        return "HOLD", live_price, bar_time, df, debug_txt, live_cci
    except Exception as ex:
        return "ERROR", 0, None, pd.DataFrame(), str(ex), 0.0

def update_trades_cache():
    try:
        trades = exchange.fetch_my_trades(symbol=SYMBOL, limit=20)
        order_markers = []
        for t in trades:
            order_markers.append({
                "datetime": pd.to_datetime(t['timestamp'], unit='ms'),
                "side": t.get('info', {}).get('positionSide', '').upper(),
                "price": float(t['price']),
                "amount": float(t['amount']),
                "trade_side": t.get('side', '').lower()
            })
        st.session_state.cached_trades = pd.DataFrame(order_markers)
    except Exception as e:
        print(f"Cache update error: {e}")

def process_virtual_orders_from_cache(live_price, active_margin, leverage, max_tickets_allowed):
    virtual_orders = []
    l_count, s_count = 0, 0
    try:
        positions = exchange.fetch_positions()
        active_sides = []
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if 'GOLD' in pos_symbol or 'NCCO' in pos_symbol:
                    active_sides.append(pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT'))

        if active_sides and not st.session_state.cached_trades.empty:
            df_t = st.session_state.cached_trades
            for _, row in df_t.iterrows():
                pos_side = row['side']
                trade_side = row['trade_side']
                
                if pos_side in active_sides:
                    if (pos_side == 'LONG' and trade_side == 'buy') or (pos_side == 'SHORT' and trade_side == 'sell'):
                        if pos_side == 'LONG' and l_count < max_tickets_allowed:
                            l_count += 1
                            idx_num = l_count
                        elif pos_side == 'SHORT' and s_count < max_tickets_allowed:
                            s_count += 1
                            idx_num = s_count
                        else:
                            continue
                            
                        virtual_orders.append({
                            "Ticket": f"ไม้ที่ #{idx_num}",
                            "Index": idx_num,
                            "Side": pos_side,
                            "Amount": row['amount'],
                            "Entry": row['price'],
                            "Margin": round(active_margin, 2),
                            "Leverage": leverage
                        })
    except:
        pass
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
            st.error("❌ ไม่รู้จักฝั่ง")
            return

        main_params = {'positionSide': pos_side}
        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=contract_amount, params=main_params)
        
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': pos_side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=SYMBOL, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': pos_side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        update_trades_cache()
        tg_msg = (f"{emoji_side} *Whale Hunter ยิงสำเร็จ!*\n• *โหมด:* {mode_text}\n• *ฝั่ง:* {pos_side}\n• *ราคาเข้าจริง:* ${entry_price}\n• *ใช้ Margin จริง:* ${margin_size:.4f}\n🎯 *TP:* ${tp_price} | 🛑 *SL:* ${sl_price}")
        send_telegram_message(tg_msg)
        st.success(f"เปิด {pos_side} สำเร็จ!")
    except Exception as e:
        st.error(f"❌ สั่งซื้อล้มเหลว: {e}")

def close_specific_virtual_order(side, amount):
    try:
        close_side = 'sell' if side == 'LONG' else 'buy'
        exchange.create_order(symbol=SYMBOL, type='market', side=close_side, amount=abs(float(amount)), params={'positionSide': side})
        update_trades_cache()
        return True
    except Exception as e:
        st.error(f"❌ ปิดไม้ย่อยล้มเหลว: {e}")
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
        update_trades_cache()
        st.success(f"✅ เคลียร์พอร์ตสำเร็จ ({closed_count} ฝั่ง)")
    except Exception as e:
        st.error(f"❌ ข้อผิดพลาดเคลียร์พอร์ต: {e}")

# --- 4. STREAMLIT UI LAYOUT ---
bot_status_indicator = "● LIVE AUTOTRADING ACTIVE" if st.session_state.bot_active else "○ AUTOTRADING DISABLED"
bot_status_color = "#00e676" if st.session_state.bot_active else "#ff1744"

st.markdown(f"""
<div style="background-color: #12161f; border: 1px solid #1e2533; padding: 10px 20px; border-radius: 6px; margin-bottom: 20px;">
    <span style="font-size: 20px; font-weight: bold; color: white;">WHALE HUNTER V8.5 - ULTIMATE ENGINE</span>
    <span style="float: right; color: {bot_status_color}; font-weight: bold; padding-top: 5px;">{bot_status_indicator}</span>
</div>
""", unsafe_allow_html=True)

# 🛠️ โหลดข้อมูล Cache เริ่มต้นถ้ามันว่างเปล่า
if 'first_init' not in st.session_state:
    update_trades_cache()
    st.session_state.first_init = True

col_left, col_center, col_right = st.columns([1, 2, 1])

# 🎯 ย้ายส่วนรับค่า Inputs มาไว้ข้างบนสุด เพื่อไม่ให้ตัวแปรขาดตอนและบั๊กฟังก์ชันหาย
with col_left:
    st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        total_capital, available_capital = get_balance()
        st.metric("เงินทุนสุทธิในกระดาน", f"${total_capital}")
        
        base_mgn = st.number_input("Margin ไม้แรก ($)", value=1.00, format="%.4f", step=0.1)  
        daily_add = st.number_input("เพิ่ม Margin วันละ ($)", value=3.00, format="%.4f", step=0.01)
        lev = st.number_input("Leverage (x)", value=250, min_value=1, max_value=250)
        max_t = st.number_input("เปิดสูงสุด (ต่อฝั่ง)", value=10)
        
        st.session_state.tp_percent = st.slider("TP (%)", 0.1, 5.0, st.session_state.tp_percent)
        st.session_state.sl_percent = st.slider("SL (%)", 0.1, 5.0, st.session_state.sl_percent)
        
        # 🛠️ ดึงช่องอินพุต EMA กับ CCI กลับมาวางตรงนี้ถาวร ไม่หายแน่นอนครับพี่
        st.markdown("<h4 style='color:#90a4ae; font-size:12px; margin-top:10px;'>EMA FILTER CONFIG</h4>", unsafe_allow_html=True)
        u_ema = st.checkbox("เปิดใช้งาน EMA Filter", value=True)
        e_len = st.number_input("EMA Length", value=200, min_value=1, step=10)
        e_dist = st.number_input("Reverse Distance From EMA (%)", value=1.5, step=0.1)
        
        st.markdown("<h4 style='color:#90a4ae; font-size:12px; margin-top:10px;'>CCI REVERSAL FILTER</h4>", unsafe_allow_html=True)
        u_cci = st.checkbox("เปิดใช้งาน CCI Reversal", value=True)
        c_len = st.number_input("CCI Length", value=100, min_value=1)
        c_ob = st.number_input("CCI Overbought (Sell)", value=40.0, step=10.0)
        c_os = st.number_input("CCI Oversold (Long)", value=-150.0, step=10.0)
        
        days_passed = math.floor((time.time() - st.session_state.bot_start_time) / (24 * 60 * 60))
        current_mgn_active = base_mgn + (days_passed * daily_add)
        
        st.markdown(f"""
            <div class='margin-box'>
                <span style='font-size: 13px; font-weight: bold; color: white;'>CURRENT ORDER MARGIN</span><br>
                <span style='font-size: 22px; font-weight: bold; color: #64b5f6; font-family: monospace;'>${current_mgn_active:.4f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        c_status_1, c_status_2 = st.columns(2)
        with c_status_1:
            if st.button("🟢 เปิดรันบอท", use_container_width=True):
                st.session_state.bot_active = True
                st.rerun()
        with c_status_2:
            if st.button("🛑 ปิดระบบบอท", use_container_width=True):
                st.session_state.bot_active = False
                st.rerun()

    st.markdown("<h3>Manual Control</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        c_buy, c_sell = st.columns(2)
        with c_buy:
            if st.button("🚀 OPEN LONG", use_container_width=True):
                fire_execution_order("LONG", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent, is_manual=True)
                st.rerun()
        with c_sell:
            if st.button("💥 OPEN SHORT", use_container_width=True):
                fire_execution_order("SHORT", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent, is_manual=True)
                st.rerun()
        
        if st.button("🛑 CLOSE ALL POSITIONS", use_container_width=True, type="secondary"):
            close_all_positions()
            st.rerun()

# 🎯 รันฟังก์ชันคำนวณสัญญาณด้านล่าง เพื่อนำข้อมูลมาพลอตกราฟตรงกลาง (ไม่พังแล้ว)
signal, live_price, bar_time, df_market, log_debug, current_cci_val = get_market_and_signal(u_ema, e_len, e_dist, u_cci, c_len, c_ob, c_os)
virtual_orders_list, active_l, active_s = process_virtual_orders_from_cache(live_price, current_mgn_active, lev, max_t)

# --- ระบบออโต้สแกนยิงคำสั่ง ---
if st.session_state.bot_active and signal in ["LONG", "SHORT"]:
    if 'last_order_time' not in st.session_state: st.session_state.last_order_time = 0
    current_time_sec = time.time()
    if (current_time_sec - st.session_state.last_order_time) > 50:
        if signal == "LONG" and active_l < max_t:
            fire_execution_order("LONG", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent)
            st.session_state.last_order_time = current_time_sec
            st.rerun()
        elif signal == "SHORT" and active_s < max_t:
            fire_execution_order("SHORT", live_price, current_mgn_active, lev, st.session_state.tp_percent, st.session_state.sl_percent)
            st.session_state.last_order_time = current_time_sec
            st.rerun()

with col_center:
    st.markdown(f"<h3>{SYMBOL} Chart</h3>", unsafe_allow_html=True)
    if not df_market.empty:
        fig = px.Figure()
        fig.add_trace(px.Candlestick(x=df_market['datetime'], open=df_market['open'], high=df_market['high'], low=df_market['low'], close=df_market['close'], name="ราคา"))
        
        # วาดจุดประวัติเข้าเทรดจากข้อมูลความจำ
        if not st.session_state.cached_trades.empty:
            df_hist = st.session_state.cached_trades
            long_m = df_hist[df_hist['side'] == 'LONG']
            short_m = df_hist[df_hist['side'] == 'SHORT']
            if not long_m.empty:
                fig.add_trace(px.scatter(long_m, x='datetime', y='price').data[0])
                fig.data[-1].update(mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676', line=dict(width=1, color='white')), name='LONG ENTRY')
            if not short_m.empty:
                fig.add_trace(px.scatter(short_m, x='datetime', y='price').data[0])
                fig.data[-1].update(mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff1744', line=dict(width=1, color='white')), name='SHORT ENTRY')

        fig.update_layout(template="plotly_dark", paper_bgcolor='#12161f', plot_bgcolor='#12161f', margin=dict(l=5, r=5, t=5, b=5), height=240, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ ไม่สามารถโหลดข้อมูลแท่งเทียนจาก BingX ได้ชั่วคราว (กรุณารอปลดแบน IP)")

    # --- แสดงตารางไม้เสมือน ---
    st.markdown("<h3>Virtual Positions</h3>", unsafe_allow_html=True)
    
    if st.button("🔄 ดึงข้อมูลตั๋วรายไม้ล่าสุด (กดเพื่ออัปเดตตั๋วใหม่)", use_container_width=True):
        update_trades_cache()
        st.rerun()
        
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
                c1.markdown(f"**{order['Ticket']}**")
                c2.markdown(f"{'🟢 LONG' if side == 'LONG' else '🔴 SHORT'}\n`Amt: {amt:.4f}`")
                
                c3.html(f"""
                    <div style='font-family: monospace; font-size: 13px; color: white; line-height: 1.4;'>
                        Real Entry: <span style='color: #fff176; font-weight:bold;'>${entry:.2f}</span><br>
                        <span style='color: #00e676; font-weight: bold;'>🎯 TP: ${target_tp:.2f}</span><br>
                        <span style='color: #ff1744; font-weight: bold;'>🛑 SL: ${target_sl:.2f}</span>
                    </div>
                """)
                c4.markdown(f"<span style='color:{pnl_color}; font-weight:bold;'>P/L: ${pnl_usd:.4f}<br>({pnl_pct:.2f}%)</span>", unsafe_allow_html=True)
                
                if c5.button("❌ ปิดไม้นี้", key=f"btn_close_{side}_{idx_num}", use_container_width=True):
                    if close_specific_virtual_order(side, amt):
                        time.sleep(1)
                        st.rerun()
    else:
        st.info("ℹ️ พอร์ตว่างเปล่า (หรือไม่มีตั๋วที่บันทึกในระบบแคช)")

with col_right:
    st.markdown("<h3>Dashboard</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Net Equity", f"${total_capital}")
        m2.metric("Available", f"${available_capital}")
        m3, m4 = st.columns(2)
        m3.metric("สรุปไม้จริง (L/S)", f"{active_l} / {active_s}")
        m4.metric("Live Price", f"${live_price}")

    st.markdown("<h3>Signal Log</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(f"⏱️ เวลาคลาวด์: {time.strftime('%H:%M:%S')}")
        st.caption(f"• ตัวเลขแกะรอยระบบ: `{log_debug}`")

time.sleep(3)
st.rerun()
