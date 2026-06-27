import os
import time
import ccxt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- CONFIGURATION & PAGE INITIALIZATION ---
st.set_page_config(page_title="Whale Hunter Hybrid Terminal", layout="wide")
load_dotenv()

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

# เชื่อมต่อกระดานเทรด BingX บัญชีหลัก
exchange = ccxt.bingx({
    'apiKey': BINGX_API_KEY,
    'secret': BINGX_SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# --- DATA FETCHING FUNCTIONS ---
def fetch_trades_safe(symbol):
    try:
        trades = exchange.fetch_my_trades(symbol=symbol, limit=30)
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

def process_virtual_orders(df_trades, symbol):
    virtual_tickets = []
    try:
        positions = exchange.fetch_positions()
        active_positions = {}
        for pos in positions:
            size = float(pos.get('contracts', 0)) or float(pos.get('size', 0))
            if size != 0:
                pos_symbol = pos.get('symbol', '').upper()
                if symbol == 'NCCOGOLD2USD/USDT:USDT' and ('GOLD' in pos_symbol or 'NCCO' in pos_symbol):
                    side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                    active_positions[side] = abs(size)
                elif symbol == 'BTC/USDT' and pos_symbol == 'BTC/USDT':
                    side = pos.get('side', '').upper() or ('LONG' if size > 0 else 'SHORT')
                    active_positions[side] = abs(size)

        if active_positions and not df_trades.empty:
            for side, current_actual_size in active_positions.items():
                target_trade_side = 'buy' if side == 'LONG' else 'sell'
                df_filtered = df_trades[(df_trades['side'] == side) & (df_trades['trade_side'] == target_trade_side)]
                accumulated_size = 0.0
                ticket_no = 1
                for _, row in df_filtered.iterrows():
                    if accumulated_size >= current_actual_size: break
                    amt = row['amount']
                    if accumulated_size + amt > current_actual_size: amt = current_actual_size - accumulated_size
                    if amt > 0.0001:
                        virtual_tickets.append({
                            "ไม้ที่": f"{side} #{ticket_no}",
                            "Side": side,
                            "Entry Price": row['price'],
                            "Amount": amt,
                            "Time": pd.to_datetime(row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M')
                        })
                        ticket_no += 1
                    accumulated_size += row['amount']
    except: pass
    return pd.DataFrame(virtual_tickets)

def close_specific_virtual_order(symbol, side, amt, label):
    try:
        order_side = 'sell' if side == 'LONG' else 'buy'
        exchange.create_order(symbol=symbol, type='market', side=order_side, amount=amt, params={'positionSide': side})
        st.success(f"❌ สั่งปิดตั๋ว {label} ฝั่ง {side} ขนาด {amt} สำเร็จ!")
        time.sleep(1)
        st.rerun()
    except Exception as e: st.error(f"ปิดตั๋วไม่สำเร็จ: {e}")

def fire_manual_order(symbol, side, entry_price, margin, lev, tp_pct, sl_pct):
    try:
        contract_amount = round((margin * lev) / entry_price, 4)
        tp_factor = tp_pct / 100
        sl_factor = sl_pct / 100
        
        # ปรับทศนิยมราคาตามประเภทสินทรัพย์
        decimal_place = 1 if 'BTC' in symbol else 2
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), decimal_place)
            sl_price = round(entry_price * (1 - sl_factor), decimal_place)
            order_side = 'buy'
        else:
            tp_price = round(entry_price * (1 - tp_factor), decimal_place)
            sl_price = round(entry_price * (1 + sl_factor), decimal_place)
            order_side = 'sell'

        # ส่งคำสั่ง Market Open Position ไปกระดานเทรด
        exchange.create_order(symbol=symbol, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
        
        # ตั้งเงื่อนไขป้องกัน TP/SL รายไม้พ่วงเข้าไปในระบบกระดานเทรดทันที
        try:
            tp_sl_side = 'sell' if side == 'LONG' else 'buy'
            exchange.create_order(symbol=symbol, type='TAKE_PROFIT_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': tp_price, 'workingType': 'MARK_PRICE'})
            exchange.create_order(symbol=symbol, type='STOP_MARKET', side=tp_sl_side, amount=contract_amount, params={'positionSide': side, 'stopPrice': sl_price, 'workingType': 'MARK_PRICE'})
        except: pass

        st.sidebar.success(f"🚀 ยิงมือสำเร็จ! เปิดฝั่ง {side} ขนาด {contract_amount}")
        time.sleep(1.5)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"ยิงออเดอร์มือล้มเหลว: {e}")

# --- NAVIGATION SIDEBAR (แถบเมนูข้างจอ) ---
st.sidebar.title("📌 เมนูควบคุมระบบ")
page = st.sidebar.radio("เลือกหน้าแดชบอร์ดที่ต้องการดู:", ["🏠 หน้าแรก (Overview)", "🏆 พอร์ตทองคำ / NCCO", "🟠 พอร์ต BTC Futures"])

# ดึงข้อมูลพอร์ตเบื้องต้นมารอแสดงผล
try:
    bal = exchange.fetch_balance()
    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
    avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
except:
    total_cap, avail_cap = 0.0, 0.0

st.sidebar.markdown("---")
# 🛠️ แผงควบคุมกดเข้าออเดอร์ด้วยมือ (MANUAL ENTRY CONTROL PANEL)
st.sidebar.subheader("🕹️ แผงสั่งยิงออเดอร์ด้วยมือ")
manual_asset = st.sidebar.selectbox("เลือกเหรียญที่จะกดมือ:", ["GOLD/NCCO", "BTC/USDT"])

# เซตค่าเริ่มต้นของสัญลักษณ์และเปอร์เซ็นต์ตามที่พี่เคยตั้งไว้ในรูป
if manual_asset == "GOLD/NCCO":
    m_symbol = 'NCCOGOLD2USD/USDT:USDT'
    default_lev = 250
    default_mgn = 0.02
    default_tp = 0.50
    default_sl = 0.30
else:
    m_symbol = 'BTC/USDT'
    default_lev = 150
    default_mgn = 1.0
    default_tp = 2.0
    default_sl = 2.5

try:
    m_ticker = exchange.fetch_ticker(m_symbol)
    m_live_price = float(m_ticker['last'])
except:
    m_live_price = 0.0

# ช่องรับค่าพารามิเตอร์การเปิดไม้ด้วยมือ
m_mgn = st.sidebar.number_input("ระบุขนาดเงินทุน (Margin $)", min_value=0.01, value=default_mgn, step=0.01, format="%.2f")
m_lev = st.sidebar.number_input("ระบุค่า Leverage (X)", min_value=1, max_value=250, value=default_lev, step=1)
m_tp = st.sidebar.number_input("ระบุเป้าหมายกําไร TP (%)", min_value=0.05, value=default_tp, step=0.05, format="%.2f")
m_sl = st.sidebar.number_input("ระบุจุดตัดขาดทุน SL (%)", min_value=0.05, value=default_sl, step=0.05, format="%.2f")

# คำนวณขนาดสัญญากลางก่อนยิงจริง
if m_live_price > 0:
    calc_amt = round((m_mgn * m_lev) / m_live_price, 4)
    st.sidebar.caption(f"📊 ขนาดสัญญาที่จะส่งออก: **{calc_amt}**")
else:
    st.sidebar.caption("⚠️ ไม่สามารถดึงราคาตลาดเพื่อคำนวณสัญญาได้")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🟢 เปิดไม้ LONG", use_container_width=True):
    if m_live_price > 0:
        fire_manual_order(m_symbol, "LONG", m_live_price, m_mgn, m_lev, m_tp, m_sl)

if col_btn2.button("🔴 เปิดไม้ SHORT", use_container_width=True):
    if m_live_price > 0:
        fire_manual_order(m_symbol, "SHORT", m_live_price, m_mgn, m_lev, m_tp, m_sl)


# ----------------------------------------------------
# 1. หน้าแรก (Overview Home Page)
# ----------------------------------------------------
if page == "🏠 หน้าแรก (Overview)":
    st.title("🐳 Whale Hunter V8.9 - Hybrid Dashboard")
    st.subheader("ยินดีต้อนรับสู่ระบบควบคุมพอร์ตแบบไฮบริด")
    st.write("พี่สามารถเลือกเหรียญเพื่อสั่งเปิดไม้มือด่วนได้ที่แถบเมนูด้านซ้าย และคลิกสลับหน้าจอเพื่อดูสถานะ Virtual Positions คราบบบ")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("💰 ทุนสุทธิในพอร์ตรวมทั้งหมด", f"${total_cap} USDT")
    col2.metric("💵 ทุนว่างที่พร้อมช้อน/แก้เกม", f"${avail_cap} USDT")
    
    st.markdown("### 📊 สถานะการทำงานหลังบ้านแยกตามสินทรัพย์")
    st.info("💡 **พอร์ตทองคำ / NCCO:** กำลังรันเฝ้ากราฟใน Timeframe 1m คู่กับระบบหลับ/ตื่นของ cron-job.org ฟรี 100%")
    st.warning("💡 **พอร์ต BTC Futures:** กำลังรันเฝ้ากราฟใน Timeframe 5m โหมด Pure Divergence ไร้ฟิลเตอร์ รันแยกบอทเงียบ ๆ")

# ----------------------------------------------------
# 2. หน้าพอร์ตทองคำ (Gold / NCCO)
# ----------------------------------------------------
elif page == "🏆 พอร์ตทองคำ / NCCO":
    st.title("🏆 Whale Hunter - Gold & NCCO Terminal")
    st.write("สถานะไม้แยกและการจัดการรายตั๋วเดี่ยว (Timeframe 1m)")
    
    SYMBOL_GOLD = 'NCCOGOLD2USD/USDT:USDT'
    LEVERAGE_GOLD = 250
    
    try:
        ticker = exchange.fetch_ticker(SYMBOL_GOLD)
        live_price = float(ticker['last'])
    except: live_price = 0.0
    
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("ราคาปัจจุบัน ทองคำ/NCCO", f"${live_price}")
    col_g2.metric("ทุนรวมในพอร์ต", f"${total_cap} USDT")
    col_g3.metric("ทุนว่างคงเหลือ", f"${avail_cap} USDT")
    
    st.markdown("---")
    df_trades_g = fetch_trades_safe(SYMBOL_GOLD)
    df_vt_g = process_virtual_orders(df_trades_g, SYMBOL_GOLD)
    
    if not df_vt_g.empty:
        for idx, row in df_vt_g.iterrows():
            entry = row['Entry Price']
            amt = row['Amount']
            side = row['Side']
            
            diff = (live_price - entry) if side == "LONG" else (entry - live_price)
            p_l_usd = diff * amt
            p_l_pct = (diff / entry) * 100 * LEVERAGE_GOLD
            color = "green" if p_l_usd >= 0 else "red"
            
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1.5, 1.5, 1.5])
            c1.write(f"**{row['ไม้ที่']}**")
            c2.write(f"ราคาเข้า: `${entry}`")
            c3.write(f"ขนาด: `{amt}`")
            c4.markdown(f"P/L ($): <span style='color:{color}; font-weight:bold;'>${p_l_usd:.2f}</span>", unsafe_allow_html=True)
            c5.markdown(f"P/L (%): <span style='color:{color}; font-weight:bold;'>{p_l_pct:.2f}%</span>", unsafe_allow_html=True)
            
            if c6.button(f"❌ ปิดไม้นี้", key=f"btn_close_gold_{idx}"):
                close_specific_virtual_order(SYMBOL_GOLD, side, amt, "Gold")
    else:
        st.info("ไม่มีตั๋วแยกฝั่งทองคำค้างในพอร์ต")

# ----------------------------------------------------
# 3. หน้าพอร์ต BTC Futures
# ----------------------------------------------------
elif page == "🟠 พอร์ต BTC Futures":
    st.title("🟠 Whale Hunter - BTC Futures Terminal")
    st.write("สถานะไม้แยกและการจัดการรายตั๋วเดี่ยว (Timeframe 5m - Pure Mode)")
    
    SYMBOL_BTC = 'BTC/USDT'
    LEVERAGE_BTC = 150
    
    try:
        ticker = exchange.fetch_ticker(SYMBOL_BTC)
        live_price = float(ticker['last'])
    except: live_price = 0.0
    
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("ราคาปัจจุบัน BTC/USDT", f"${live_price:,}")
    col_b2.metric("ทุนรวมในพอร์ต", f"${total_cap} USDT")
    col_b3.metric("ทุนว่างคงเหลือ", f"${avail_cap} USDT")
    
    st.markdown("---")
    df_trades_b = fetch_trades_safe(SYMBOL_BTC)
    df_vt_b = process_virtual_orders(df_trades_b, SYMBOL_BTC)
    
    if not df_vt_b.empty:
        for idx, row in df_vt_b.iterrows():
            entry = row['Entry Price']
            amt = row['Amount']
            side = row['Side']
            
            diff = (live_price - entry) if side == "LONG" else (entry - live_price)
            p_l_usd = diff * amt
            p_l_pct = (diff / entry) * 100 * LEVERAGE_BTC
            color = "green" if p_l_usd >= 0 else "red"
            
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1.5, 1.5, 1.5])
            c1.write(f"**{row['ไม้ที่']}**")
            c2.write(f"ราคาเข้า: `${entry:,}`")
            c3.write(f"ขนาด: `{amt}`")
            c4.markdown(f"P/L ($): <span style='color:{color}; font-weight:bold;'>${p_l_usd:.2f}</span>", unsafe_allow_html=True)
            c5.markdown(f"P/L (%): <span style='color:{color}; font-weight:bold;'>{p_l_pct:.2f}%</span>", unsafe_allow_html=True)
            
            if c6.button(f"❌ ปิดไม้นี้", key=f"btn_close_btc_{idx}"):
                close_specific_virtual_order(SYMBOL_BTC, side, amt, "BTC")
    else:
        st.info("ไม่มีตั๋วแยกฝั่ง BTC ค้างในพอร์ต")

# --- AUTO REFRESH EVERY 10 SECONDS ---
time.sleep(10)
st.rerun()
