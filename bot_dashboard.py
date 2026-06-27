import os
import time
import ccxt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- CONFIGURATION & PAGE INITIALIZATION ---
st.set_page_config(page_title="Whale Hunter Hybrid Terminal", layout="wide")

# สไตล์การตกแต่งเพิ่มเติมให้ UI สวยงามน่าใช้ยิ่งขึ้น
st.markdown("""
    <style>
    .reportview-container { background: #111216; }
    .stMetric { background-color: #1e2026; padding: 15px; border-radius: 10px; border: 1px solid #2b2f3a; }
    div.stButton > button:first-child { background-color: #ff4b4b; color: white; border-radius: 8px; width: 100%; font-weight: bold; }
    div.stButton > button:hover { background-color: #d32f2f; color: white; }
    .position-card { background-color: #1e2026; padding: 18px; border-radius: 12px; margin-bottom: 12px; border-left: 5px solid #2b2f3a; }
    </style>
""", unsafe_allow_html=True)

load_dotenv()
BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

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
        st.success(f"❌ สั่งปิดตั๋ว {label} ฝั่ง {side} ขนาด {amt:.4f} สำเร็จ!")
        time.sleep(1)
        st.rerun()
    except Exception as e: st.error(f"ปิดตั๋วไม่สำเร็จ: {e}")

def fire_manual_order(symbol, side, entry_price, margin, lev, tp_pct, sl_pct):
    try:
        contract_amount = round((margin * lev) / entry_price, 4)
        tp_factor = tp_pct / 100
        sl_factor = sl_pct / 100
        
        decimal_place = 1 if 'BTC' in symbol else 2
        
        if side == "LONG":
            tp_price = round(entry_price * (1 + tp_factor), decimal_place)
            sl_price = round(entry_price * (1 - sl_factor), decimal_place)
            order_side = 'buy'
        else:
            tp_price = round(entry_price * (1 - tp_factor), decimal_place)
            sl_price = round(entry_price * (1 + sl_factor), decimal_place)
            order_side = 'sell'

        exchange.create_order(symbol=symbol, type='market', side=order_side, amount=contract_amount, params={'positionSide': side})
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

# --- NAVIGATION SIDEBAR ---
st.sidebar.title("🐳 WHALE HUNTER")
st.sidebar.markdown("⚙️ **ระบบเมนูไฮบริด v8.9**")
page = st.sidebar.radio("เลือกหน้าแดชบอร์ด:", ["🏠 หน้าแรก Overview", "🏆 พอร์ตทองคำ / NCCO", "🟠 พอร์ต BTC Futures"])

try:
    bal = exchange.fetch_balance()
    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
    avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
except:
    total_cap, avail_cap = 0.0, 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ แผงสั่งยิงออเดอร์ด้วยมือ")
manual_asset = st.sidebar.selectbox("เลือกเหรียญ:", ["GOLD/NCCO", "BTC/USDT"])

if manual_asset == "GOLD/NCCO":
    m_symbol = 'NCCOGOLD2USD/USDT:USDT'
    default_lev = 250
    default_mgn = 0.02
    default_tp = 0.50
    default_sl = 0.30
else:
    m_symbol = 'BTC/USDT'
    default_lev = 150
    default_mgn = 0.5
    default_tp = 2.0
    default_sl = 2.5

try:
    m_ticker = exchange.fetch_ticker(m_symbol)
    m_live_price = float(m_ticker['last'])
except:
    m_live_price = 0.0

m_mgn = st.sidebar.number_input("เงินทุน (Margin $)", min_value=0.01, value=default_mgn, step=0.01, format="%.2f")
m_lev = st.sidebar.number_input("Leverage (X)", min_value=1, max_value=250, value=default_lev, step=1)
m_tp = st.sidebar.number_input("เป้ากำไร TP (%)", min_value=0.05, value=default_tp, step=0.05, format="%.2f")
m_sl = st.sidebar.number_input("ตัดขาดทุน SL (%)", min_value=0.05, value=default_sl, step=0.05, format="%.2f")

if m_live_price > 0:
    calc_amt = round((m_mgn * m_lev) / m_live_price, 4)
    st.sidebar.info(f"📊 สัญญาที่จะส่งออก: {calc_amt}")
else:
    st.sidebar.caption("⚠️ รอราคาตลาดคำนวณสัญญา...")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🟢 ยิง LONG", use_container_width=True):
    if m_live_price > 0: fire_manual_order(m_symbol, "LONG", m_live_price, m_mgn, m_lev, m_tp, m_sl)

if col_btn2.button("🔴 ยิง SHORT", use_container_width=True):
    if m_live_price > 0: fire_manual_order(m_symbol, "SHORT", m_live_price, m_mgn, m_lev, m_tp, m_sl)


# ----------------------------------------------------
# 1. หน้าแรก (Overview)
# ----------------------------------------------------
if page == "🏠 หน้าแรก Overview":
    st.title("🐳 Whale Hunter V8.9 - Hybrid Terminal")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    col1.metric("💰 ทุนสุทธิในพอร์ตรวมทั้งหมด", f"${total_cap:,} USDT")
    col2.metric("💵 ทุนว่างพร้อมใช้งาน", f"${avail_cap:,} USDT")
    
    st.markdown("<br>### 📊 สถานะการทำงานของบอทหลังบ้าน", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #1e2026; padding: 20px; border-radius: 10px; border-left: 6px solid #4caf50; margin-bottom: 15px;'>
        <h4 style='margin:0; color:#4caf50;'>🏆 พอร์ตทองคำ / NCCO (เฝ้าระบบ 1m)</h4>
        <p style='margin:5px 0 0 0; color:#aaa;'>สถานะ: ออนไลน์ปกติ ครอบคลุมลูปเช็กสัญญาณ MFI Divergence ตลอด 24 ชั่วโมง</p>
    </div>
    <div style='background-color: #1e2026; padding: 20px; border-radius: 10px; border-left: 6px solid #ff9800; margin-bottom: 15px;'>
        <h4 style='margin:0; color:#ff9800;'>🟠 พอร์ต BTC Futures (Pure Mode 5m)</h4>
        <p style='margin:5px 0 0 0; color:#aaa;'>สถานะ: ออนไลน์ปกติ วิ่งโหมดเพียวสไนเปอร์ ไม่ใช้ฟิลเตอร์กรองสัญญาณ เพื่อความไวในการเข้าไม้</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 2. หน้าพอร์ตทองคำ
# ----------------------------------------------------
elif page == "🏆 พอร์ตทองคำ / NCCO":
    st.title("🏆 Gold & NCCO Live Monitor")
    
    SYMBOL_GOLD = 'NCCOGOLD2USD/USDT:USDT'
    LEVERAGE_GOLD = 250
    
    try:
        ticker = exchange.fetch_ticker(SYMBOL_GOLD)
        live_price = float(ticker['last'])
    except: live_price = 0.0
    
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("ราคาปัจจุบัน ทองคำ", f"${live_price:,}")
    col_g2.metric("ทุนในพอร์ต", f"${total_cap:,} USDT")
    col_g3.metric("ทุนว่างคงเหลือ", f"${avail_cap:,} USDT")
    
    st.markdown("<br><b>📋 ตั๋วค้างแยกรายไม้เดี่ยว (Virtual Positions)</b>", unsafe_allow_html=True)
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
            
            color = "#4caf50" if p_l_usd >= 0 else "#f44336"
            border_side_color = "#4caf50" if side == "LONG" else "#f44336"
            
            st.markdown(f"""
            <div style='background-color: #1e2026; padding: 15px; border-radius: 10px; border-left: 5px solid {border_side_color}; margin-bottom: 10px;'>
                <table style='width:100%; border:none; border-collapse:collapse;'>
                    <tr>
                        <td style='width:15%; font-weight:bold; color:{border_side_color};'>{row['ไม้ที่']}</td>
                        <td style='width:20%; color:#fff;'>ราคาเข้า: <b>${entry:,}</b></td>
                        <td style='width:15%; color:#fff;'>ขนาด: <b>{amt:.4f}</b></td>
                        <td style='width:20%; color:#fff;'>P/L ($): <span style='color:{color}; font-weight:bold;'>${p_l_usd:.2f}</span></td>
                        <td style='width:15%; color:#fff;'>P/L (%): <span style='color:{color}; font-weight:bold;'>{p_l_pct:.2f}%</span></td>
                        <td style='width:15%; text-align:right;' id='btn_cell_{idx}'></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # วางปุ่มปิดไม้เดี่ยวให้เยื้องไปทางขวาสุดอย่างสวยงาม
            col_space, col_act = st.columns([5, 1])
            if col_act.button(f"❌ ปิดไม้นี้", key=f"btn_close_gold_{idx}"):
                close_specific_virtual_order(SYMBOL_GOLD, side, amt, "Gold")
    else:
        st.info("ไม่มีตั๋วฝั่งทองคำค้างในพอร์ต")

# ----------------------------------------------------
# 3. หน้าพอร์ต BTC
# ----------------------------------------------------
elif page == "🟠 พอร์ต BTC Futures":
    st.title("🟠 BTC Futures Live Monitor")
    
    SYMBOL_BTC = 'BTC/USDT'
    LEVERAGE_BTC = 150
    
    try:
        ticker = exchange.fetch_ticker(SYMBOL_BTC)
        live_price = float(ticker['last'])
    except: live_price = 0.0
    
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("ราคาปัจจุบัน BTC", f"${live_price:,}")
    col_b2.metric("ทุนในพอร์ต", f"${total_cap:,} USDT")
    col_b3.metric("ทุนว่างคงเหลือ", f"${avail_cap:,} USDT")
    
    st.markdown("<br><b>📋 ตั๋วค้างแยกรายไม้เดี่ยว (Virtual Positions)</b>", unsafe_allow_html=True)
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
            
            color = "#4caf50" if p_l_usd >= 0 else "#f44336"
            border_side_color = "#4caf50" if side == "LONG" else "#f44336"
            
            st.markdown(f"""
            <div style='background-color: #1e2026; padding: 15px; border-radius: 10px; border-left: 5px solid {border_side_color}; margin-bottom: 10px;'>
                <table style='width:100%; border:none; border-collapse:collapse;'>
                    <tr>
                        <td style='width:15%; font-weight:bold; color:{border_side_color};'>{row['ไม้ที่']}</td>
                        <td style='width:20%; color:#fff;'>ราคาเข้า: <b>${entry:,}</b></td>
                        <td style='width:15%; color:#fff;'>ขนาด: <b>{amt:.4f}</b></td>
                        <td style='width:20%; color:#fff;'>P/L ($): <span style='color:{color}; font-weight:bold;'>${p_l_usd:.2f}</span></td>
                        <td style='width:15%; color:#fff;'>P/L (%): <span style='color:{color}; font-weight:bold;'>{p_l_pct:.2f}%</span></td>
                        <td style='width:15%; text-align:right;'></td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            col_space, col_act = st.columns([5, 1])
            if col_act.button(f"❌ ปิดไม้นี้", key=f"btn_close_btc_{idx}"):
                close_specific_virtual_order(SYMBOL_BTC, side, amt, "BTC")
    else:
        st.info("ไม่มีตั๋วฝั่ง BTC ค้างในพอร์ต")

# --- AUTO REFRESH EVERY 10 SECONDS ---
time.sleep(10)
st.rerun()
