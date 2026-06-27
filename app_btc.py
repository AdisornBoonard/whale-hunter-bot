import os
import time
import ccxt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Whale Hunter BTC Terminal", layout="wide")
load_dotenv()

BINGX_API_KEY = os.getenv("BINGX_API_KEY")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

exchange = ccxt.bingx({
    'apiKey': BINGX_API_KEY,
    'secret': BINGX_SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

SYMBOL = 'BTC/USDT'
LEVERAGE = 150

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

def process_virtual_orders_from_cache(df_trades):
    virtual_tickets = []
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

def close_specific_virtual_order(side, amt):
    try:
        order_side = 'sell' if side == 'LONG' else 'buy'
        exchange.create_order(symbol=SYMBOL, type='market', side=order_side, amount=amt, params={'positionSide': side})
        st.success(f"❌ สั่งปิดตั๋ว BTC ฝั่ง {side} ขนาด {amt} สำเร็จ!")
        time.sleep(1)
        st.rerun()
    except Exception as e: st.error(f"ปิดตั๋วไม่สำเร็จ: {e}")

# --- UI DISPLAY ---
st.title("🟠 Whale Hunter V8.9 - BTC Futures Terminal")
st.write("แดชบอร์ดติดตามสถานะไม้แยกเฉพาะ BTC/USDT (Timeframe 5m)")

try:
    ticker = exchange.fetch_ticker(SYMBOL)
    live_price = float(ticker['last'])
    bal = exchange.fetch_balance()
    total_cap = round(float(bal.get('USDT', {}).get('total', 0.0)), 2)
    avail_cap = round(float(bal.get('USDT', {}).get('free', 0.0)), 2)
except Exception as e:
    st.error(f"เชื่อมต่อพอร์ตผิดพลาด: {e}")
    live_price, total_cap, avail_cap = 0.0, 0.0, 0.0

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("ราคาปัจจุบัน BTC/USDT", f"${live_price:,}")
col_info2.metric("ทุนสุทธิในพอร์ต", f"${total_cap} USDT")
col_info3.metric("ทุนว่างคงเหลือ", f"${avail_cap} USDT")

st.markdown("---")
st.subheader("📋 รายการตั๋วแยกรายไม้ (Virtual Positions ของ BTC)")

df_trades = fetch_trades_safe()
df_vt = process_virtual_orders_from_cache(df_trades)

if not df_vt.empty:
    for idx, row in df_vt.iterrows():
        entry = row['Entry Price']
        amt = row['Amount']
        side = row['Side']
        
        if side == "LONG":
            diff = live_price - entry
            p_l_usd = diff * amt
            p_l_pct = (diff / entry) * 100 * LEVERAGE
        else:
            diff = entry - live_price
            p_l_usd = diff * amt
            p_l_pct = (diff / entry) * 100 * LEVERAGE

        color = "green" if p_l_usd >= 0 else "red"
        
        col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1.5, 1.5, 1.5])
        col1.write(f"**{row['ไม้ที่']}**")
        col2.write(f"ราคาเข้า: `${entry:,}`")
        col3.write(f"ขนาด: `{amt}`")
        col4.markdown(f"P/L ($): <span style='color:{color}; font-weight:bold;'>${p_l_usd:.2f}</span>", unsafe_allow_html=True)
        col5.markdown(f"P/L (%): <span style='color:{color}; font-weight:bold;'>{p_l_pct:.2f}%</span>", unsafe_allow_html=True)
        
        if col6.button(f"❌ ปิดไม้นี้", key=f"btn_close_btc_{idx}"):
            close_specific_virtual_order(side, amt)
else:
    st.info("ไม่มีตั๋วแยก BTC ค้างในระบบ")

time.sleep(10)
st.rerun()
