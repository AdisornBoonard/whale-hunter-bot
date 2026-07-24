//@version=5
indicator("High-Probability Confluence + Distance Filter Dashboard (3 Conditions)",
     overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)

// ============================================================================
//  SHARED INPUTS  (เหมือนกันทั้ง 3 เงื่อนไข ตามต้นฉบับที่ส่งมา)
// ============================================================================
useRSI = input.bool(true, "Use RSI Divergence Filter", group="Filters (Shared)")
rsiLen = input.int(9, "RSI Length", group="RSI Settings (Shared)")
rsiOB  = input.int(65, "RSI Overbought", group="RSI Settings (Shared)")
rsiOS  = input.int(5, "RSI Oversold", group="RSI Settings (Shared)")

min_dist_pct = input.float(2.0, "ระยะห่างขั้นต่ำจากไม้ออกล่าสุด (%)", group="Entry Filter (Shared)") / 100

use_auto_tp_sl = input.bool(false, "ใช้ TP/SL อัตโนมัติจาก Order Block (ติ๊ก = Auto, ไม่ติ๊ก = Fix)", group="TP/SL Settings (Shared)")
fix_tp_pct     = input.float(5.5, "Fix Take Profit (%)", group="TP/SL Settings (Shared)") / 100
fix_sl_pct     = input.float(3.0, "Fix Stop Loss (%)", group="TP/SL Settings (Shared)") / 100

initial_cap     = input.float(10.0, "ทุนเริ่มต้นของพอร์ต (USDT)", group="Risk & Capital (Shared)")
base_order_usdt = input.float(1.0, "ขนาดเงินต่อไม้ (USDT / Base Order)", group="Risk & Capital (Shared)")
leverage        = input.int(25, "Leverage", group="Risk & Capital (Shared)")
fee_pct         = input.float(0.04, "ค่าธรรมเนียม / Fee (%)", group="Risk & Capital (Shared)") / 100

// ---- แต่ละเงื่อนไข: จุดต่างเดียวที่พบระหว่าง 3 ไฟล์ คือตัวกรองเทรนด์ EMA ----
c1_useEMA = input.bool(false, "ใช้ EMA Trend Filter", group="เงื่อนไข 1")
c1_emaLen = input.int(150, "EMA Length", group="เงื่อนไข 1")

c2_useEMA = input.bool(true, "ใช้ EMA Trend Filter", group="เงื่อนไข 2")
c2_emaLen = input.int(150, "EMA Length", group="เงื่อนไข 2")

c3_useEMA = input.bool(true, "ใช้ EMA Trend Filter", group="เงื่อนไข 3")
c3_emaLen = input.int(100, "EMA Length", group="เงื่อนไข 3")

// ============================================================================
//  SHARED: MARKET STRUCTURE (Order Block) + RSI DIVERGENCE
//  (คำนวณครั้งเดียว ใช้ร่วมกันทั้ง 3 เงื่อนไข เพราะต้นฉบับเหมือนกันทุกตัว)
// ============================================================================
len = 5
ph = ta.pivothigh(high, len, len)
pl = ta.pivotlow(low, len, len)

var float lastBullishOB = na
var float lastBearishOB = na
if pl
    lastBullishOB := low[len]
if ph
    lastBearishOB := high[len]

nearBullOB = not na(lastBullishOB) and math.abs(close - lastBullishOB) / lastBullishOB < 0.015
nearBearOB = not na(lastBearishOB) and math.abs(close - lastBearishOB) / lastBearishOB < 0.015

rsi = ta.rsi(close, rsiLen)
bullishDiv = ta.crossover(rsi, rsiOS) and not useRSI
bearishDiv = ta.crossunder(rsi, rsiOB) and not useRSI

if useRSI
    priceLL = low < ta.valuewhen(pl, low[len], 1)
    rsiHL   = rsi > ta.valuewhen(pl, rsi[len], 1)
    bullishDiv := priceLL and rsiHL and (rsi < 45)

    priceHH = high > ta.valuewhen(ph, high[len], 1)
    rsiLH   = rsi < ta.valuewhen(ph, high[len], 1)
    bearishDiv := priceHH and rsiLH and (rsi > 55)

// ============================================================================
//  EMA TREND FILTER ต่อเงื่อนไข
// ============================================================================
c1_ema = ta.ema(close, c1_emaLen)
c2_ema = ta.ema(close, c2_emaLen)
c3_ema = ta.ema(close, c3_emaLen)

c1_uptrend   = not c1_useEMA or (close > c1_ema)
c1_downtrend = not c1_useEMA or (close < c1_ema)
c2_uptrend   = not c2_useEMA or (close > c2_ema)
c2_downtrend = not c2_useEMA or (close < c2_ema)
c3_uptrend   = not c3_useEMA or (close > c3_ema)
c3_downtrend = not c3_useEMA or (close < c3_ema)

plot(c1_useEMA ? c1_ema : na, "EMA เงื่อนไข 1", color=color.new(color.orange, 0),  linewidth=1)
plot(c2_useEMA ? c2_ema : na, "EMA เงื่อนไข 2", color=color.new(color.aqua, 0),    linewidth=1)
plot(c3_useEMA ? c3_ema : na, "EMA เงื่อนไข 3", color=color.new(color.fuchsia, 0), linewidth=1)

// ============================================================================
//  RAW SIGNAL ต่อเงื่อนไข (ใช้ RSI/OB ร่วมกัน กรองด้วยเทรนด์ของตัวเอง)
// ============================================================================
c1_raw_long  = c1_uptrend   and nearBullOB and (bullishDiv or rsi < rsiOS)
c1_raw_short = c1_downtrend and nearBearOB and (bearishDiv or rsi > rsiOB)
c2_raw_long  = c2_uptrend   and nearBullOB and (bullishDiv or rsi < rsiOS)
c2_raw_short = c2_downtrend and nearBearOB and (bearishDiv or rsi > rsiOB)
c3_raw_long  = c3_uptrend   and nearBullOB and (bullishDiv or rsi < rsiOS)
c3_raw_short = c3_downtrend and nearBearOB and (bearishDiv or rsi > rsiOB)

// ============================================================================
//  ENGINE ที่ใช้ซ้ำ 3 ครั้ง — แต่ละจุดที่เรียกจะได้ state (var) เป็นของตัวเอง
//  แยกขาดจากกัน: ไม่เปิดซ้ำในเงื่อนไขตัวเอง แต่ไม่บล็อกเงื่อนไขอื่น
// ============================================================================
runCondition(raw_long, raw_short) =>
    var float entry_price = na
    var bool  in_position = false
    var int   pos_type    = 0     // 1 = Long, -1 = Short, 0 = Flat
    var float current_tp  = na
    var float current_sl  = na
    var float last_long_entry_price  = na
    var float last_short_entry_price = na

    var int   total_trades   = 0
    var int   winning_trades = 0
    var int   losing_trades  = 0
    var float total_win_amt  = 0.0
    var float total_loss_amt = 0.0
    var float net_profit     = 0.0

    if in_position
        notional_val = base_order_usdt * leverage
        trade_fee    = notional_val * fee_pct * 2

        if pos_type == 1
            if high >= current_tp
                total_trades   += 1
                winning_trades += 1
                tp_rate    = use_auto_tp_sl ? ((current_tp - entry_price) / entry_price) : fix_tp_pct
                actual_win = (notional_val * tp_rate) - trade_fee
                total_win_amt += actual_win
                net_profit    += actual_win
                in_position   := false
            else if low <= current_sl
                total_trades  += 1
                losing_trades += 1
                sl_rate     = use_auto_tp_sl ? ((entry_price - current_sl) / entry_price) : fix_sl_pct
                actual_loss = (notional_val * sl_rate) + trade_fee
                total_loss_amt += actual_loss
                net_profit     -= actual_loss
                in_position    := false
        else if pos_type == -1
            if low <= current_tp
                total_trades   += 1
                winning_trades += 1
                tp_rate    = use_auto_tp_sl ? ((entry_price - current_tp) / entry_price) : fix_tp_pct
                actual_win = (notional_val * tp_rate) - trade_fee
                total_win_amt += actual_win
                net_profit    += actual_win
                in_position   := false
            else if high >= current_sl
                total_trades  += 1
                losing_trades += 1
                sl_rate     = use_auto_tp_sl ? ((current_sl - entry_price) / entry_price) : fix_sl_pct
                actual_loss = (notional_val * sl_rate) + trade_fee
                total_loss_amt += actual_loss
                net_profit     -= actual_loss
                in_position    := false

    can_long  = na(last_long_entry_price)  or (math.abs(close - last_long_entry_price)  / last_long_entry_price  >= min_dist_pct)
    can_short = na(last_short_entry_price) or (math.abs(close - last_short_entry_price) / last_short_entry_price >= min_dist_pct)

    // สำคัญ: "not in_position" กันไม่ให้เงื่อนไขนี้เปิดไม้ซ้ำตัวเอง (แต่ไม่กระทบเงื่อนไขอื่น เพราะ state แยกกัน)
    long_signal  = raw_long  and can_long  and not in_position
    short_signal = raw_short and can_short and not in_position

    if long_signal
        entry_price := close
        sl_calc = use_auto_tp_sl ? (not na(lastBullishOB) ? lastBullishOB * 0.995 : close * (1 - 0.01)) : close * (1 - fix_sl_pct)
        tp_calc = use_auto_tp_sl ? close + (math.abs(close - sl_calc) * 1.618) : close * (1 + fix_tp_pct)
        current_tp := tp_calc
        current_sl := sl_calc
        pos_type   := 1
        in_position:= true
        last_long_entry_price := close
    else if short_signal
        entry_price := close
        sl_calc = use_auto_tp_sl ? (not na(lastBearishOB) ? lastBearishOB * 1.005 : close * (1 + 0.01)) : close * (1 + fix_sl_pct)
        tp_calc = use_auto_tp_sl ? close - (math.abs(sl_calc - close) * 1.618) : close * (1 - fix_tp_pct)
        current_tp := tp_calc
        current_sl := sl_calc
        pos_type   := -1
        in_position:= true
        last_short_entry_price := close

    win_rate = total_trades > 0 ? (winning_trades / total_trades) * 100.0 : 0.0
    [long_signal, short_signal, in_position, current_tp, current_sl,
     total_trades, winning_trades, losing_trades, total_win_amt, total_loss_amt, net_profit, win_rate]

[c1_long, c1_short, c1_inpos, c1_tp, c1_sl, c1_tot, c1_win, c1_loss, c1_winamt, c1_lossamt, c1_net, c1_winrate] = runCondition(c1_raw_long, c1_raw_short)
[c2_long, c2_short, c2_inpos, c2_tp, c2_sl, c2_tot, c2_win, c2_loss, c2_winamt, c2_lossamt, c2_net, c2_winrate] = runCondition(c2_raw_long, c2_raw_short)
[c3_long, c3_short, c3_inpos, c3_tp, c3_sl, c3_tot, c3_win, c3_loss, c3_winamt, c3_lossamt, c3_net, c3_winrate] = runCondition(c3_raw_long, c3_raw_short)

// ============================================================================
//  PLOTS ON CHART  (แยกสี/ป้ายต่อเงื่อนไข ไม่ให้งงว่าใครยิง)
// ============================================================================
plotshape(c1_long,  title="C1 Buy",  style=shape.labelup,   location=location.belowbar, color=color.new(color.green, 0), text="B1", textcolor=color.white, size=size.tiny)
plotshape(c1_short, title="C1 Sell", style=shape.labeldown, location=location.abovebar, color=color.new(color.red, 0),   text="S1", textcolor=color.white, size=size.tiny)
plotshape(c2_long,  title="C2 Buy",  style=shape.labelup,   location=location.belowbar, color=color.new(color.aqua, 0),  text="B2", textcolor=color.black, size=size.tiny)
plotshape(c2_short, title="C2 Sell", style=shape.labeldown, location=location.abovebar, color=color.new(color.aqua, 0),  text="S2", textcolor=color.black, size=size.tiny)
plotshape(c3_long,  title="C3 Buy",  style=shape.labelup,   location=location.belowbar, color=color.new(color.fuchsia,0),text="B3", textcolor=color.white, size=size.tiny)
plotshape(c3_short, title="C3 Sell", style=shape.labeldown, location=location.abovebar, color=color.new(color.fuchsia,0),text="S3", textcolor=color.white, size=size.tiny)

plot(c1_inpos ? c1_tp : na, "C1 TP", style=plot.style_linebr, color=color.new(color.green, 0),   linewidth=1)
plot(c1_inpos ? c1_sl : na, "C1 SL", style=plot.style_linebr, color=color.new(color.red, 0),     linewidth=1)
plot(c2_inpos ? c2_tp : na, "C2 TP", style=plot.style_linebr, color=color.new(color.aqua, 0),     linewidth=1)
plot(c2_inpos ? c2_sl : na, "C2 SL", style=plot.style_linebr, color=color.new(color.aqua, 60),    linewidth=1)
plot(c3_inpos ? c3_tp : na, "C3 TP", style=plot.style_linebr, color=color.new(color.fuchsia, 0),  linewidth=1)
plot(c3_inpos ? c3_sl : na, "C3 SL", style=plot.style_linebr, color=color.new(color.fuchsia, 60), linewidth=1)

// ============================================================================
//  DASHBOARD  (แยกรายเงื่อนไข + สรุปรวมพอร์ต — ใช้ทุนก้อนเดียวกัน)
// ============================================================================
combined_trades = c1_tot + c2_tot + c3_tot
combined_win    = c1_win + c2_win + c3_win
combined_loss   = c1_loss + c2_loss + c3_loss
combined_winrate = combined_trades > 0 ? (combined_win / combined_trades) * 100.0 : 0.0
combined_net    = c1_net + c2_net + c3_net
combined_balance = initial_cap + combined_net

var table dash = table.new(position=position.top_right, columns=4, rows=10,
     bgcolor=color.new(color.black, 20), border_color=color.gray, border_width=1)

if barstate.islast
    table.cell(dash, 0, 0, "DASHBOARD", text_color=color.white, text_size=size.small, bgcolor=color.navy)
    table.cell(dash, 1, 0, "เงื่อนไข 1", text_color=color.white, text_size=size.small, bgcolor=color.navy)
    table.cell(dash, 2, 0, "เงื่อนไข 2", text_color=color.white, text_size=size.small, bgcolor=color.navy)
    table.cell(dash, 3, 0, "เงื่อนไข 3", text_color=color.white, text_size=size.small, bgcolor=color.navy)

    table.cell(dash, 0, 1, "สถานะ", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 1, c1_inpos ? "🟢 เปิดอยู่" : "⚪ ว่าง", text_color=c1_inpos ? color.orange : color.gray, text_size=size.small)
    table.cell(dash, 2, 1, c2_inpos ? "🟢 เปิดอยู่" : "⚪ ว่าง", text_color=c2_inpos ? color.orange : color.gray, text_size=size.small)
    table.cell(dash, 3, 1, c3_inpos ? "🟢 เปิดอยู่" : "⚪ ว่าง", text_color=c3_inpos ? color.orange : color.gray, text_size=size.small)

    table.cell(dash, 0, 2, "Total Orders", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 2, str.tostring(c1_tot), text_color=color.yellow, text_size=size.small)
    table.cell(dash, 2, 2, str.tostring(c2_tot), text_color=color.yellow, text_size=size.small)
    table.cell(dash, 3, 2, str.tostring(c3_tot), text_color=color.yellow, text_size=size.small)

    table.cell(dash, 0, 3, "Win / Loss", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 3, str.tostring(c1_win) + " / " + str.tostring(c1_loss), text_color=color.green, text_size=size.small)
    table.cell(dash, 2, 3, str.tostring(c2_win) + " / " + str.tostring(c2_loss), text_color=color.green, text_size=size.small)
    table.cell(dash, 3, 3, str.tostring(c3_win) + " / " + str.tostring(c3_loss), text_color=color.green, text_size=size.small)

    table.cell(dash, 0, 4, "Win Rate", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 4, str.tostring(c1_winrate, "#.##") + "%", text_color=color.aqua, text_size=size.small)
    table.cell(dash, 2, 4, str.tostring(c2_winrate, "#.##") + "%", text_color=color.aqua, text_size=size.small)
    table.cell(dash, 3, 4, str.tostring(c3_winrate, "#.##") + "%", text_color=color.aqua, text_size=size.small)

    table.cell(dash, 0, 5, "Net Profit", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 5, str.tostring(c1_net, "#.##"), text_color=c1_net>=0?color.green:color.red, text_size=size.small)
    table.cell(dash, 2, 5, str.tostring(c2_net, "#.##"), text_color=c2_net>=0?color.green:color.red, text_size=size.small)
    table.cell(dash, 3, 5, str.tostring(c3_net, "#.##"), text_color=c3_net>=0?color.green:color.red, text_size=size.small)

    table.cell(dash, 0, 6, "EMA Filter", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 6, c1_useEMA ? ("EMA"+str.tostring(c1_emaLen)) : "ปิด", text_color=color.orange, text_size=size.small)
    table.cell(dash, 2, 6, c2_useEMA ? ("EMA"+str.tostring(c2_emaLen)) : "ปิด", text_color=color.orange, text_size=size.small)
    table.cell(dash, 3, 6, c3_useEMA ? ("EMA"+str.tostring(c3_emaLen)) : "ปิด", text_color=color.orange, text_size=size.small)

    table.cell(dash, 0, 7, "── รวมทั้งพอร์ต ──", text_color=color.white, text_size=size.small, bgcolor=color.new(color.blue, 60))
    table.cell(dash, 1, 7, "", bgcolor=color.new(color.blue, 60))
    table.cell(dash, 2, 7, "", bgcolor=color.new(color.blue, 60))
    table.cell(dash, 3, 7, "", bgcolor=color.new(color.blue, 60))

    table.cell(dash, 0, 8, "Total / Win-Loss / WR", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 8, str.tostring(combined_trades) + " | " + str.tostring(combined_win) + "-" + str.tostring(combined_loss) + " | " + str.tostring(combined_winrate, "#.##") + "%", text_color=color.aqua, text_size=size.small)
    table.merge_cells(dash, 1, 8, 3, 8)

    table.cell(dash, 0, 9, "Balance (รวม)", text_color=color.white, text_size=size.small)
    table.cell(dash, 1, 9, str.tostring(combined_balance, "#.##") + " USDT (Net " + str.tostring(combined_net, "#.##") + ")", text_color=color.white, text_size=size.small, bgcolor=color.new(color.blue, 70))
    table.merge_cells(dash, 1, 9, 3, 9)
