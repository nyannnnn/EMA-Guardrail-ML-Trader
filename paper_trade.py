# code red/paper_trade.py
import datetime
import math
import requests
import json
import csv
import pandas as pd
import xgboost as xgb
import pytz 
import sys
import time
from collections import deque
from ib_insync import *
from src import config
from src.strategy import features

class MLTrader:
    def __init__(self):
        self.ib = IB()
        self.models = {}    
        self.positions = {} 
        self.account_id = "" 
        self.minutes_running = 0 
        self.log_file = config.PROJECT_ROOT / "trade_log.csv"
        self.summary_generated = False 
        
        # RISK & STATE
        self.starting_equity = 0.0
        self.daily_loss_limit = 0.0 
        self.order_labels = {}
        
        # --- NEW FEATURES STATE ---
        self.last_trade_time = {}   # Cooldown Timer
        self.market_is_safe = False # Market Guard (SPY Trend)

        # EVENT LISTENER
        self.ib.execDetailsEvent += self.on_fill

    def log(self, msg):
        tz_ny = pytz.timezone('US/Eastern')
        timestamp = datetime.datetime.now(tz_ny).strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        sys.stdout.flush()

        if "[!]" in msg or "[CRITICAL]" in msg or "[ERROR]" in msg:
            self.send_discord_embed(
                title="⚠️ SYSTEM ALERT",
                description=f"**Error Logged:**\n`{msg}`",
                color=0xe74c3c
            )

    def on_fill(self, trade, fill):
        symbol = fill.contract.symbol
        side = fill.execution.side
        qty = fill.execution.shares
        price = fill.execution.price
        order_id = fill.execution.orderId
        
        # Update Cooldown on Exit/Entry
        self.last_trade_time[symbol] = datetime.datetime.now(pytz.timezone('US/Eastern'))
        
        label = self.order_labels.get(order_id, "Manual/Unknown")
        
        if side == 'BOT':
            title = f"🚀 BOUGHT: {symbol}"
            color = 0x2ecc71 
            desc = f"**Entry Executed.**\nStrategy: {label}"
        else:
            if "Profit" in label:
                title = f"💰 PROFIT TAKEN: {symbol}"
                color = 0xf1c40f 
                desc = "🎯 **Target Hit!** Cash locked in."
            elif "Stop" in label:
                title = f"🛡️ STOPPED OUT: {symbol}"
                color = 0xe67e22 
                desc = "🛑 **Trailing Stop Hit.** Protecting capital."
            else:
                title = f"📉 SOLD: {symbol}"
                color = 0x95a5a6 
                desc = f"Order Type: {label}"

        self.log(f"  [FILL] {title} | {qty} @ ${price:.2f}")
        
        self.send_discord_embed(
            title=title, description=desc, color=color,
            fields=[
                {"name": "Qty", "value": str(qty), "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Total", "value": f"${(qty * price):,.2f}", "inline": True}
            ]
        )

    def connect(self):
        self.log(f"--> Connecting to IBKR (Port {config.IB_PORT})...")
        try:
            if self.ib.isConnected(): self.ib.disconnect()
            
            self.ib.connect('127.0.0.1', config.IB_PORT, clientId=config.CLIENT_ID)
            self.ib.reqMarketDataType(3) 
            
            accounts = self.ib.managedAccounts()
            self.account_id = accounts[0] if accounts else "Unknown"
            
            current_equity = self.get_account_equity()
            
            if self.starting_equity == 0.0:
                self.starting_equity = current_equity
                loss_amount = self.starting_equity * config.MAX_DAILY_LOSS_PCT
                self.daily_loss_limit = -abs(loss_amount)
                self.log(f"  [RISK] Starting Equity: ${self.starting_equity:,.2f}")
                self.log(f"  [RISK] Max Daily Loss ({config.MAX_DAILY_LOSS_PCT:.1%}): ${self.daily_loss_limit:,.2f}")

            self.send_discord_embed(
                title="🟢 System Online",
                description=f"Sniper Mode Active.\nWindow: {config.TRADING_START_HOUR}:00 - {config.TRADING_END_HOUR}:00 ET",
                color=0x3498db, 
                fields=[
                    {"name": "Account", "value": f"`{self.account_id}`", "inline": True},
                    {"name": "Equity", "value": f"${current_equity:,.2f}", "inline": True}
                ]
            )
            self.log(f"  [SUCCESS] Connected to Account: {self.account_id}")
            return True
        except Exception as e:
            self.log(f"  [!] Connection failed: {e}")
            return False

    def update_market_guard(self):
        """
        Checks SPY (Market) AND XLK (Tech Sector) Trends.
        Rule: BOTH must be above their 5-minute EMA-20.
        """
        self.market_is_safe = False # Default to Unsafe
        
        try:
            # Define the "Guardians"
            tickers = ['SPY', 'XLK'] 
            statuses = []
            
            for symbol in tickers:
                contract = Stock(symbol, 'SMART', 'USD')
                # Request 5-minute bars (Institutional Trend)
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime='', durationStr='7200 S', 
                    barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True, timeout=5
                )
                
                if not bars or len(bars) < 20:
                    self.log(f"  [GUARD] ⚠️ Missing Data for {symbol}. Halting Buys.")
                    return # Fail Safe

                df = util.df(bars)
                df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
                
                last_close = df['close'].iloc[-1]
                last_ema = df['ema'].iloc[-1]
                
                # Individual Check
                is_bullish = last_close > last_ema
                statuses.append(is_bullish)
                
                # Optional debug log
                # dist_pct = ((last_close - last_ema) / last_ema) * 100
                # self.log(f"  [DEBUG] {symbol}: {last_close:.2f} vs EMA {last_ema:.2f} ({dist_pct:+.2f}%)")

            # THE DOUBLE LOCK: Both must be True
            if all(statuses):
                self.market_is_safe = True
                if self.minutes_running % 10 == 0:
                    self.log("  [GUARD] ✅ MARKET & SECTOR ALIGNED (SPY+XLK Bullish). Trading Active.")
            else:
                self.market_is_safe = False
                if self.minutes_running % 10 == 0:
                    self.log("  [GUARD] 🛑 SECTOR CONFLICT. Tech (XLK) or Market (SPY) is weak. Halting.")

        except Exception as e:
            self.log(f"  [!] Market Guard Error: {e}")
            self.market_is_safe = False

    def start(self):
        while True:
            try:
                if not self.ib.isConnected():
                    if not self.connect():
                        self.log("  [!] Retry in 10s...")
                        time.sleep(10)
                        continue

                if not self.models: self.load_models()
                self.update_positions()
                self.run_strategy_loop()

            except KeyboardInterrupt:
                self.log("\n  [STOP] Manual Shutdown.")
                self.generate_daily_summary() 
                self.ib.disconnect()
                break
            except Exception as e:
                self.log(f"\n  [CRITICAL CRASH] {e}")
                self.ib.disconnect()
                time.sleep(10)

    def run_strategy_loop(self):
        self.log(f"--> STARTING LIVE TRADING LOOP: {config.ACTIVE_TRADING_LIST}")
        
        while True:
            self.ib.sleep(0.1) 
            
            self.check_circuit_breaker()
            self.update_market_guard() 
            
            if self.minutes_running % 5 == 0: self.update_positions()

            tz_ny = pytz.timezone('US/Eastern')
            now = datetime.datetime.now(tz_ny)
            start_time = now.replace(hour=config.TRADING_START_HOUR, minute=0, second=0, microsecond=0)
            end_time = now.replace(hour=config.TRADING_END_HOUR, minute=0, second=0, microsecond=0)

            if now < start_time:
                wait_seconds = (start_time - now).total_seconds()
                self.log(f"  [WAIT] Market not open. Sleeping {wait_seconds:.0f}s...")
                self.ib.sleep(wait_seconds + 1)
                continue 

            if now >= end_time:
                if not self.summary_generated: self.generate_daily_summary()
                self.log(f"  [WAIT] Market Closed. Sleeping 60s...")
                self.ib.sleep(60)
                continue
            
            self.summary_generated = False

            for symbol in config.ACTIVE_TRADING_LIST:
                # 1. OWNERSHIP CHECK
                if self.positions.get(symbol, False): 
                    # self.log(f"  [SKIP] {symbol} (Already Owned)")
                    continue 

                # 2. MARKET GUARD CHECK
                if not self.market_is_safe:
                    # self.log(f"  [SKIP] {symbol} (Market Red)")
                    continue

                # 3. COOLDOWN CHECK
                if symbol in self.last_trade_time:
                    last_trade = self.last_trade_time[symbol]
                    minutes_since = (now - last_trade).total_seconds() / 60
                    if minutes_since < 30: 
                        # self.log(f"  [SKIP] {symbol} (Cooldown)")
                        continue 

                if symbol not in self.models: continue
                
                # 4. DATA CHECK
                X_live, price = self.get_live_features(symbol)
                if X_live is None or X_live.empty: 
                    self.log(f"  [SKIP] {symbol} (Data Fetch Failed)")
                    continue
                
                # 5. RSI CEILING CHECK (NEW!)
                # Check if the stock is "Overheated"
                if 'feat_rsi_14' in X_live.columns:
                    current_rsi = X_live['feat_rsi_14'].iloc[-1]
                    if current_rsi > 75:
                        self.log(f"  [SKIP] {symbol} is Overbought (RSI: {current_rsi:.1f} > 75)")
                        continue
                
                dtest = xgb.DMatrix(X_live)
                prob = self.models[symbol].predict(dtest)[0]
                
                self.log(f"  {symbol}: {prob:.1%} (Price: ${price:.2f})")
                
                if prob >= config.ENTRY_THRESHOLD:
                    self.execute_trade(symbol, prob, price)
            
            self.log("  ... scanning complete. Sleeping 60s ...")
            self.ib.sleep(60)
            self.minutes_running += 1

    def execute_trade(self, symbol, confidence, price):
        if self.positions.get(symbol, False): return
        entry_price = float(price)
        if entry_price <= 0: return

        equity = self.get_account_equity()
        target_value = equity * config.POSITION_PCT
        qty = math.floor(target_value / entry_price)
        if qty < 1: return

        self.last_trade_time[symbol] = datetime.datetime.now(pytz.timezone('US/Eastern'))

        trail_pct = 0.8  
        parent_limit_price = float(round(entry_price * 1.005, 2))
        lmt_price = float(round(entry_price * (1 + config.PROFIT_TARGET_PCT), 2))
        
        initial_stop_price = entry_price * (1 - (trail_pct/100))
        risk_per_share = entry_price - initial_stop_price
        total_risk = risk_per_share * qty

        self.log(f"  [$$$] SIGNAL FIRED: {symbol} ({confidence:.1%}) -> BUY {qty} @ {entry_price}")

        self.send_discord_embed(
            title=f"🚀 SIGNAL: {symbol}", 
            description="Attempting to Buy...", 
            color=0x2ecc71,
            fields=[
                {"name": "Entry", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "Win Prob", "value": f"**{confidence:.2%}**", "inline": True},
                {"name": "Stop Type", "value": f"Trailing {trail_pct}%", "inline": True},
                {"name": "Risk", "value": f"⚠️ ${total_risk:.2f}", "inline": True}
            ]
        )
        self.log_trade_to_csv(symbol, "BUY", qty, entry_price, lmt_price, initial_stop_price, confidence, equity)

        contract = Stock(symbol, 'SMART', 'USD')
        parent_id = self.ib.client.getReqId()
        take_profit_id = self.ib.client.getReqId()
        stop_loss_id = self.ib.client.getReqId()

        self.order_labels[parent_id] = "Entry (Limit)"
        self.order_labels[take_profit_id] = "Profit Target"
        self.order_labels[stop_loss_id] = "Trailing Stop"

        parent = Order(orderId=parent_id, action='BUY', totalQuantity=qty, orderType='LMT', lmtPrice=parent_limit_price, transmit=False, tif='DAY')
        take_profit = Order(orderId=take_profit_id, action='SELL', totalQuantity=qty, orderType='LMT', lmtPrice=lmt_price, parentId=parent_id, transmit=False, tif='DAY')
        stop_loss = Order(orderId=stop_loss_id, action='SELL', totalQuantity=qty, orderType='TRAIL', trailingPercent=trail_pct, parentId=parent_id, transmit=True, tif='DAY')

        self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, take_profit)
        self.ib.placeOrder(contract, stop_loss)
            
        self.positions[symbol] = True
        self.log(f"  [EXECUTE] ORDERS SENT: Parent #{parent_id} | Trail {trail_pct}%")

    def check_circuit_breaker(self):
        current_equity = self.get_account_equity()
        daily_pnl = current_equity - self.starting_equity
        if self.minutes_running % 10 == 0:
            self.log(f"  [PnL CHECK] Day PnL: ${daily_pnl:,.2f}")
        if daily_pnl < self.daily_loss_limit:
            self.log(f"  [CRITICAL] CIRCUIT BREAKER HIT! PnL: ${daily_pnl:,.2f}")
            self.send_discord_embed(title="🛑 CIRCUIT BREAKER", description="Daily Loss Limit Hit.", color=0xe74c3c)
            self.generate_daily_summary()
            sys.exit("Circuit Breaker Hit.")

    def update_positions(self):
        try:
            current_positions = self.ib.positions()
            self.positions = {}
            for pos in current_positions:
                if pos.position > 0: self.positions[pos.contract.symbol] = True
        except: pass

    def load_models(self):
        self.log("--> Loading Brains...")
        for symbol in config.ACTIVE_TRADING_LIST:
            model_path = config.MODELS_DIR / f"{symbol}_xgb.json"
            if model_path.exists():
                bst = xgb.Booster()
                bst.load_model(str(model_path))
                self.models[symbol] = bst
                self.log(f"  [+] Loaded Model: {symbol}")

    def get_live_features(self, symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        try:
            bars = self.ib.reqHistoricalData(contract, endDateTime='', durationStr='2 D', barSizeSetting='1 min', whatToShow='TRADES', useRTH=True, timeout=10)
            if not bars: return None, 0.0
            df = util.df(bars)
            df.columns = df.columns.str.lower()
            current_price = df['close'].iloc[-1]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df_features = features.add_technical_features(df)
            latest_row = df_features.iloc[[-1]].copy()
            expected_cols = ['average', 'vwap', 'feat_dist_vwap', 'log_ret', 'feat_vol_15m', 'feat_vol_impact', 'feat_rsi_14', 'feat_spread_proxy']
            if any(c not in latest_row.columns for c in expected_cols): return None, 0.0
            return latest_row[expected_cols], current_price
        except: return None, 0.0

    def generate_daily_summary(self):
        if self.summary_generated: return
        self.log("--> Generating End-of-Day PnL Report...")
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        summary_dir = config.PROJECT_ROOT / "daily_summary"
        summary_dir.mkdir(parents=True, exist_ok=True) 
        filename = summary_dir / f"{today_str}_trade_summary.txt"
        
        try:
            exec_filter = ExecutionFilter() 
            fills = self.ib.reqExecutions(exec_filter)
            tz_est = pytz.timezone('US/Eastern')
            today_fills_data = []

            for f in fills:
                t_utc = f.execution.time.replace(tzinfo=datetime.timezone.utc)
                t_est = t_utc.astimezone(tz_est)
                if t_est.strftime('%Y%m%d') == datetime.datetime.now(tz_est).strftime('%Y%m%d'):
                    today_fills_data.append((f, t_est))
            today_fills_data.sort(key=lambda x: x[1])

            portfolio = {}
            for item in today_fills_data:
                fill = item[0]
                sym = fill.contract.symbol
                side = fill.execution.side
                qty = float(fill.execution.shares)
                price = float(fill.execution.price)

                if sym not in portfolio: portfolio[sym] = {'realized_pnl': 0.0, 'inventory': deque()}

                if side == 'BOT':
                    portfolio[sym]['inventory'].append((qty, price))
                elif side == 'SLD':
                    shares_to_sell = qty
                    while shares_to_sell > 0 and portfolio[sym]['inventory']:
                        batch_qty, batch_price = portfolio[sym]['inventory'][0]
                        if batch_qty <= shares_to_sell:
                            profit = (price - batch_price) * batch_qty
                            portfolio[sym]['realized_pnl'] += profit
                            shares_to_sell -= batch_qty
                            portfolio[sym]['inventory'].popleft() 
                        else:
                            profit = (price - batch_price) * shares_to_sell
                            portfolio[sym]['realized_pnl'] += profit
                            remaining = batch_qty - shares_to_sell
                            portfolio[sym]['inventory'][0] = (remaining, batch_price)
                            shares_to_sell = 0

            equity = self.get_account_equity()
            total_realized_pnl = 0.0
            
            with open(filename, "w") as f:
                f.write(f"=== TRADING SUMMARY: {today_str} ===\n")
                f.write(f"Account: {self.account_id}\n")
                f.write(f"Ending Equity: ${equity:,.2f}\n")
                f.write("-" * 65 + "\n")
                f.write(f"{'SYM':<6} {'REALIZED PnL':<15} {'OPEN POS':<10}\n")
                f.write("-" * 65 + "\n")
                
                for sym, data in portfolio.items():
                    pnl = data['realized_pnl']
                    total_realized_pnl += pnl
                    
                    pnl_str = f"${pnl:,.2f}"
                    open_qty = sum(item[0] for item in data['inventory'])
                    open_str = f"{open_qty:.0f} sh" if open_qty > 0 else "-"
                    f.write(f"{sym:<6} | {pnl_str:<15} | {open_str:<10}\n")

                f.write("-" * 65 + "\n")
                f.write(f"TOTAL REALIZED DAY P&L: ${total_realized_pnl:,.2f}\n")
                f.write("-" * 65 + "\n")
                f.write("DETAILED EXECUTION LOG (EST):\n")
                f.write(f"{'TIME':<12} {'SYMBOL':<6} {'SIDE':<5} {'QTY':<5} {'PRICE'}\n")
                
                for item in today_fills_data:
                    fill = item[0]
                    t_est = item[1]
                    t_str = t_est.strftime('%H:%M:%S')
                    f.write(f"{t_str:<12} {fill.contract.symbol:<6} {fill.execution.side:<5} {fill.execution.shares:<5} ${fill.execution.price:.2f}\n")
            
            self.log(f"  [REPORT] Saved to {filename}")
            self.send_discord_embed(title="🏁 Day Complete", description=f"**Realized P&L: ${total_realized_pnl:,.2f}**\nReport: `{filename.name}`", color=0x2ecc71 if total_realized_pnl > 0 else 0xe74c3c)
            self.summary_generated = True

        except Exception as e:
            self.log(f"  [!] Report Generation Failed: {e}")

    def send_discord_embed(self, title, description, color, fields=None):
        if not config.DISCORD_WEBHOOK_URL: return
        embed = {"title": title, "description": description, "color": color, "timestamp": datetime.datetime.utcnow().isoformat(), "footer": {"text": "ML Trader | Quant V2"}}
        if fields: embed["fields"] = fields
        try: requests.post(config.DISCORD_WEBHOOK_URL, json={"embeds": [embed]})
        except: pass

    def log_trade_to_csv(self, symbol, action, qty, entry, target, stop, confidence, equity):
        file_exists = self.log_file.exists()
        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists: writer.writerow(["Timestamp", "Symbol", "Action", "Qty", "Entry", "Target", "Stop", "Confidence", "Equity"])
                writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, action, qty, f"{entry:.2f}", f"{target:.2f}", f"{stop:.2f}", f"{confidence:.4f}", f"{equity:.2f}"])
        except Exception: pass

    def get_account_equity(self):
        try:
            summary = self.ib.accountSummary(self.account_id)
            net_liq = next((v.value for v in summary if v.tag == 'NetLiquidation'), None)
            return float(net_liq) if net_liq else config.FALLBACK_EQUITY
        except: return config.FALLBACK_EQUITY

if __name__ == "__main__":
    bot = MLTrader()
    bot.start()