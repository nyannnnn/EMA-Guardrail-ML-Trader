# code red/fetch_market.py
from ib_insync import *
import pandas as pd
from src import config

def fetch_qqq():
    print("--> Connecting to IBKR to fetch QQQ (Market Context)...")
    
    ib = IB()
    try:
        # Connect using the same config as your main bot
        ib.connect('127.0.0.1', config.IB_PORT, clientId=999) # ID 999 to avoid conflict
    except Exception as e:
        print(f"[!] Connection failed: {e}")
        print("    Make sure TWS/Gateway is open.")
        return

    # Define the Market Proxy (QQQ = Nasdaq 100 ETF)
    contract = Stock('QQQ', 'SMART', 'USD')
    
    print("--> Requesting 30 days of 1-min QQQ data...")
    
    # We fetch 30 days to match your stock data duration
    bars = ib.reqHistoricalData(
        contract, 
        endDateTime='', 
        durationStr='30 D', 
        barSizeSetting='1 min', 
        whatToShow='TRADES', 
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )
    
    if bars:
        # Convert to DataFrame
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Save to the RAW folder, exactly where backtest.py looks for it
        save_path = config.DATA_RAW / "QQQ_1min.parquet"
        df.to_parquet(save_path)
        
        print(f"  [SUCCESS] Downloaded {len(df)} rows.")
        print(f"  [+] Saved to: {save_path}")
    else:
        print("  [!] ERROR: No data received for QQQ.")

    ib.disconnect()

if __name__ == "__main__":
    fetch_qqq()