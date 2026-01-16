# quant_v2/src/data/ingest.py
import pandas as pd
from ib_insync import *
from src import config

def fetch_data():
    """
    Connects to IBKR TWS/Gateway and downloads 1-minute historical trade data.
    Saves raw data to Parquet format in the data/raw directory.
    """
    ib = IB()
    try:
        print(f"--> Connecting to IBKR at {config.IB_HOST}:{config.IB_PORT}...")
        ib.connect(config.IB_HOST, config.IB_PORT, clientId=config.CLIENT_ID)
        
        for symbol in config.ALL_SYMBOLS:
            print(f"Fetching {symbol} [{config.DURATION}]...")
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request historical trades (High precision for ML features)
            bars = ib.reqHistoricalData(
                contract, endDateTime='', durationStr=config.DURATION,
                barSizeSetting=config.RAW_INTERVAL, whatToShow=config.WHAT_TO_SHOW,
                useRTH=config.USE_RTH, formatDate=1, keepUpToDate=False
            )
            
            if not bars:
                print(f"  [!] NO DATA for {symbol}")
                continue
                
            df = util.df(bars)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Select only necessary columns to reduce file size
            df = df[['open', 'high', 'low', 'close', 'volume', 'average']]
            
            file_path = config.DATA_RAW / f"{symbol}_1min.parquet"
            df.to_parquet(file_path)
            print(f"  [+] Saved {len(df)} rows to {file_path}")

    except Exception as e:
        print(f"  [!] Error: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == "__main__":
    fetch_data()