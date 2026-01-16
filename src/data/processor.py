# quant_v2/src/data/process.py
import pandas as pd
from src import config

def resample_and_align():
    """
    Resamples raw 1-minute data to the 15-minute strategy frequency and 
    performs an inner join to align Target assets with the Hedge asset.
    """
    print("--> Starting Data Processing (Resample & Align)...")
    
    hedge_path = config.DATA_RAW / f"{config.HEDGE_SYMBOL}_1min.parquet"
    if not hedge_path.exists():
        print(f"CRITICAL: Hedge file {hedge_path} not found.")
        return

    df_hedge_1m = pd.read_parquet(hedge_path)
    
    # Resample Hedge asset to strategy frequency
    df_hedge_15m = df_hedge_1m.resample(config.RESAMPLE_INTERVAL, label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    for symbol in config.TARGET_SYMBOLS:
        target_path = config.DATA_RAW / f"{symbol}_1min.parquet"
        if not target_path.exists():
            continue
            
        print(f"Processing {symbol} vs {config.HEDGE_SYMBOL}...")
        
        df_target_1m = pd.read_parquet(target_path)
        
        # Resample Target asset to strategy frequency
        df_target_15m = df_target_1m.resample(config.RESAMPLE_INTERVAL, label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        # Inner Join to align timestamps and remove data gaps
        df_aligned = df_target_15m.join(
            df_hedge_15m, 
            how='inner', 
            lsuffix='_Y', 
            rsuffix='_X'
        )
        
        save_path = config.DATA_PROCESSED / f"{symbol}_{config.HEDGE_SYMBOL}_15m.parquet"
        df_aligned.to_parquet(save_path)
        print(f"  [+] Saved Aligned Data: {save_path}")

if __name__ == "__main__":
    resample_and_align()