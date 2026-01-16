# code red/optimize.py
import pandas as pd
import xgboost as xgb
import numpy as np
from src import config

def analyze_strategy():
    print(f"--> Running Deep Dive Optimization for {config.ACTIVE_TRADING_LIST}...")
    
    # Store all potential trades
    all_signals = []

    for symbol in config.ACTIVE_TRADING_LIST:
        # Load Data
        data_path = config.DATA_PROCESSED / f"{symbol}_labeled.parquet"
        if not data_path.exists(): continue
        df = pd.read_parquet(data_path)
        
        # Load Model
        model_path = config.MODELS_DIR / f"{symbol}_xgb.json"
        if not model_path.exists(): continue
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        
        # Prepare Features
        exclude = ['bin', 'ret', 'exit_time', 'open', 'high', 'low', 'close', 'volume']
        features = [c for c in df.columns if c not in exclude]
        
        # Predict
        dtest = xgb.DMatrix(df[features])
        df['prob'] = bst.predict(dtest)
        
        # Use only Test Data (Last 20%)
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:].copy()
        test_df['symbol'] = symbol
        
        # Store essential info
        all_signals.append(test_df[['symbol', 'prob', 'bin']])

    if not all_signals:
        print("[!] No data found.")
        return

    # Combine into one giant dataframe
    master_df = pd.concat(all_signals)
    master_df['hour'] = master_df.index.hour

    print(f"    Analyzed {len(master_df)} total minutes of test data.\n")

    # --- TEST 1: THRESHOLD OPTIMIZATION ---
    print("=== 1. THRESHOLD SENSITIVITY (How picky should we be?) ===")
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Exp. Return (per trade)':<20}")
    print("-" * 60)
    
    best_thresh = 0.50
    best_score = -999
    
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # Filter trades
        trades = master_df[master_df['prob'] > thresh]
        
        if len(trades) < 5: continue # Ignore empty results
        
        wins = trades[trades['bin'] == 1]
        win_rate = len(wins) / len(trades)
        
        # Approx return: (Win% * 1.0%) - (Loss% * 0.5%)
        # This is the "Expected Value" (EV)
        ev = (win_rate * 1.0) - ((1 - win_rate) * 0.5)
        
        print(f"{thresh:<10.2f} | {len(trades):<8} | {win_rate:<10.2%} | {ev:+.4f}%")
        
        if ev > best_score and len(trades) > 20: # Ensure enough sample size
            best_score = ev
            best_thresh = thresh

    print(f"\n[>>] RECOMMENDED THRESHOLD: {best_thresh} (EV: {best_score:.4f}%)\n")

    # --- TEST 2: TIME OF DAY ANALYSIS ---
    print("=== 2. HOURLY PERFORMANCE (When do we lose money?) ===")
    print(f"Using Threshold: {best_thresh}")
    print(f"{'Hour':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Status':<10}")
    print("-" * 50)
    
    # Filter by the BEST threshold first
    optimized_df = master_df[master_df['prob'] > best_thresh]
    
    # Group by Hour (9, 10, 11, etc.)
    hourly_stats = optimized_df.groupby('hour').agg(
        trades=('bin', 'count'),
        wins=('bin', 'sum')
    )
    hourly_stats['win_rate'] = hourly_stats['wins'] / hourly_stats['trades']
    
    bad_hours = []
    
    for hour, row in hourly_stats.iterrows():
        wr = row['win_rate']
        status = "✅ CLEAN"
        if wr < 0.33: # If Win Rate is below break-even (33% for 2:1 ratio)
            status = "❌ TOXIC"
            bad_hours.append(hour)
        elif wr < 0.40:
            status = "⚠️ WEAK"
            
        print(f"{hour}:00      | {row['trades']:<8} | {wr:<10.2%} | {status}")

    if bad_hours:
        print(f"\n[>>] ACTION: Update paper_trade.py to BLOCK trading during hours: {bad_hours}")
    else:
        print(f"\n[>>] ACTION: No time restrictions needed. All hours are profitable.")

if __name__ == "__main__":
    analyze_strategy()