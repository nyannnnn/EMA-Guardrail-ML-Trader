# check_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import config

def inspect_data(symbol):
    """
    Diagnostic script to validate Feature and Label integrity.
    Run this BEFORE training any models.
    """
    print(f"--> Inspecting Data for {symbol}...")
    
    # Load the fully labeled dataset
    # Note: Ensure you have run the pipeline to generate this file first!
    file_path = config.DATA_PROCESSED / f"{symbol}_labeled.parquet"
    if not file_path.exists():
        print(f"  [!] Error: File not found {file_path}")
        print("      Run your feature/label pipeline first.")
        return

    df = pd.read_parquet(file_path)
    
    # 1. Class Balance Check
    print("\n--- 1. LABEL DISTRIBUTION ---")
    counts = df['bin'].value_counts()
    ratios = df['bin'].value_counts(normalize=True)
    print(f"  Total Rows: {len(df)}")
    print(f"  Wins (1):   {counts.get(1, 0)} ({ratios.get(1, 0):.2%})")
    print(f"  Loss (0):   {counts.get(0, 0)} ({ratios.get(0, 0):.2%})")
    
    if ratios.get(1, 0) < 0.10:
        print("  [WARNING] Win rate is very low (<10%). Check your Take Profit / Stop Loss width.")

    # 2. Feature Health Check
    print("\n--- 2. FEATURE HEALTH ---")
    # Select only feature columns (starting with 'feat_')
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    
    if not feat_cols:
        print("  [!] No columns found starting with 'feat_'. Check features.py.")
    else:
        print(f"  Checking {len(feat_cols)} features...")
        is_clean = True
        for col in feat_cols:
            n_nans = df[col].isna().sum()
            n_infs = np.isinf(df[col]).sum()
            if n_nans > 0 or n_infs > 0:
                print(f"    [!] {col}: {n_nans} NaNs, {n_infs} Infs")
                is_clean = False
        
        if is_clean:
            print("  [SUCCESS] No NaNs or Infinite values in features.")

    # 3. Visual Verification (The "Eye Test")
    print("\n--- 3. VISUAL CHECK (Single Trade) ---")
    # Find the first 'Win' to plot
    wins = df[df['bin'] == 1]
    if not wins.empty:
        entry_time = wins.index[0]
        exit_time = wins.loc[entry_time, 'exit_time']
        
        print(f"  Plotting Trade: {entry_time} -> {exit_time}")
        print(f"  Return: {wins.loc[entry_time, 'ret']:.4%}")
        
        # Load raw 1-min data for plotting context
        # (Assuming you have the raw file or can slice the processed one)
        price_slice = df.loc[entry_time : exit_time + pd.Timedelta(minutes=30), 'close']
        
        plt.figure(figsize=(10, 5))
        plt.plot(price_slice.index, price_slice.values, label='Price', color='gray')
        plt.scatter(entry_time, df.loc[entry_time, 'close'], color='blue', label='Entry', marker='^', s=100)
        plt.scatter(exit_time, df.loc[exit_time, 'close'], color='green', label='Exit (Win)', marker='x', s=100)
        plt.title(f"Trade Verification: {symbol} (Win)")
        plt.legend()
        plt.show()
    else:
        print("  [!] No wins found to plot.")

if __name__ == "__main__":
    # Change this to a symbol you have processed
    SYMBOL = "AMAT" 
    inspect_data(SYMBOL)