# code red/run_pipeline.py
import pandas as pd
from src import config
from src.strategy import features, labeling

def run_full_pipeline(symbol):
    print(f"--> Starting Pipeline for {symbol}...")

    # --- 1. Load Data ---
    raw_path_parquet = config.DATA_RAW / f"{symbol}_1min.parquet"
    processed_path = config.DATA_PROCESSED / f"{symbol}_1min.parquet"
    
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
    elif raw_path_parquet.exists():
        df = pd.read_parquet(raw_path_parquet)
        df.columns = df.columns.str.lower()
    else:
        print(f"  [SKIP] No data found for {symbol}")
        return

    # --- 2. Features ---
    try:
        df_features = features.add_technical_features(df)
    except Exception as e:
        print(f"  [!] Features failed for {symbol}: {e}")
        return

    # --- 3. Labels ---
    # Risk: 1.0% Profit, 0.5% Stop
    risk_params = [0.005, 0.010] 
    try:
        df_labels = labeling.get_triple_barrier_labels(
            prices=df_features['close'],
            events=df_features.index,
            sl_tp_limits=risk_params,
            vertical_barrier_bars=12
        )
    except Exception as e:
        print(f"  [!] Labeling failed for {symbol}: {e}")
        return

    # --- 4. Save ---
    df_final = df_features.join(df_labels[['bin', 'ret', 'exit_time']], how='inner')
    save_path = config.DATA_PROCESSED / f"{symbol}_labeled.parquet"
    df_final.to_parquet(save_path)
    print(f"  [SUCCESS] {symbol} Ready. Rows: {len(df_final)}")

if __name__ == "__main__":
    # Loop through the entire universe defined in config.py
    for sym in config.TARGET_SYMBOLS:
        run_full_pipeline(sym)