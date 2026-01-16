# code red/backtest.py
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from src import config

def run_backtest():
    print(f"--> Starting Backtest with MARKET REGIME FILTER (QQQ)...")
    print(f"    Initial Capital: ${config.FALLBACK_EQUITY:,.2f}")
    
    # 1. Load Market Data (QQQ)
    qqq_path = config.DATA_RAW / "QQQ_1min.parquet"
    if not qqq_path.exists():
        print("[!] QQQ data missing. Run 'fetch_market.py' first.")
        return
        
    print("    Loading QQQ Data...", end="")
    df_qqq = pd.read_parquet(qqq_path)
    
    # Calculate Regime Filter: Is Price > 20-min Moving Average?
    # We use 'close' column.
    df_qqq['qqq_ma'] = df_qqq['close'].rolling(window=20).mean()
    df_qqq['market_safe'] = df_qqq['close'] > df_qqq['qqq_ma']
    
    # Keep only the safety flag and index (time) for merging
    df_qqq = df_qqq[['market_safe']]
    print(" Done.")

    all_trades = []
    
    # 2. Process Stocks
    for symbol in config.ACTIVE_TRADING_LIST:
        print(f"\n  [Processing {symbol}]...")
        
        # Load Stock Data
        data_path = config.DATA_PROCESSED / f"{symbol}_labeled.parquet"
        if not data_path.exists(): continue
        df = pd.read_parquet(data_path)
        
        # --- MERGE WITH QQQ ---
        # Join on the Timestamp Index
        # 'inner' join ensures we only look at times where we have BOTH stock and QQQ data
        df = df.join(df_qqq, how='inner')
        
        # Load Model
        model_path = config.MODELS_DIR / f"{symbol}_xgb.json"
        if not model_path.exists(): continue
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        
        # Prepare Features (exclude non-feature cols + our new 'market_safe' col)
        exclude = ['bin', 'ret', 'exit_time', 'open', 'high', 'low', 'close', 'volume', 'market_safe']
        features = [c for c in df.columns if c not in exclude]
        
        # Predict
        dmatrix = xgb.DMatrix(df[features])
        df['prob'] = bst.predict(dmatrix)
        
        # --- SIMULATION (Test Set Only) ---
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:].copy()
        
        # FILTER LOGIC:
        # 1. High Confidence (Prob > Threshold)
        # 2. Safe Market (QQQ > MA20)
        signals = test_df[
            (test_df['prob'] > config.ENTRY_THRESHOLD) & 
            (test_df['market_safe'] == True)  # <--- THE GUARD
        ].copy()
        
        print(f"    -> Raw Signals: {len(test_df[test_df['prob'] > config.ENTRY_THRESHOLD])}")
        print(f"    -> Safe Signals: {len(signals)} (Filtered out risky trades)")
        
        for t, row in signals.iterrows():
            pnl = 0.01 if row['bin'] == 1 else -0.005
            all_trades.append({
                'time': t, 'symbol': symbol, 'pnl_pct': pnl
            })

    # 3. Calculate Results
    if not all_trades:
        print("[!] No trades passed the filter.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values('time')
    equity = config.FALLBACK_EQUITY
    
    wins = 0
    losses = 0
    
    for _, trade in trades_df.iterrows():
        pos_size = equity * config.POSITION_PCT
        pnl_amt = (pos_size * trade['pnl_pct']) - 2.0 # $2 comms
        equity += pnl_amt
        
        if trade['pnl_pct'] > 0: wins += 1
        else: losses += 1

    total_return = ((equity - config.FALLBACK_EQUITY) / config.FALLBACK_EQUITY) * 100
    win_rate = (wins / (wins + losses)) * 100
    
    print("\n=== REGIME FILTERED RESULTS (QQQ Guard) ===")
    print(f"  Final Equity:   ${equity:,.2f}")
    print(f"  Total Return:   {total_return:.2f}%")
    print(f"  Total Trades:   {wins + losses}")
    print(f"  Win Rate:       {win_rate:.2f}%")

if __name__ == "__main__":
    run_backtest()