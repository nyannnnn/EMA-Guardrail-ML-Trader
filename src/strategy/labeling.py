# quant_v2/src/strategy/labeling.py
import numpy as np
import pandas as pd
from src import config

def get_triple_barrier_labels(prices, events, sl_tp_limits, vertical_barrier_bars=12):
    """
    Implements the Triple Barrier Method (Meta-Labeling).
    Determines if a trade hits Profit, Stop Loss, or Time Stop first.
    """
    # Vertical Barrier (Time Stop)
    # Find the timestamp N bars in the future for every event
    t1 = prices.index.searchsorted(events + pd.Timedelta(minutes=5 * vertical_barrier_bars))
    t1 = t1[t1 < prices.shape[0]] # Ensure index is valid
    t1 = pd.Series(prices.index[t1], index=events[:len(t1)])

    # Initialize output container
    out = pd.DataFrame(index=events)
    out['bin'] = np.nan        # 1 = Profit, 0 = Loss/Time
    out['ret'] = np.nan        # Realized return
    out['exit_time'] = pd.NaT  # Timestamp of exit

    stop_loss = sl_tp_limits[0]
    take_profit = sl_tp_limits[1]

    # Iterate Through Events
    # Loop required to handle path dependency (SL vs TP order)
    for loc, entry_time in enumerate(events):
        if entry_time not in prices.index:
            continue

        entry_price = prices.loc[entry_time]
        
        # Determine max hold time (Vertical Barrier)
        end_time = t1.loc[entry_time] if entry_time in t1.index else prices.index[-1]

        # Slice future price path (Entry+1 to Time Stop)
        path = prices.loc[entry_time:end_time].iloc[1:]

        if path.empty:
            continue

        # Calculate returns relative to entry
        path_rets = (path / entry_price) - 1

        # Check Barriers
        # Did we hit PT?
        first_touch_pt = path_rets[path_rets >= take_profit].index.min()
        # Did we hit SL?
        first_touch_sl = path_rets[path_rets <= -stop_loss].index.min()

        # Determine First Touch
        exit_time = end_time   # Default: Time Stop
        outcome = 0            # Default: Loss/Neutral

        if pd.notna(first_touch_pt) and pd.notna(first_touch_sl):
            if first_touch_pt < first_touch_sl:
                exit_time = first_touch_pt
                outcome = 1 # Win
            else:
                exit_time = first_touch_sl
                outcome = 0 # Loss (SL hit first)
        elif pd.notna(first_touch_pt):
            exit_time = first_touch_pt
            outcome = 1 # Win
        elif pd.notna(first_touch_sl):
            exit_time = first_touch_sl
            outcome = 0 # Loss

        # Record Result
        realized_ret = (prices.loc[exit_time] / entry_price) - 1
        
        out.loc[entry_time, 'bin'] = int(outcome)
        out.loc[entry_time, 'ret'] = realized_ret
        out.loc[entry_time, 'exit_time'] = exit_time

    return out.dropna()

def run_labeling_on_symbol(target_symbol, barrier_conf):
    """
    Wrapper to load data, apply labeling, and save results.
    """
    print(f"--> Running Triple Barrier Labeling on {target_symbol}...")
    
    # Load processed data
    file_path = config.DATA_PROCESSED / f"{target_symbol}_features.parquet"
    if not file_path.exists():
        print(f"  [!] Error: File not found {file_path}")
        return None
        
    df = pd.read_parquet(file_path)
    
    # Run Labeling
    labels = get_triple_barrier_labels(
        prices=df['close'],
        events=df.index,
        sl_tp_limits=barrier_conf,
        vertical_barrier_bars=12
    )
    
    # Merge Labels back
    df = df.join(labels[['bin', 'ret', 'exit_time']], how='left').dropna()
    
    # Save
    save_path = config.DATA_PROCESSED / f"{target_symbol}_labeled.parquet"
    df.to_parquet(save_path)
    print(f"  [SUCCESS] Saved Labeled Data: {save_path}")
    
    return df