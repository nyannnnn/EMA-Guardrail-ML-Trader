# code red/train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score
from src import config
import os

def train_xgb_model(symbol):
    print(f"\n--> Training Model for {symbol}...")
    
    # 1. Load Data
    file_path = config.DATA_PROCESSED / f"{symbol}_labeled.parquet"
    if not file_path.exists():
        print(f"  [SKIP] No labeled data for {symbol}")
        return None
        
    df = pd.read_parquet(file_path)
    
    # 2. Setup Features (X) and Target (y)
    drop_cols = ['bin', 'ret', 'exit_time', 'open', 'high', 'low', 'close', 'volume']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['bin']
    
    # 3. Chronological Split (No shuffling!)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 4. Class Imbalance (Scale Weights)
    # If 0s are 80% of data, we weight 1s higher
    n_zeros = (y_train == 0).sum()
    n_ones = (y_train == 1).sum()
    
    if n_ones == 0:
        print(f"  [!] Error: No 'Wins' in training set for {symbol}. Check labeling.")
        return None
        
    scale_pos_weight = n_zeros / n_ones

    # 5. Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds, zero_division=0)
    
    print(f"  [RESULT] {symbol} Test Precision: {precision:.2%}")
    if precision < 0.45:
        print("    -> [WARNING] Precision is low. Model might be guessing.")
    
    # 7. Save
    save_dir = config.PROJECT_ROOT / "models"
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(save_dir / f"{symbol}_xgb.json")
    
    return precision

if __name__ == "__main__":
    print(f"Targeting Universe: {config.TARGET_SYMBOLS}")
    
    results = {}
    for sym in config.TARGET_SYMBOLS:
        score = train_xgb_model(sym)
        if score is not None:
            results[sym] = score
            
    print("\n=== FINAL UNIVERSE SCORES ===")
    for s, p in results.items():
        print(f"{s}: {p:.2%}")