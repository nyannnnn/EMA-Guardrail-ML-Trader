# quant_v2/src/strategy/kalman.py
import numpy as np
import pandas as pd
from src import config
import matplotlib.pyplot as plt

class KalmanFilterReg:
    """
    Online Kalman Filter for Dynamic Linear Regression.
    Updates regression coefficients (alpha, beta) on every new observation.
    """
    def __init__(self, delta=1e-5, R=1e-3):
        self.delta = delta # Process noise (adaptability)
        self.R = R         # Measurement noise (sensitivity)
        self.n_states = 2 
        self.state_mean = np.zeros(self.n_states)
        self.state_mean[1] = 1.0 # Initialize beta to 1.0
        self.P = np.eye(self.n_states)
        self.wt = self.delta / (1 - self.delta) * np.eye(self.n_states)

    def update(self, y, x):
        # Time Update (Prediction)
        self.P = self.P + self.wt
        
        # Measurement Update (Correction)
        H = np.array([1.0, x])
        y_pred = H.dot(self.state_mean)
        error = y - y_pred
        
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T) / S
        
        self.state_mean = self.state_mean + K * error
        self.P = (np.eye(self.n_states) - np.outer(K, H)).dot(self.P)
        
        return y_pred, error, self.state_mean[0], self.state_mean[1]

def run_kalman_on_pair(target_symbol):
    """
    Generates Phase 2 signals (Beta, Z-Score) for a single pair 
    and saves the enriched dataset for Phase 3.
    """
    print(f"--> Running Kalman Filter on {target_symbol} vs {config.HEDGE_SYMBOL}...")
    
    file_path = config.DATA_PROCESSED / f"{target_symbol}_{config.HEDGE_SYMBOL}_15m.parquet"
    if not file_path.exists():
        print(f"  [!] Error: File not found {file_path}")
        return None
    
    df = pd.read_parquet(file_path)
    kf = KalmanFilterReg(delta=1e-4, R=1e-3)
    
    # Vectorize inputs for speed
    obs_y = df['close_Y'].values
    obs_x = df['close_X'].values
    
    preds, errors, alphas, betas = [], [], [], []
    
    for i in range(len(df)):
        y_pred, error, alpha, beta = kf.update(obs_y[i], obs_x[i])
        preds.append(y_pred)
        errors.append(error)
        alphas.append(alpha)
        betas.append(beta)
        
    df['model_price'] = preds
    df['spread'] = errors
    df['alpha'] = alphas
    df['beta'] = betas
    
    # Calculate Z-Score on a rolling window to normalize volatility
    window = 30 
    df['z_score'] = (df['spread'] - df['spread'].rolling(window).mean()) / df['spread'].rolling(window).std()
    
    df.dropna(inplace=True)
    
    # Save Phase 2 Output
    save_path = config.DATA_PROCESSED / f"{target_symbol}_kalman.parquet"
    df.to_parquet(save_path)
    print(f"  [SUCCESS] Saved Raw Signals: {save_path}")
    
    return df