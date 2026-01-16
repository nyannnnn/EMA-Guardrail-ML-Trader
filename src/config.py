# quant_v2/src/config.py
import os
from pathlib import Path

# --- DIRECTORIES ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- IBKR CONNECTION ---
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497 = Paper TWS, 4002 = Paper Gateway
CLIENT_ID = 101

# --- UNIVERSE ---
# The Hedge (Benchmark)
HEDGE_SYMBOL = 'SMH'

# The Targets (Semi Basket)
TARGET_SYMBOLS = [
    'MU', 'WDC', 'STX', 'PSTG', 'SMCI', 'NTAP'
]

ACTIVE_TRADING_LIST = ['PSTG', 'WDC', 'STX']

# Market Features (For ML Context only, NOT traded)
MARKET_SYMBOLS = ['SPY', 'QQQ', 'SMH']

# Combined list for downloader
ALL_SYMBOLS = list(set(TARGET_SYMBOLS + [HEDGE_SYMBOL] + MARKET_SYMBOLS))

# --- SETTINGS ---
RAW_INTERVAL = '1 min'   # Download resolution (Keep high res for ML)
RESAMPLE_INTERVAL = '15T' # Trading resolution (15 mins) for Kalman Filter
DURATION = '30 D'        # How much history to fetch
WHAT_TO_SHOW = 'TRADES'
USE_RTH = True           # Regular Trading Hours only
MODELS_DIR = PROJECT_ROOT / "models"
ENTRY_THRESHOLD = 0.55 # Kalman entry threshold
POSITION_PCT = 0.10 # 10% of portfolio per trade
FALLBACK_EQUITY = 200000.0  # Used if no broker connection
MAX_DAILY_LOSS_PCT = 0.03 # 3% max daily drawdown
TRAILING_STOP_PCT = 0.8 # 0.4% trailing stop
PROFIT_TARGET_PCT = 0.05 # 5% profit target
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1449887948521734276/xfDVr5-EGqqfv4nHTzMSHN4RhCIwgBMHYviXfG_oy0sBMagatn4bNUYtuBN9N_4hvCJG"  # Optional: For trade alerts
TRADING_START_HOUR = 10
TRADING_END_HOUR = 16