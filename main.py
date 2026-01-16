# code red/main.py
import argparse
import sys
from pathlib import Path

# Add 'src' to python path to enable modular imports
sys.path.append(str(Path(__file__).parent / "src"))

import config
# Import the actual scripts we have been using
from src.data import ingest
import run_pipeline  # Your feature/label pipeline
import train_model   # Your XGBoost trainer

def run_task(task):
    """
    Master Controller for the Quant Pipeline.
    """
    print(f"\n=== QUANT V2: Executing Task [{task.upper()}] ===")

    # 1. DATA INGESTION (Download from IBKR)
    if task == 'ingest':
        print("--> Starting Data Ingestion...")
        # Reload config in case you changed symbols
        import importlib
        importlib.reload(config)
        ingest.fetch_data()

    # 2. PIPELINE (Features + Labels)
    elif task == 'pipeline':
        print("--> Running Feature & Label Pipeline...")
        # Loop through universe defined in config
        for sym in config.TARGET_SYMBOLS:
            try:
                run_pipeline.run_full_pipeline(sym)
            except Exception as e:
                print(f"  [!] Pipeline failed for {sym}: {e}")

    # 3. TRAINING (XGBoost Models)
    elif task == 'train':
        print("--> Training Models...")
        results = {}
        for sym in config.TARGET_SYMBOLS:
            try:
                score = train_model.train_xgb_model(sym)
                if score is not None:
                    results[sym] = score
            except Exception as e:
                print(f"  [!] Training failed for {sym}: {e}")
        
        print("\n=== FINAL SCOREBOARD ===")
        for s, p in results.items():
            print(f"{s}: {p:.2%}")

    # 4. RUN EVERYTHING
    elif task == 'all':
        run_task('ingest')
        run_task('pipeline')
        run_task('train')

    else:
        print(f"[!] Error: Unknown task '{task}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Fund Controller")
    
    # We make --task OPTIONAL and default to 'all' so you can just run 'python main.py'
    parser.add_argument(
        '--task', 
        type=str, 
        default='all',
        choices=['ingest', 'pipeline', 'train', 'all'],
        help='Task to run (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        run_task(args.task)
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user.")
    except Exception as e:
        print(f"\n[!] Critical Error: {e}")