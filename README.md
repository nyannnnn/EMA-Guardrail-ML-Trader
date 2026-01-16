# EMA Guardrail ML Trader — Intraday ML Paper Trader (IBKR)

Intraday **long-only** paper trading bot that scans a watchlist, scores entries with **XGBoost**, and executes **risk-managed bracket orders** through **Interactive Brokers (IBKR)** using `ib_insync`.

Built as a practical end-to-end system: **features → model inference → execution → logs → daily summaries**.

---

## Highlights

- **IBKR execution (ib_insync)** with automatic reconnect + event-based fill handling
- **Per-symbol XGBoost models** (`models/<TICKER>_xgb.json`)
- **Live feature generation** from 1-min bars (RTH)
- **Market Guard**: only trade when **SPY + XLK** are both above their **5-min EMA(20)**
- **Cooldown**: per-symbol cooldown after any fill (prevents rapid re-entry)
- **Overbought filter**: skips entries when RSI(14) is too high (default > 75)
- **Bracket-style execution**
  - parent **limit buy**
  - **profit target** limit sell
  - **trailing stop** (protects downside)
- **Circuit breaker**: stops trading if daily P&L breaches max loss limit
- **Discord alerts** for fills + critical errors + end-of-day summary
- **Trade logging** to `trade_log.csv`
- **Daily report** generated to `daily_summary/YYYY-MM-DD_trade_summary.txt`

---

## Strategy Overview (What it does)

Every minute, during the configured trading window (ET), the bot:

1. Ensures IBKR is connected and models are loaded
2. Runs a **Market Guard** check (SPY + XLK trend filter)
3. For each symbol in the active list:
   - skips if already owned
   - skips if market guard is red
   - skips if symbol is in cooldown
   - pulls recent 1-min bars (RTH) and builds technical features
   - runs XGBoost inference → `prob_up`
   - if `prob_up >= ENTRY_THRESHOLD`, sends a bracket order

Position sizing is simple and controlled:
- `target_position_value = equity * POSITION_PCT`
- `qty = floor(target_position_value / price)`

---

## Risk Controls

Configured in `src/config.py` (names below reflect usage in the trading script):

- **Max daily loss**: `MAX_DAILY_LOSS_PCT` computed off `starting_equity`
- **Circuit breaker**: exits if `current_equity - starting_equity < daily_loss_limit`
- **Trailing stop**: `TRAIL` order with `trailingPercent` (default in code: 0.8%)
- **Profit target**: `PROFIT_TARGET_PCT`
- **Trade gating**:
  - market/sector trend filter (SPY + XLK > EMA20 on 5-min)
  - RSI ceiling (skip if RSI too high)
  - cooldown timer after fills

> Note: This is a paper trading system. Real-money deployment needs stricter safeguards, slippage modeling, and better order/connection fault tolerance.

