# Reinforcement Learning Trading

PPO-based reinforcement learning agents for trading EURUSD and Equinix.

## Structure

- `src/` - Source code
  - `trading_env.py` - Trading environment
  - `train_agent.py` - Training script
  - `test_agent.py` - Testing script
  - `indicators.py` - Technical indicators
- `data/` - Historical trading data
- `checkpoints/` - Training checkpoints (saved every 50k steps)
- `model_*_best.zip` - Best performing models

## Usage

### Training
```bash
python src/train_agent.py
```

### Testing
```bash
python src/test_agent.py
```

## Data

- `EURUSD_Candlestick_1_Hour_BID_*.csv` - EUR/USD forex data
- `Equinix.csv` - Equinix stock data
- `Residential_Equity.csv` - Residential equity data

## Model Checkpoints

Checkpoints are saved every 50,000 steps from 50k to 600k steps for both:
- PPO EURUSD agent
- PPO Equinix agent
