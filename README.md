# Market Simulation with AI

This repository contains two main projects exploring quantitative finance and market microstructure through simulation and machine learning.

---

## 🏘️ Market_Sim: Housing Liquidity Study

**Research Question:** To what extent does blockchain-based tokenization improve liquidity, price discovery, and market efficiency in commercial real estate, and how do these effects differ from those observed in traditional CRE markets?

This project uses **empirical REIT data** (VNQ, 2005-2026) to model how tokenized housing markets might behave compared to traditional illiquid housing markets. The simulation employs Markov regime-switching models with microstructure dynamics.

### Key Finding

**Tokenized housing reduces crisis time by 47%**
- Traditional housing: 46.9% of time in volatile/panic states
- Tokenized housing: 24.9% of time in stressed states
- Mechanism: Faster price discovery and better liquidity prevent coordination failures

### What It Does

1. **Analyzes Real REIT Data** - Fetches VNQ (Vanguard Real Estate ETF) data and classifies market regimes based on volatility
2. **Extracts Transition Matrices** - Derives empirical probabilities for regime transitions (Calm → Neutral → Volatile → Panic)
3. **Runs Comparative Simulations** - Projects 991 months of traditional vs. tokenized housing markets
4. **Generates Insights** - Shows tokenized markets spend 88% of time in calm/neutral states vs. 53% for traditional housing

### Quick Start

**One-command run:**
```bash
cd Market_Sim/Market_sim
python3 scripts/run_complete_analysis.py
```

This will:
- Fetch real REIT data
- Run both market simulations
- Generate all visualizations
- Print comprehensive report

**Manual step-by-step:**
```bash
# Step 1: Analyze REIT regime dynamics
python3 scripts/analyze_reit_regimes.py

# Step 2: Run housing market comparison
python3 scripts/housing_liquidity_comparison.py

# Step 3: Analyze residential equity
python3 scripts/residential_equity_regimes.py
```

### Key Results

| Metric | Traditional | Tokenized | Improvement |
|--------|-------------|-----------|-------------|
| Time in Stress | 46.9% | 24.9% | **-47%** |
| Panic Episodes | 36 | 8 | **-78%** |
| Time in Calm/Neutral | 53.1% | 88.1% | **+66%** |
| Avg Panic Duration | 3.0 months | 6.5 months | Longer but rarer |

### Core Components

- **sim/** - Market simulation engine
  - `market_simulator.py` - Core market simulator with order book
  - `regimes.py` - Markov regime switching logic
  - `microstructure.py` - Order book and matching engine
  - `calibration.py` - Parameter estimation from historical data
  - `portfolio.py` - Portfolio optimization (Merton model)
  - `agents/` - Trading agent implementations

- **scripts/** - Analysis scripts
  - `run_complete_analysis.py` - Complete pipeline (recommended)
  - `analyze_reit_regimes.py` - REIT data analysis
  - `housing_liquidity_comparison.py` - Main simulation
  - `residential_equity_regimes.py` - Residential market analysis

- **outputs/** - Generated results
  - `VNQ_regime_analysis.png` - REIT regime visualization
  - `housing_liquidity_comparison.png` - Main comparison chart
  - `FINDINGS_SUMMARY.md` - Detailed research findings
  - `README_LIQUIDITY_STUDY.md` - Methodology documentation

### Technical Details

- **Data**: US housing price index (1953-2023, 871 months)
- **Regimes**: 4-state Markov chain (Calm, Neutral, Volatile, Panic)
- **Microstructure**: Order book simulation with continuous trading
- **Calibration**: Empirical transition matrices from VNQ ETF

---

## 🤖 RL_Trading: Reinforcement Learning for Trading

PPO (Proximal Policy Optimization) agents trained to trade forex (EUR/USD) and equities (Equinix) using technical indicators.

### What It Does

Trains reinforcement learning agents to make trading decisions based on:
- Price action (OHLC candlesticks)
- Technical indicators (RSI, MACD, moving averages, Bollinger Bands)
- Historical patterns

The agents learn optimal entry/exit strategies through reward-based learning.

### Quick Start

**Train a new agent:**
```bash
cd RL_Trading
pip install -r requirements.txt
python src/train_agent.py
```

**Test trained models:**
```bash
python src/test_agent.py
```

### Assets Trained

1. **EUR/USD Forex**
   - 1-hour candlesticks (July 2020 - July 2023)
   - Best model: `model_eurusd_best.zip`
   - 24 checkpoints (50k to 600k steps)

2. **Equinix (EQIX)**
   - Stock data with technical indicators
   - Best model: `model_equinix_best.zip`
   - 24 checkpoints (50k to 600k steps)

3. **Residential Equity**
   - Housing market data for regime analysis

### Components

- **src/** - Source code
  - `trading_env.py` - Custom Gym environment for trading
  - `train_agent.py` - PPO training loop
  - `test_agent.py` - Model evaluation
  - `indicators.py` - Technical indicator calculations

- **data/** - Historical market data (CSV files)
- **checkpoints/** - Training snapshots every 50k steps
- **model_*_best.zip** - Best performing models

### Technical Stack

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Environment**: Custom OpenAI Gym
- **Features**: Price + volume + 10+ technical indicators

---

## 📊 Notebooks

Jupyter notebooks for exploratory analysis and prototyping:
- `markov_chains (1).ipynb` - Markov chain analysis and visualization

---

## Project Structure

```
Market_Sim(wAI)/
├── Market_Sim/              # Housing liquidity research project
│   └── Market_sim/
│       ├── main.py              # Main entry point for simulations
│       ├── QUICK_START.txt      # Quick start guide with key findings
│       │
│       ├── scripts/             # Analysis scripts
│       │   ├── run_complete_analysis.py      # Complete pipeline (recommended)
│       │   ├── analyze_reit_regimes.py       # REIT data analysis & regime classification
│       │   ├── housing_liquidity_comparison.py   # Traditional vs tokenized simulation
│       │   └── residential_equity_regimes.py     # Residential market regime analysis
│       │
│       ├── outputs/             # Generated results and visualizations
│       │   ├── VNQ_regime_analysis.png          # REIT regime dynamics chart
│       │   ├── housing_liquidity_comparison.png # Main comparison visualization
│       │   ├── residential_equity_regimes.png   # Residential equity regimes
│       │   ├── cre_monthly.csv                  # CRE historical data
│       │   ├── FINDINGS_SUMMARY.md              # Detailed research findings
│       │   ├── README_LIQUIDITY_STUDY.md        # Methodology documentation
│       │   └── COMPLETION_SUMMARY.md            # Project completion overview
│       │
│       └── sim/                 # Core simulation engine
│           ├── __init__.py
│           ├── market_simulator.py    # Main market simulator with order book
│           ├── regimes.py             # Markov regime switching logic
│           ├── microstructure.py      # Order book & matching engine
│           ├── calibration.py         # Parameter estimation from history
│           ├── portfolio.py           # Portfolio optimization (Merton)
│           ├── data_loader.py         # CSV data loading utilities
│           ├── types.py               # Type definitions & dataclasses
│           ├── utils.py               # Utility functions
│           └── agents/                # Trading agent implementations
│               ├── __init__.py
│               └── rule_based.py      # Rule-based trading agents
│
├── RL_Trading/              # Reinforcement learning trading project
│   ├── src/                     # Source code
│   │   ├── trading_env.py       # Custom Gym environment for trading
│   │   ├── train_agent.py       # PPO training script
│   │   ├── test_agent.py        # Model evaluation script
│   │   └── indicators.py        # Technical indicator calculations
│   │
│   ├── data/                    # Historical market data
│   │   ├── EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv
│   │   ├── test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv
│   │   ├── Equinix.csv
│   │   └── Residential_Equity.csv
│   │
│   ├── checkpoints/             # Training checkpoints (every 50k steps)
│   │   ├── ppo_eurusd_*.zip     # EUR/USD checkpoints (50k-600k)
│   │   └── ppo_equinix_*.zip    # Equinix checkpoints (50k-600k)
│   │
│   ├── model_eurusd_best.zip    # Best EUR/USD model
│   ├── model_equinix_best.zip   # Best Equinix model
│   ├── trade_history_output.csv # Trading history logs
│   ├── requirements.txt         # Python dependencies for RL
│   └── README.md                # RL Trading documentation
│
├── notebooks/               # Jupyter notebooks for analysis
│   └── markov_chains (1).ipynb  # Markov chain analysis & visualization
│
├── requirements.txt         # Main project dependencies
├── .gitignore              # Git ignore rules (Python, macOS, models)
├── README.md               # This file
└── ORGANIZATION_SUMMARY.md # Documentation of project reorganization
```

---

## Installation

### Market_Sim Requirements
```bash
pip install numpy pandas matplotlib requests
```

### RL_Trading Requirements
```bash
cd RL_Trading
pip install -r requirements.txt
```

---

## Research Context

### Market_Sim Findings

The housing liquidity study demonstrates that **increased market liquidity through tokenization could dramatically reduce market stress**. The key insight is that liquid markets avoid getting stuck in crisis states through better price discovery and coordination.

**Implications for policy:**
- Housing tokenization could reduce systemic stress
- Faster crisis resolution through continuous trading
- Improved democratic access via fractional ownership
- Trade-off: May import systemic shock sensitivity

### RL_Trading Applications

The reinforcement learning models explore how AI agents learn to:
- Identify trading patterns in different asset classes
- Manage risk through position sizing
- Adapt to changing market conditions

**Use cases:**
- Algorithmic trading strategy development
- Market behavior analysis
- Pattern recognition research

---

## Documentation

- **Market_Sim/Market_sim/QUICK_START.txt** - Quick start guide
- **Market_Sim/Market_sim/outputs/FINDINGS_SUMMARY.md** - Complete research findings
- **Market_Sim/Market_sim/outputs/README_LIQUIDITY_STUDY.md** - Methodology details
- **RL_Trading/README.md** - RL training documentation

---

## Notes

- Model checkpoints are large files (~24 files, ~100MB total)
- Consider Git LFS for version controlling checkpoints
- Market_Sim simulations can run for 15+ minutes
- RL training requires several hours for full convergence

---

## Citation

If you use this research or code:

```
Housing Market Liquidity Study with REIT-Based Transition Matrices
Market Simulation Analysis, January 2026
Data: VNQ (Vanguard Real Estate ETF), 2005-2026
```
