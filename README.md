# Market Simulation with AI

This repository contains two main projects:

## Project Structure

```
Market_Sim(wAI)/
├── Market_Sim/              # Main market simulation project
│   ├── main.py              # Main entry point
│   ├── QUICK_START.txt      # Quick start guide
│   ├── scripts/             # Analysis scripts
│   │   ├── analyze_reit_regimes.py
│   │   ├── housing_liquidity_comparison.py
│   │   ├── residential_equity_regimes.py
│   │   └── run_complete_analysis.py
│   ├── outputs/             # Generated outputs (plots, CSVs, reports)
│   │   ├── *.png            # Visualization outputs
│   │   ├── *.csv            # Data outputs
│   │   └── *.md             # Analysis reports
│   └── sim/                 # Core simulation engine
│       ├── agents/          # Trading agents
│       ├── calibration.py
│       ├── data_loader.py
│       ├── market_simulator.py
│       ├── microstructure.py
│       ├── portfolio.py
│       ├── regimes.py
│       ├── types.py
│       └── utils.py
│
├── RL_Trading/              # Reinforcement Learning Trading Project
│   ├── src/                 # Source code
│   │   ├── indicators.py
│   │   ├── trading_env.py
│   │   ├── train_agent.py
│   │   └── test_agent.py
│   ├── data/                # Trading data (EURUSD, Equinix, etc.)
│   ├── checkpoints/         # Model checkpoints
│   ├── model_*.zip          # Best trained models
│   └── requirements.txt     # Python dependencies for RL Trading
│
├── notebooks/               # Jupyter notebooks
│   └── markov_chains (1).ipynb
│
├── requirements.txt         # Main project dependencies
└── .gitignore              # Git ignore rules
```

## Getting Started

### Market Simulation
```bash
cd Market_Sim
python main.py
```

### RL Trading
```bash
cd RL_Trading
pip install -r requirements.txt
python src/train_agent.py
```

## What Changed?

This repository was recently reorganized for better clarity:
- **Separated projects**: Market_Sim and RL_Trading are now clearly separated
- **Organized outputs**: All generated files (plots, CSVs, reports) are in `Market_Sim/outputs/`
- **Consolidated scripts**: All analysis scripts are in `Market_Sim/scripts/`
- **Cleaned caches**: Removed all `__pycache__`, `.DS_Store`, and `.pyc` files
- **Added .gitignore**: Prevents cache files from accumulating in the future

## Notes

- Model checkpoints are large files (~24 checkpoint files total)
- Consider using Git LFS if you plan to version control the checkpoints
- The `.gitignore` file is configured to exclude common Python and macOS artifacts
