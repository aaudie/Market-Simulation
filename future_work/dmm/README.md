# Deep Markov Model (DMM) - Organized Structure

A neural network-based extension of Hidden Markov Models for learning context-dependent regime dynamics in tokenized vs traditional CRE markets.

## 📁 Folder Structure

```
dmm/
├── __init__.py                    # Package initialization - imports from core/
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── core/                          # Core model implementation
│   ├── __init__.py
│   └── deep_markov_model.py      # Main DeepMarkovModel class (596 lines)
│
├── training/                      # Training scripts
│   ├── __init__.py
│   ├── train_dmm.py              # Original training script
│   ├── train_dmm_simple.py       # Simplified training
│   └── train_dmm_with_qfclient.py # Training with QFClient data (recommended)
│
├── utils/                         # Utility scripts
│   ├── __init__.py
│   ├── integrate_dmm.py          # Integration with simulator
│   ├── examples.py               # Usage examples
│   ├── check_data_sufficiency.py # Data validation utility
│   ├── qfclient_data_loader.py   # QFClient data loader
│   └── use_empirical_matrices.py # Empirical matrix utilities
│
└── docs/                          # Documentation
    ├── START_HERE.md             # Entry point for new users
    ├── QUICKSTART.md             # Step-by-step setup guide
    ├── IMPLEMENTATION.md         # Technical implementation details
    ├── DATA_REQUIREMENTS.md      # Data format and requirements
    ├── FIXING_POSTERIOR_COLLAPSE.md # Troubleshooting guide
    └── REORGANIZATION_SUMMARY.md # Previous reorganization notes
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd Market_Sim/Market_sim
pip install -r dmm/requirements.txt
```

### 2. Import and Use
```python
# Simple import (recommended)
from dmm import DeepMarkovModel

# Or import from core directly
from dmm.core.deep_markov_model import DeepMarkovModel

# Load trained model
dmm = DeepMarkovModel()
dmm.load('outputs/deep_markov_model_qfclient.pt')

# Predict next regime
next_regime, probs = dmm.predict_next_regime(
    current_regime='calm',
    context={'is_tokenized': 1.0, 'time_normalized': 0.5}
)
```

### 3. Train a New Model
```bash
# Recommended: Train with QFClient data
python3 dmm/training/train_dmm_with_qfclient.py

# Or use simple training
python3 dmm/training/train_dmm_simple.py
```

### 4. Run Examples
```bash
python3 dmm/utils/examples.py
```

## 📚 Documentation Guide

- **New to DMM?** Start with `docs/START_HERE.md`
- **Want to train quickly?** Read `docs/QUICKSTART.md`
- **Need technical details?** See `docs/IMPLEMENTATION.md`
- **Data issues?** Check `docs/DATA_REQUIREMENTS.md`
- **Training problems?** See `docs/FIXING_POSTERIOR_COLLAPSE.md`

## 🔧 Main Components

### Core Model (`core/deep_markov_model.py`)
- **DeepMarkovModel**: Main model class with training/inference
- **TransitionNetwork**: Learns context-dependent regime transitions
- **EmissionNetwork**: Generates price/volatility distributions
- **InferenceNetwork**: Infers regimes from observations

### Training Scripts (`training/`)
- `train_dmm_with_qfclient.py`: **Recommended** - Uses real QFClient data
- `train_dmm_simple.py`: Simplified training with synthetic data
- `train_dmm.py`: Original training script

### Utilities (`utils/`)
- `qfclient_data_loader.py`: Load and preprocess QFClient data
- `check_data_sufficiency.py`: Validate data quality and quantity
- `integrate_dmm.py`: Integration demo with market simulator
- `examples.py`: Usage examples and code snippets
- `use_empirical_matrices.py`: Work with empirical transition matrices

## 🎯 Key Features

✅ **Context-aware transitions**: Learns how market context affects regime changes
✅ **Flexible architecture**: Configurable network sizes and parameters
✅ **Real data integration**: Works with QFClient historical data
✅ **Comprehensive logging**: Detailed training metrics and visualizations
✅ **Easy to use**: Simple API for training and inference
✅ **Well-organized**: Clear separation of concerns

## 📊 Model Inputs/Outputs

### Inputs
- **Current regime**: Market regime (calm, volatile, high_appreciation, declining)
- **Context features**: 
  - `is_tokenized`: 0.0 or 1.0
  - `time_normalized`: 0.0 to 1.0 (normalized simulation time)
  - `adoption_rate`: 0.0 to 1.0 (optional)

### Outputs
- **Next regime prediction**: Predicted regime name
- **Transition probabilities**: Dict of regime → probability
- **Confidence**: Prediction confidence score

## 🔄 Recent Changes (Feb 2026)

This folder was reorganized to improve maintainability:

1. **Core code separated** from training/utilities
2. **Documentation consolidated** in `docs/` folder
3. **Training scripts grouped** together
4. **Utilities centralized** for easy discovery
5. **Package structure improved** with proper `__init__.py` files

## 🐛 Troubleshooting

### Import Errors
```python
# ✅ Correct (new structure)
from dmm import DeepMarkovModel

# ❌ Old structure (will not work)
from dmm.deep_markov_model import DeepMarkovModel
```

### Training Issues
- Check `docs/FIXING_POSTERIOR_COLLAPSE.md` for common problems
- Validate data with `python3 dmm/utils/check_data_sufficiency.py`
- Review data requirements in `docs/DATA_REQUIREMENTS.md`

### Path Issues
All scripts should be run from `Market_Sim/Market_sim/` directory:
```bash
cd Market_Sim/Market_sim
python3 dmm/training/train_dmm_with_qfclient.py
```

## 📝 Version

Current version: **1.0.0** (Reorganized February 2026)

## 🤝 Contributing

When adding new files:
- **Models/algorithms** → `core/`
- **Training scripts** → `training/`
- **Helper functions** → `utils/`
- **Documentation** → `docs/`

---

**Questions?** Check the documentation in `docs/` or read the inline comments in the code.
