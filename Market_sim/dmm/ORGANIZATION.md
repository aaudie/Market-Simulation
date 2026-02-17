# DMM Folder Organization (February 2026)

## 📋 Summary

The DMM folder has been reorganized into a clean, modular structure with clear separation of concerns.

## 🗂️ New Structure

```
dmm/
├── README.md                      # Main documentation (start here!)
├── requirements.txt               # Python dependencies
├── __init__.py                    # Package initialization
├── ORGANIZATION.md                # This file
│
├── core/                          # 🧠 Core Model Implementation
│   ├── __init__.py
│   └── deep_markov_model.py      # Main DeepMarkovModel class (596 lines)
│
├── training/                      # 🏋️ Training Scripts
│   ├── __init__.py
│   ├── train_dmm.py              # Original training script
│   ├── train_dmm_simple.py       # Simplified training with synthetic data
│   └── train_dmm_with_qfclient.py # **Recommended**: Training with real QFClient data
│
├── utils/                         # 🔧 Utilities & Helpers
│   ├── __init__.py
│   ├── qfclient_data_loader.py   # Load data from QFClient API
│   ├── check_data_sufficiency.py # Validate data quality
│   ├── use_empirical_matrices.py # Empirical matrix utilities
│   ├── integrate_dmm.py          # Integration demo with simulator
│   └── examples.py               # Usage examples
│
└── docs/                          # 📚 Documentation
    ├── START_HERE.md             # Entry point for new users
    ├── QUICKSTART.md             # Step-by-step setup guide
    ├── IMPLEMENTATION.md         # Technical details
    ├── DATA_REQUIREMENTS.md      # Data format requirements
    ├── FIXING_POSTERIOR_COLLAPSE.md # Troubleshooting guide
    ├── README_FIRST.md           # Alternative entry point
    └── REORGANIZATION_SUMMARY.md # Previous reorganization notes
```

## 🎯 Design Principles

### 1. Separation of Concerns
- **Core**: Model implementation only
- **Training**: Scripts to train models
- **Utils**: Helper functions and data loaders
- **Docs**: All documentation

### 2. Clear Entry Points
- New users → `README.md`
- Quick setup → `docs/QUICKSTART.md`
- Technical details → `docs/IMPLEMENTATION.md`

### 3. Pythonic Structure
- Proper `__init__.py` in all packages
- Clean imports: `from dmm import DeepMarkovModel`
- Modular design for easy extension

### 4. Self-Documenting
- Clear folder names
- Comprehensive README
- Inline comments in code

## 📝 Import Patterns

### ✅ Recommended (Clean)
```python
# Import from main package
from dmm import DeepMarkovModel, TORCH_AVAILABLE

# Import utilities
from dmm.utils.qfclient_data_loader import load_reit_data
from dmm.utils.use_empirical_matrices import HybridMarkovModel

# Import training helpers
from dmm.training.train_dmm import prepare_training_data
```

### ✅ Also Valid (Explicit)
```python
# Import directly from core
from dmm.core.deep_markov_model import DeepMarkovModel
```

### ❌ Avoid (Old Structure)
```python
# These won't work with the new organization
from dmm.deep_markov_model import DeepMarkovModel  # deep_markov_model is now in core/
from dmm.qfclient_data_loader import load_reit_data  # qfclient_data_loader is now in utils/
```

## 🚀 Quick Commands

All commands should be run from `Market_Sim/Market_sim/` directory:

```bash
# Install dependencies
pip install -r dmm/requirements.txt

# Train model (recommended)
python3 dmm/training/train_dmm_with_qfclient.py

# Run examples
python3 dmm/utils/examples.py

# Check data quality
python3 dmm/utils/check_data_sufficiency.py

# Integration demo
python3 dmm/utils/integrate_dmm.py
```

## 🔄 Migration Guide

If you have existing code using the old structure:

### 1. Update Imports
```python
# OLD
from dmm.deep_markov_model import DeepMarkovModel
from dmm.qfclient_data_loader import load_reit_data

# NEW
from dmm import DeepMarkovModel
from dmm.utils.qfclient_data_loader import load_reit_data
```

### 2. Update Script Paths
```bash
# OLD
python3 dmm/train_dmm_with_qfclient.py

# NEW (still works, but location changed)
python3 dmm/training/train_dmm_with_qfclient.py
```

### 3. Update Documentation References
- `DMM_README.md` → `dmm/README.md`
- `DMM_QUICKSTART.md` → `dmm/docs/QUICKSTART.md`
- `IMPLEMENTATION.md` → `dmm/docs/IMPLEMENTATION.md`

## ✅ Benefits

1. **Easier Navigation**: Find what you need quickly
2. **Better Organization**: Logical grouping of related files
3. **Cleaner Imports**: Simpler import statements
4. **Maintainability**: Easy to add new features
5. **Professional**: Standard Python package structure
6. **Portable**: Can share/move entire dmm folder
7. **Scalable**: Easy to extend with new modules

## 📊 File Count

- **Core**: 1 file (596 lines)
- **Training**: 3 scripts
- **Utils**: 5 utilities
- **Docs**: 7 documentation files
- **Total**: 16+ files organized into 4 clear categories

## 🐛 Troubleshooting

### Import errors?
- Make sure you're using the new import paths
- Check that `__init__.py` exists in each package folder
- Run from `Market_Sim/Market_sim/` directory

### Can't find files?
- Check the folder structure above
- Use `ls dmm/` to see the new layout
- All markdown docs are now in `docs/`

### Training scripts not working?
- Imports have been updated automatically
- Make sure to use: `python3 dmm/training/script_name.py`
- Check that dependencies are installed

## 📅 Change Log

**February 11, 2026**: Major reorganization
- Created `core/`, `training/`, `utils/`, `docs/` folders
- Moved all files to appropriate locations
- Updated all import statements
- Created comprehensive documentation
- Added proper `__init__.py` files

## 🤝 Contributing

When adding new files, follow these guidelines:

- **Model code** → `core/`
- **Training scripts** → `training/`
- **Helper functions** → `utils/`
- **Documentation** → `docs/`
- **Configuration** → Root of `dmm/`

Keep the structure clean and logical!

---

**Questions?** Check `README.md` or the documentation in `docs/`
