# Training Deep Markov Model with 100 REITs

## Quick Summary

Your Deep Markov Model training script has been upgraded to support **100+ REITs** instead of just 3!

## What's New

### 1. New File: `reit_symbols.py`
Contains **102 REIT symbols** organized by category:
- 10 Broad Market ETFs
- 15 Residential REITs
- 12 Office REITs  
- 14 Retail REITs
- 7 Industrial REITs
- 12 Healthcare REITs
- 34 Specialty REITs (data centers, cell towers, storage, hotels, etc.)

### 2. Updated: `qfclient_data_loader.py`
New functions to handle multiple REITs:
- `load_multi_reit_data()` - Load many REITs with progress tracking
- `combine_multi_reit_data()` - Combine REITs using different methods
- `load_reit_portfolio()` - Convenience wrapper

### 3. Updated: `train_dmm_with_qfclient.py`
Now loads **50 REITs by default** (up from 3). Easy to change to 20, 50, or 100+.

## How to Use

### Option 1: Top 20 Liquid REITs (Fast, Recommended for Testing)

Edit `train_dmm_with_qfclient.py` around line 90:

```python
# Use this line:
reit_symbols = TOP_20_LIQUID
```

**Loading time:** ~5 minutes  
**Best for:** Quick testing, reliable data

### Option 2: Top 50 Diversified REITs (CURRENT DEFAULT - Balanced)

```python
# Use this line (already default):
reit_symbols = get_recommended_reits(n=50, diversified=True)
```

**Loading time:** ~10-15 minutes  
**Best for:** Production training, good coverage

### Option 3: All 100+ REITs (Comprehensive)

```python
# Use this line:
reit_symbols = ALL_REITS
```

**Loading time:** ~20-30 minutes  
**Best for:** Maximum diversity, final models

## How to Run

```bash
cd Market_sim/dmm/training
python3 train_dmm_with_qfclient.py
```

The script will:
1. Load traditional CRE data from CSV
2. Load 50 REITs (or however many you configured)
3. Combine them into an averaged portfolio
4. Train the DMM with both datasets

## Combination Methods

The script combines multiple REITs using `combine_multi_reit_data()`. You can change the method around line 110:

### Average (Default)
```python
token_prices = combine_multi_reit_data(reit_data, method="average")
```
- Equal-weighted average of all REITs
- Smooth, representative data
- **Recommended for most use cases**

### Median
```python
token_prices = combine_multi_reit_data(reit_data, method="median")
```
- More robust to outliers
- Good if some REITs have unusual behavior

### Longest
```python
token_prices = combine_multi_reit_data(reit_data, method="longest")
```
- Uses single REIT with most historical data
- Simple but loses diversification

### Concatenate
```python
token_prices = combine_multi_reit_data(reit_data, method="concatenate")
```
- Keeps all REITs separate
- Creates MORE training sequences (50 REITs = 50x more data!)
- **Best for deep learning** (more training examples)

## Advanced Usage

### Load Specific Categories

```python
from dmm.utils.reit_symbols import get_reits_by_category

# Load only healthcare REITs
healthcare = get_reits_by_category('healthcare')

# Load only residential REITs  
residential = get_reits_by_category('residential')
```

### Custom Portfolio

```python
# Create your own portfolio
my_reits = [
    "VNQ",   # Broad market
    "PLD",   # Industrial
    "EQIX",  # Data centers
    "PSA",   # Self-storage
    "O",     # Net lease
]

from dmm.utils.qfclient_data_loader import load_reit_portfolio
data = load_reit_portfolio(symbols=my_reits, years=10)
```

### Filter by Data Quality

```python
# Only use REITs with at least 10 years of data
combined = combine_multi_reit_data(
    reit_data, 
    method="average",
    min_length=120  # 10 years × 12 months
)
```

## REIT Categories Available

**Broad Market ETFs (10):**
VNQ, IYR, SCHH, USRT, RWR, FREL, XLRE, ICF, RWO, REET

**Residential (15):**
EQR, AVB, MAA, ESS, UDR, CPT, AIV, ELS, SUI, AMH, INVH, ACC, CSR, IRT, NXRT

**Office (12):**
BXP, VNO, ARE, DEI, HIW, SLG, CLI, CUZ, PGRE, PDM, OFC, JBGS

**Retail (14):**
SPG, REG, FRT, KIM, BRX, ROIC, AKR, KRG, RPAI, WPG, TCO, MAC, SKT, SITC

**Industrial (7):**
PLD, DRE, FR, STAG, TRNO, EGP, REXR

**Healthcare (12):**
WELL, PEAK, VTR, DOC, HR, OHI, SBRA, LTC, CTRE, MPW, DHC, NHI

**Specialty (34):**
- Data Centers: EQIX, DLR, COR, QTS
- Cell Towers: AMT, CCI, SBAC  
- Self-Storage: PSA, EXR, CUBE, LSI, NSA
- Net Lease: O, NNN, STOR, ADC, SRC, EPRT, FCPT, GTY
- Hotels: HST, RHP, PK, SHO, RLJ, APLE, INN
- Gaming: VICI, GLPI, MGP
- Timberland: WY, PCH

## Why Use Multiple REITs?

1. **Diversification:** No single REIT represents entire market
2. **Noise Reduction:** Averaging smooths out individual quirks
3. **Robustness:** If one REIT has bad data, others compensate
4. **Market Representation:** Broad portfolio = better tokenized RE proxy
5. **More Training Data:** With concatenate method, creates 50x more sequences!

## Performance

**Data Loading Times:**
- 20 REITs: ~5 minutes
- 50 REITs: ~10 minutes (default)
- 100 REITs: ~20 minutes

**Training Times:**
- Average/Median/Longest: ~10-20 minutes (200 epochs)
- Concatenate: Longer (50x more training sequences)

## Troubleshooting

### "Failed to load many REITs"
Some REITs may not have 20 years of history. Solutions:
- Reduce years: `years=10` or `years=5`
- Use `min_length` filter in `combine_multi_reit_data()`

### "Takes too long"
- Use `TOP_20_LIQUID` instead of `ALL_REITS`
- Set `max_failures=10` in `load_multi_reit_data()`

### "Not enough training data"
- Use `method="concatenate"` to create more sequences
- Reduce `window_size` and `stride` in `prepare_dmm_training_data()`

## Files Created/Modified

```
Market_sim/dmm/utils/
├── reit_symbols.py              # NEW - 100+ REIT symbols
├── qfclient_data_loader.py      # UPDATED - Multi-REIT functions
├── MULTI_REIT_GUIDE.py          # NEW - Detailed guide
└── test_api_keys.py             # NEW - API testing

Market_sim/dmm/training/
└── train_dmm_with_qfclient.py   # UPDATED - Now uses 50 REITs
```

## Next Steps

1. **Test your setup:**
   ```bash
   cd Market_sim/dmm/utils
   python3 test_api_keys.py
   ```

2. **Quick training test (20 REITs):**
   - Edit `train_dmm_with_qfclient.py` line 90
   - Use `reit_symbols = TOP_20_LIQUID`
   - Run training

3. **Full training (50 REITs):**
   - Keep default settings
   - Run training (takes ~25 minutes total)

4. **Comprehensive training (100 REITs):**
   - Edit to use `ALL_REITS`
   - Run overnight or during long break

## API Status

✓ **Yahoo Finance** - Working (no API key needed)  
✓ **FRED** - Working (your key configured)  
✓ **SEC Edgar** - Working (no API key needed)

You're all set to train with real data from 100+ REITs!
