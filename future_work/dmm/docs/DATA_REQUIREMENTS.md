# Data Requirements for Deep Markov Model Training

## 📊 Current Data vs Required Data

### What You Have Now:
```
Sequences:           83 total
├─ Traditional:      67 sequences (72 months each) = 4,824 observations
└─ Tokenized:        16 sequences (72 months each) = 1,152 observations

Model Parameters:    ~50,000 (with hidden_dim=128)
Data:Parameter Ratio: 0.1x (critically insufficient)

Result: Posterior collapse (uniform 0.25 predictions)
```

### What You Need:

#### **Minimum Viable Training** (Conservative)
```
Sequences:           1,000+ total
├─ Traditional:      500 sequences (60-120 months each)
└─ Tokenized:        500 sequences (60-120 months each)

Total observations:  ~80,000-100,000
Data:Parameter Ratio: 2x (minimum for basic learning)
Expected Result:     May learn patterns, but still risky
```

#### **Recommended for Good Performance**
```
Sequences:           5,000-10,000 total
├─ Traditional:      2,000-4,000 sequences
├─ Tokenized:        2,000-4,000 sequences
└─ Transition:       1,000-2,000 sequences (mixed adoption levels)

Total observations:  ~400,000-800,000
Data:Parameter Ratio: 10-20x (good learning expected)
Expected Result:     Reliable context-dependent predictions
```

#### **Ideal for Production Use**
```
Sequences:           20,000+ total
├─ Diverse contexts: Multiple tokenization levels (0%, 25%, 50%, 75%, 100%)
├─ Various regimes:  Balanced representation of all 4 regimes
├─ Different eras:   Bull markets, bear markets, crashes, recoveries
└─ Multiple assets:  Different property types, geographies

Total observations:  1,000,000+
Data:Parameter Ratio: 20x+
Expected Result:     Robust, generalizable model
```

---

## 🔍 Types of Data Needed

### 1. **More Historical Time Series** (Primary Need)

#### Traditional Real Estate (Need: 500+ sequences)
```
Sources:
├─ NCREIF Property Index (quarterly 1978-present)
├─ Case-Shiller Home Price Index (monthly 1987-present)
├─ CoStar CRE data (various metros, property types)
├─ NAREIT data (pre-REIT era traditional holdings)
└─ International CRE indices (UK, Europe, Asia)

Approach:
- Slice by property type: office, retail, industrial, multifamily, hotel
- Slice by geography: major MSAs (50+ cities)
- Slice by time windows: rolling 72-month periods

Example:
  Property Type × Geography × Time Windows
  5 types × 50 cities × 20 windows = 5,000 sequences
```

#### Tokenized/REIT Data (Need: 500+ sequences)
```
Sources:
├─ Individual REIT stocks (200+ publicly traded REITs)
├─ REIT ETFs (VNQ, IYR, RWR, SCHH, etc.)
├─ Tokenized real estate platforms:
│   ├─ RealT (Ethereum-based tokenized properties)
│   ├─ Lofty.ai (Algorand-based)
│   ├─ Slice (fractional ownership)
│   └─ Fundrise eREITs
└─ International REITs (Europe, Asia-Pacific)

Approach:
- Individual REIT monthly returns: 200 REITs × 5 years = 200 sequences
- Subsector indices: Commercial, Residential, Industrial, etc.
- Different tokenization platforms to capture varying liquidity
```

### 2. **Synthetic Data Generation** (Practical Supplement)

When real data is limited, generate synthetic sequences that preserve statistical properties:

#### Method A: Bootstrap from Existing Data
```python
# Generate 1,000 new sequences from your 83 real ones
for i in range(1000):
    # Randomly select a base sequence
    base_seq = random.choice(real_sequences)
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, len(base_seq))
    synthetic_seq = base_seq * (1 + noise)
    
    # Randomly perturb volatility regime transitions
    # (keep statistical properties but create variation)
```

#### Method B: Regime-Switching Model
```python
# Fit a regime-switching model to each of your empirical matrices
# Generate new sequences by:
# 1. Sampling regime path from transition matrix
# 2. Sampling returns from regime-conditional distributions
# 3. Creating diverse but realistic price paths

for tokenization_level in [0.0, 0.25, 0.5, 0.75, 1.0]:
    P_matrix = interpolate_matrices(P_trad, P_token, tokenization_level)
    sequences = generate_sequences(P_matrix, n_sequences=200)
```

#### Method C: GAN-based Generation
```python
# Train a GAN to generate realistic price sequences
# Conditions: tokenization level, starting regime, property type
# Output: Synthetic sequences matching statistical properties
# (Advanced - requires significant ML expertise)
```

### 3. **Richer Context Features** (Improve Learning)

Current context (3 features):
```python
context = [is_tokenized, time_normalized, adoption_rate]
```

Enhanced context (10+ features):
```python
context = [
    # Market state
    is_tokenized,           # 0/1
    adoption_rate,          # 0-1
    liquidity_score,        # 0-1 (estimated from bid-ask spreads)
    
    # Economic conditions
    interest_rate,          # Fed funds rate, normalized
    gdp_growth,            # Real GDP growth
    inflation_rate,         # CPI
    
    # Property-specific
    property_type_onehot,   # [office, retail, industrial, residential, hotel]
    geography_code,         # Encoded region/MSA
    
    # Time features
    time_normalized,        # 0-1
    month_of_year,         # Seasonal effects
    years_since_crisis,     # Time since last major shock
]
```

Better features → Model learns richer patterns → Less data needed

---

## 📥 How to Acquire the Data

### Option 1: Purchase Commercial Data (**Best for Production**)

#### Providers:
```
CoStar (CRE data):           $10,000-50,000/year
├─ 50+ markets
├─ All property types
├─ Transaction-level data
└─ Monthly/quarterly updates

NCREIF ($2,500-5,000/year):
├─ Property-level returns
├─ 1978-present
└─ Institutional-grade data

NAREIT (Free-$500):
├─ REIT indices
├─ Subsector breakdowns
└─ Monthly data

Bloomberg Terminal ($24,000/year):
├─ Global REIT data
├─ Tokenized assets
└─ Extensive historical data
```

**Cost:** $15,000-75,000/year for comprehensive coverage

**Benefit:** Clean, validated, extensive data → 5,000+ sequences possible

### Option 2: Free/Low-Cost Data Sources (**Practical Starting Point**)

```python
# Script to gather free data

import yfinance as yf
import pandas as pd

# 1. Download all US REITs (200+ tickers)
reit_tickers = [
    'VNQ', 'IYR', 'SCHH', 'RWR',  # ETFs
    'PLD', 'AMT', 'CCI', 'EQIX',  # Individual REITs
    'PSA', 'O', 'DLR', 'SPG',
    # ... 200+ more
]

for ticker in reit_tickers:
    data = yf.download(ticker, start='2000-01-01', interval='1mo')
    # Extract 72-month windows
    # Label as tokenized (is_tokenized=1.0)
    sequences.append(...)

# 2. FRED (Federal Reserve Economic Data) - FREE
# - House price indices by metro
# - Commercial real estate indices
# - 30+ metros × 20 years = 600+ sequences
```

**Cost:** $0 (just API rate limits)

**Benefit:** 500-1,000 sequences achievable

### Option 3: Synthetic Augmentation (**Fastest**)

```python
# Use your existing 83 sequences to generate 1,000+

from dmm.train_dmm import prepare_training_data

real_data = prepare_training_data()

# Bootstrap with variations
synthetic_data = augment_with_bootstrap(
    real_data,
    target_sequences=1000,
    noise_level=0.02,  # 2% price noise
    regime_perturbation=0.1  # 10% transition probability noise
)

# Result: 1,000 sequences from 83 real ones
# Trade-off: Less real diversity, but still useful for training
```

**Cost:** $0 (just compute time)

**Benefit:** Immediate 10-20x data increase

---

## 🔬 Detailed Data Generation Example

Here's a complete script to generate sufficient data:

```python
"""
Generate 1,000+ training sequences from limited real data
"""

import numpy as np
import yfinance as yf
from typing import List, Tuple

def download_all_reit_data() -> List[np.ndarray]:
    """
    Download monthly data for 200+ REITs and REIT ETFs.
    Each becomes a potential training sequence.
    """
    # Major REIT ETFs
    etf_tickers = ['VNQ', 'IYR', 'SCHH', 'RWR', 'XLRE', 'USRT', 'ICF', 'FREL']
    
    # Top 100 individual REITs by market cap
    reit_tickers = [
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'O', 'DLR', 'SPG',
        'WELL', 'AVB', 'EQR', 'SBAC', 'WY', 'BXP', 'ARE', 'VICI',
        'VTR', 'INVH', 'ESS', 'MAA', 'KIM', 'DOC', 'UDR', 'HST',
        'REG', 'CPT', 'FRT', 'EXR', 'PEAK', 'SUI', 'CUBE', 'ELS',
        # ... add 70+ more
    ]
    
    all_sequences = []
    
    for ticker in etf_tickers + reit_tickers:
        try:
            # Download monthly data
            data = yf.download(ticker, start='2000-01-01', interval='1mo', progress=False)
            prices = data['Adj Close'].values
            
            # Create 72-month sliding windows
            for start_idx in range(0, len(prices) - 72, 12):  # Stride=12 months
                window = prices[start_idx:start_idx + 72]
                all_sequences.append(window)
                
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            continue
    
    return all_sequences


def generate_synthetic_sequences(
    real_sequences: List[np.ndarray],
    n_synthetic: int = 1000,
    noise_std: float = 0.02
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate synthetic sequences by:
    1. Sampling from real sequences
    2. Adding realistic noise
    3. Perturbing returns slightly
    """
    synthetic_sequences = []
    labels = []  # is_tokenized flag
    
    np.random.seed(42)
    
    for _ in range(n_synthetic):
        # Sample a base sequence
        base_seq = real_sequences[np.random.randint(len(real_sequences))]
        
        # Calculate returns
        returns = np.diff(np.log(base_seq))
        
        # Add noise to returns
        noisy_returns = returns + np.random.normal(0, noise_std, len(returns))
        
        # Reconstruct prices
        synthetic_prices = [base_seq[0]]
        for ret in noisy_returns:
            synthetic_prices.append(synthetic_prices[-1] * np.exp(ret))
        
        synthetic_sequences.append(np.array(synthetic_prices))
        
        # Randomly assign tokenization label
        is_tokenized = np.random.rand()  # 0-1
        labels.append(is_tokenized)
    
    return synthetic_sequences, labels


def main():
    """Generate comprehensive training dataset."""
    
    print("Step 1: Downloading real REIT data...")
    reit_sequences = download_all_reit_data()
    print(f"  Downloaded {len(reit_sequences)} sequences")
    
    print("\nStep 2: Loading your existing CRE data...")
    from dmm.train_dmm import prepare_training_data
    existing_data = prepare_training_data()
    cre_sequences = list(existing_data['prices'])
    print(f"  Loaded {len(cre_sequences)} sequences")
    
    print("\nStep 3: Generating synthetic augmentations...")
    all_real = reit_sequences + cre_sequences
    synthetic_sequences, labels = generate_synthetic_sequences(
        all_real,
        n_synthetic=1000,
        noise_std=0.02
    )
    print(f"  Generated {len(synthetic_sequences)} synthetic sequences")
    
    print("\nFinal Dataset:")
    print(f"  Total sequences: {len(all_real) + len(synthetic_sequences)}")
    print(f"  Real REIT: {len(reit_sequences)}")
    print(f"  Real CRE: {len(cre_sequences)}")
    print(f"  Synthetic: {len(synthetic_sequences)}")
    
    return {
        'sequences': all_real + synthetic_sequences,
        'labels': [1.0]*len(reit_sequences) + [0.0]*len(cre_sequences) + labels
    }

if __name__ == "__main__":
    dataset = main()
```

**This approach gets you:**
- ~200-300 real REIT sequences (from yfinance - FREE)
- ~83 real CRE sequences (your existing data)
- ~1,000 synthetic sequences (augmented)
- **Total: ~1,300 sequences** → 93,600 observations
- **Ratio: ~2x parameters** → Should prevent posterior collapse!

---

## ⚖️ Trade-offs: More Data vs Hybrid Model

| Aspect | Collect More Data + DMM | Use Hybrid Model |
|--------|-------------------------|------------------|
| **Upfront Cost** | $0-$50K (depends on sources) | $0 |
| **Time Investment** | 1-4 weeks data collection + 2-6 hours training | 0 minutes (works now) |
| **Data Required** | 1,000-10,000 sequences | 0 (uses existing) |
| **Interpretability** | Black box (neural network) | Fully transparent |
| **Maintenance** | Retrain when new data arrives | Update empirical matrices |
| **Risk** | May still collapse with synthetic data | Zero risk |
| **Flexibility** | Can capture complex nonlinear effects | Linear interpolation only |
| **Result Quality** | Potentially better if data is diverse | Proven to work well |

---

## 🎯 Recommendation Matrix

### Use **Hybrid Model** if you:
- ✅ Have <500 real sequences
- ✅ Need results now
- ✅ Want interpretable predictions
- ✅ Have 2 main contexts (traditional vs tokenized)
- ✅ Can't afford $10K+ for commercial data

### Use **Simplified DMM** (`train_dmm_simple.py`) if you:
- ✅ Can collect 500-1,000 sequences (free sources + augmentation)
- ✅ Can wait 1-2 weeks for data collection
- ✅ Want to learn from the data
- ✅ Don't mind 2-3 hour training time

### Use **Full DMM** (hidden_dim=128) if you:
- ✅ Can collect 5,000+ real sequences
- ✅ Have budget for commercial data ($15K+)
- ✅ Need to capture complex nonlinear patterns
- ✅ Have >2 contexts (e.g., multiple tokenization platforms, geographies)
- ✅ Can validate on held-out test sets

---

## 📋 Action Plan to Get More Data

### Week 1: Free Data Collection
```bash
# Day 1-2: REIT data (yfinance)
python3 dmm/download_reit_data.py  # 200-300 sequences

# Day 3: FRED data (house prices)
python3 dmm/download_fred_data.py  # 50-100 sequences

# Day 4-5: Web scraping public sources
# - Zillow ZHVI (home values)
# - Real Capital Analytics (free tier)
# - Academic datasets

Total: 300-500 sequences (FREE)
```

### Week 2-3: Commercial Data (Optional)
```bash
# Option 1: Purchase NCREIF access ($2,500)
# → 1,000+ institutional-grade sequences

# Option 2: Purchase CoStar trial ($500/month)
# → 2,000+ metro-level sequences

Total: 1,000-3,000 sequences ($500-$2,500)
```

### Week 4: Synthetic Augmentation
```bash
# Use bootstrap + GAN to generate 2,000 synthetic sequences
python3 dmm/generate_synthetic_data.py

Total: 3,000-5,500 sequences (depending on Weeks 1-3)
```

### Result after 4 weeks:
- **3,000-5,500 sequences**
- **~250,000 observations**
- **Ratio: 5-10x parameters**
- **Sufficient for DMM training** ✅

---

## 🧮 Quick Calculator

**To determine if you have enough data:**

```python
def check_data_sufficiency(
    n_sequences: int,
    sequence_length: int = 72,
    hidden_dim: int = 128
):
    """
    Check if you have enough data for DMM training.
    """
    # Estimate model parameters
    n_regimes = 4
    context_dim = 3
    
    # Transition network
    trans_params = (n_regimes + context_dim) * hidden_dim + \
                   hidden_dim * hidden_dim + \
                   hidden_dim * n_regimes
    
    # Emission network (similar size)
    emis_params = trans_params
    
    # Inference network (similar size)
    infer_params = trans_params
    
    total_params = trans_params + emis_params + infer_params
    
    # Total observations
    total_obs = n_sequences * sequence_length
    
    # Ratio
    ratio = total_obs / total_params
    
    print(f"Model parameters: {total_params:,}")
    print(f"Total observations: {total_obs:,}")
    print(f"Data:Parameter ratio: {ratio:.2f}x")
    print()
    
    if ratio < 0.5:
        print("❌ CRITICAL: Severe underfitting risk (posterior collapse likely)")
        print(f"   Need {int(total_params * 2):,} observations (minimum)")
    elif ratio < 2:
        print("⚠️  WARNING: Underfitting risk (may collapse)")
        print(f"   Recommended: {int(total_params * 5):,} observations")
    elif ratio < 10:
        print("✅ ACCEPTABLE: Should work but may be unstable")
        print(f"   Ideal: {int(total_params * 10):,} observations")
    else:
        print("✅ EXCELLENT: Well-resourced training expected")

# Test with your current data
check_data_sufficiency(n_sequences=83, hidden_dim=128)

# Test with 1,000 sequences
check_data_sufficiency(n_sequences=1000, hidden_dim=128)

# Test with simplified model (hidden_dim=32)
check_data_sufficiency(n_sequences=1000, hidden_dim=32)
```

---

## 💡 Key Insight

**The uncomfortable truth:** To justify using a complex neural network model, you need **1,000-10,000 sequences**. 

**Your options:**
1. **Spend 2-4 weeks collecting data** (500-1000 sequences achievable for free)
2. **Use simplified DMM** with augmentation (try `train_dmm_simple.py`)
3. **Use hybrid model** (works now, proven results)

**For most practical purposes, Option 3 (hybrid model) is the best choice** unless you're doing academic research or have funding for commercial data.

---

Would you like me to create the data collection scripts (`download_reit_data.py`, `generate_synthetic_data.py`) to help you gather 1,000+ sequences?
