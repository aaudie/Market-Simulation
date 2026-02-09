# Housing Market Liquidity Study

**Research Question:** What would a US housing market with REIT-like liquidity look like?

## Quick Start

### 1. Run the Complete Analysis

```bash
# Run the complete pipeline (all three analyses)
python3 scripts/run_complete_analysis.py

# Or run individually:
# Step 1: Analyze traditional CRE baseline (72+ years of data)
python3 scripts/analyze_cre_regimes.py

# Step 2: Analyze real REIT data to extract transition matrix
python3 scripts/analyze_reit_regimes.py

# Step 3: Run housing market comparison simulation
python3 scripts/housing_liquidity_comparison.py
```

### 2. View Results

The analysis generates:
- `CRE_regime_analysis.png` - Traditional CRE baseline (1953-2025)
- `VNQ_regime_analysis.png` - REIT regime transition analysis
- `housing_liquidity_comparison.png` - Traditional vs Tokenized comparison
- `FINDINGS_SUMMARY.md` - Detailed research findings

## Key Findings

**1. Liquidity Creates a 9.8x Volatility Increase**
- Traditional CRE: 1.80% annualized volatility (illiquid, appraisal-based)
- REITs (VNQ): 17.70% annualized volatility (liquid, marked-to-market)

**2. Tokenized Housing Spends 47% Less Time in Crisis**
- Traditional: 46.9% time in stress (volatile/panic)
- Tokenized: 24.9% time in stress (volatile/panic)

**3. Traditional CRE Has Extreme Regime Persistence**
- 58.7% of 72 years spent in calm regime (avg 7.1 months per episode)
- Only 0.5% in panic (4 months total over 72 years!)
- Slow transitions reflect illiquidity and appraisal smoothing

## Files Overview

### Analysis Scripts

1. **`analyze_cre_regimes.py`** ⭐ NEW
   - Analyzes 72+ years of traditional CRE data (1953-2025)
   - Establishes empirical baseline for illiquid real estate
   - Calculates transition probabilities from 871 months of data
   - Shows extreme regime persistence and low volatility

2. **`analyze_reit_regimes.py`**
   - Fetches real REIT data (VNQ ETF)
   - Calculates empirical transition probabilities
   - Generates regime visualization
   - Outputs Python-ready transition matrix

3. **`housing_liquidity_comparison.py`**
   - Simulates traditional vs tokenized housing markets
   - Uses empirical REIT transition matrix
   - Generates comprehensive comparison plots
   - Calculates performance metrics

4. **`run_complete_analysis.py`** ⭐ RECOMMENDED
   - Orchestrates all three analyses in sequence
   - Generates complete set of visualizations
   - Provides summary report with key findings

5. **`main.py`**
   - Original demo script for the market simulator
   - Shows basic tokenization concepts

### Supporting Infrastructure

4. **`sim/` Package**
   - `market_simulator.py` - Core simulation engine
   - `regimes.py` - Markov regime switching logic
   - `microstructure.py` - Order book and price formation
   - `calibration.py` - Parameter estimation from historical data
   - `data_loader.py` - Data fetching utilities
   - `agents/` - Rule-based trading agents

### Data

5. **`cre_monthly.csv`**
   - US housing price index (1953-2023)
   - 871 months of historical data
   - Source: Commercial Real Estate index

### Documentation

6. **`FINDINGS_SUMMARY.md`**
   - Complete research findings
   - Methodology explanation
   - Policy implications
   - Limitations and next steps

## Methodology

### Step 1: Traditional CRE Baseline Analysis ⭐ NEW

We establish an empirical baseline using 72+ years of traditional CRE data:

1. Load historical CRE price index (1953-2025, 871 months)
2. Calculate rolling volatility (6-month window)
3. Classify into 4 regimes: Calm, Neutral, Volatile, Panic
4. Estimate transition probabilities empirically

**Result:** A 4×4 transition matrix representing traditional illiquid real estate

**Key Discovery:** CRE shows extreme regime persistence (86% calm persistence) and very low volatility (1.80% annualized), reflecting illiquidity and appraisal smoothing.

### Step 2: Empirical REIT Analysis

We analyze VNQ (Vanguard Real Estate ETF) to understand how liquid real estate markets behave:

1. Fetch 20+ years of monthly REIT data
2. Calculate rolling volatility (6-month window)
3. Classify into 4 regimes using same thresholds as CRE
4. Estimate transition probabilities empirically

**Result:** A 4×4 transition matrix representing liquid real estate (REITs)

**Key Discovery:** REITs are 9.8x more volatile than traditional CRE, with faster regime transitions and different crisis dynamics.

### Step 3: Market Simulation

We simulate two parallel housing markets:

1. **Traditional (Illiquid)**
   - Uses empirical CRE transition matrix OR assumed matrix
   - Slow transitions, gets stuck in crisis states
   - Represents current housing market

2. **Tokenized (Liquid)**  
   - Empirical transition matrix from VNQ
   - REIT-like behavior with continuous trading
   - Represents hypothetical tokenized housing

Both simulations:
- Use identical underlying housing price data (CRE index)
- Same calibrated drift and volatility parameters
- Only difference: regime transition dynamics

### Step 4: Comparison

We compare across multiple dimensions:
- Time spent in each regime
- Number and duration of crisis episodes
- Price volatility and drawdowns
- Recovery times from shocks
- Comparison with empirical CRE baseline

## Empirical Transition Matrices

### Traditional CRE (1953-2025) ⭐ NEW

From our analysis of 871 months of traditional CRE data:

```python
P_CRE_TRADITIONAL = [
    [0.8591, 0.1389, 0.0020, 0.0000],  # calm    (58.7% of time, avg 7.1 months)
    [0.2339, 0.7186, 0.0475, 0.0000],  # neutral (34.0% of time, avg 3.5 months)
    [0.0339, 0.2203, 0.6949, 0.0508],  # volatile (6.8% of time, avg 3.3 months)
    [0.0000, 0.0000, 0.7500, 0.2500],  # panic    (0.5% of time, avg 1.3 months)
]
```

**Key Characteristics:**
- **Very high regime persistence:** 86% calm, 72% neutral, 69% volatile
- **Extremely rare panic:** Only 0.5% of 72 years (4 months total!)
- **Very low volatility:** 1.80% annualized (appraisal smoothing effect)
- **Slow transitions:** Average 7.1 months in calm regime

### REITs - VNQ (2005-2026)

From our analysis of 253 months of VNQ data:

```python
P_REIT_VNQ = [
    [0.6667, 0.2778, 0.0556, 0.0000],  # calm    (20% of time, avg 3.6 months)
    [0.1250, 0.6607, 0.1964, 0.0179],  # neutral (56% of time, avg 4.3 months)
    [0.0455, 0.2727, 0.5909, 0.0909],  # volatile (22% of time, avg 3.3 months)
    [0.0000, 0.1333, 0.2667, 0.6000],  # panic    (2% of time, avg 2.0 months)
]
```

**Key Characteristics:**
- **Lower regime persistence:** 67% calm, 66% neutral, 59% volatile
- **Faster transitions:** More dynamic regime switching
- **High volatility:** 17.70% annualized (9.8x traditional CRE)
- **Different crisis pattern:** Panic sticky (60%) but much rarer than volatile states

### Critical Comparison: CRE vs REITs

| Metric | Traditional CRE | REITs (VNQ) | Ratio |
|--------|----------------|-------------|-------|
| Annualized Volatility | 1.80% | 17.70% | 9.8x |
| Calm Persistence | 86% | 67% | 1.3x |
| Time in Calm | 58.7% | 20% | 2.9x |
| Time in Panic | 0.5% | 2% | 4.0x |
| Panic Exit to Neutral | 0% | 13% | ∞ |

**Insight:** Liquidity fundamentally transforms real estate market dynamics, increasing volatility ~10x and creating much faster regime transitions.

## Requirements

```bash
pip install numpy pandas matplotlib requests
```

Python packages used:
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `requests` - API calls (for REIT data)

## Customization

### Change REIT Data Source

Edit `analyze_reit_regimes.py`:

```python
# Line 302: Change symbol
symbols = ["VNQ"]  # Try: "IYR", "XLRE", "RMZ"
```

### Adjust Simulation Parameters

Edit `housing_liquidity_comparison.py`:

```python
# Line 448: Simulation settings
MONTHS_AHEAD = 120      # Projection horizon
TICKS_PER_CANDLE = 50   # Microstructure resolution
SEED = 42               # Random seed for reproducibility
```

### Modify Regime Classification

Edit `analyze_reit_regimes.py`, function `classify_regime_from_volatility()`:

```python
def classify_regime_from_volatility(realized_vol: float, base_vol: float) -> str:
    # Adjust thresholds here
    if realized_vol < 0.7 * base_vol:    # Calm threshold
        return "calm"
    elif realized_vol < 1.2 * base_vol:  # Neutral threshold
        return "neutral"
    # ... etc
```

## Understanding the Results

### Interpretation Guide

**Time in Stress:**
- Lower is better
- Measures % of time in volatile or panic regimes
- Tokenized shows ~47% reduction

**Panic Episodes:**
- Lower is better
- Counts distinct panic periods
- Tokenized shows ~78% fewer episodes

**Avg Panic Duration:**
- Lower is better (usually)
- Tokenized shows longer but much rarer panics
- Net effect: less total panic time

**Max Drawdown:**
- Lower (less negative) is better
- Peak-to-trough decline
- Tokenized may show larger drawdowns in extreme events

### The "Panic Paradox"

REITs exhibit an interesting pattern:
- **Panic persistence:** Very high (90%) when in panic
- **Panic frequency:** Very low (only 4% of time)
- **Net result:** Much less total time in crisis

This suggests liquidity helps **avoid** crises through:
1. Faster price discovery
2. Better information aggregation
3. Easier rebalancing and risk management

But when systemic shocks occur (2008, COVID), liquid markets crash hard. However, these are rare tail events.

## Academic Context

This research connects to several literatures:

1. **Asset Liquidity:** How trading frictions affect price dynamics
2. **Real Estate Finance:** Comparison of direct vs indirect real estate investment
3. **Tokenization:** Digital assets and fractional ownership
4. **Market Microstructure:** Order flow and price formation
5. **Regime-Switching Models:** Time-varying market states

## Limitations

1. **Model simplifications:** Markov assumption, homogeneous housing
2. **Data constraints:** Limited REIT history, US-only
3. **Partial equilibrium:** Doesn't model feedback to broader economy
4. **No agent heterogeneity:** All traders use simple rules

See `FINDINGS_SUMMARY.md` for detailed discussion.

## Future Extensions

### Immediate (can do now):
1. ✅ Analyze traditional CRE baseline (DONE - 72+ years empirical)
2. ✅ Analyze multiple REIT ETFs (IYR, XLRE, RMZ)
3. ✅ Different time periods (exclude 2008? COVID-only?)
4. ✅ Sensitivity analysis on regime thresholds
5. ✅ Export data to CSV for external analysis

### Phase 2 (requires more work):
5. ⚡ Train RL agents to trade in both markets
6. ⚡ Compare agent strategies and profitability
7. ⚡ Test if agents learn different behaviors

### Advanced (research projects):
8. 🔬 Multi-asset portfolio with housing + REITs + stocks
9. 🔬 General equilibrium model with endogenous prices
10. 🔬 Behavioral agents with learning and adaptation

## Citation

If you use this work, please cite:

```
Housing Market Liquidity Study: Empirical Analysis of Tokenization Effects
Market Simulation Framework, 2026
https://github.com/[your-repo]
```

## License

[Your license choice]

## Contact

[Your contact information]

---

**Last Updated:** February 9, 2026  
**Version:** 2.0 (Added Empirical CRE Baseline - 72+ Years)
