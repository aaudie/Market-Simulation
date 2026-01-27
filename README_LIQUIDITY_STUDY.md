# Housing Market Liquidity Study

**Research Question:** What would a US housing market with REIT-like liquidity look like?

## Quick Start

### 1. Run the Complete Analysis

```bash
# Step 1: Analyze real REIT data to extract transition matrix
python3 analyze_reit_regimes.py

# Step 2: Run housing market comparison simulation
python3 housing_liquidity_comparison.py
```

### 2. View Results

The analysis generates:
- `VNQ_regime_analysis.png` - REIT regime transition analysis
- `housing_liquidity_comparison.png` - Traditional vs Tokenized comparison
- `FINDINGS_SUMMARY.md` - Detailed research findings

## Key Finding

**Tokenized housing markets (with REIT-like liquidity) spend 47% less time in crisis states compared to traditional illiquid housing markets.**

- Traditional: 46.9% time in stress (volatile/panic)
- Tokenized: 24.9% time in stress (volatile/panic)

## Files Overview

### Analysis Scripts

1. **`analyze_reit_regimes.py`**
   - Fetches real REIT data (VNQ ETF)
   - Calculates empirical transition probabilities
   - Generates regime visualization
   - Outputs Python-ready transition matrix

2. **`housing_liquidity_comparison.py`**
   - Simulates traditional vs tokenized housing markets
   - Uses empirical REIT transition matrix
   - Generates comprehensive comparison plots
   - Calculates performance metrics

3. **`main.py`**
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

### Step 1: Empirical REIT Analysis

We analyze VNQ (Vanguard Real Estate ETF) to understand how liquid real estate markets behave:

1. Fetch 20+ years of monthly REIT data
2. Calculate rolling volatility (6-month window)
3. Classify into 4 regimes: Calm, Neutral, Volatile, Panic
4. Estimate transition probabilities empirically

**Result:** A 4×4 transition matrix representing real REIT market dynamics

### Step 2: Market Simulation

We simulate two parallel housing markets:

1. **Traditional (Illiquid)**
   - Assumed transition matrix based on housing characteristics
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

### Step 3: Comparison

We compare across multiple dimensions:
- Time spent in each regime
- Number and duration of crisis episodes
- Price volatility and drawdowns
- Recovery times from shocks

## Empirical Transition Matrix (VNQ)

From our analysis of 253 months of VNQ data (2005-2026):

```python
P_REIT = [
    [0.82, 0.17, 0.01, 0.00],  # calm    (46% of time, avg 5.3 months)
    [0.19, 0.77, 0.03, 0.01],  # neutral (42% of time, avg 4.4 months)
    [0.05, 0.20, 0.75, 0.00],  # volatile (8% of time, avg 4.0 months)
    [0.00, 0.00, 0.10, 0.90],  # panic    (4% of time, avg 10 months)
]
```

**Key Insights:**
- REITs spend 88% of time in calm/neutral (vs ~53% for housing)
- Panic is rare (4%) but persistent when it occurs (90%)
- Strong tendency to avoid crisis states

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
1. ✅ Analyze multiple REIT ETFs (IYR, XLRE, RMZ)
2. ✅ Different time periods (exclude 2008? COVID-only?)
3. ✅ Sensitivity analysis on regime thresholds
4. ✅ Export data to CSV for external analysis

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

**Last Updated:** January 20, 2026  
**Version:** 1.0 (Empirical REIT Matrix)
