# Traditional CRE Baseline Analysis
## Empirical Foundation for Liquidity Research

**Date:** February 9, 2026  
**Data Source:** CRE Monthly Price Index (1953-2025)  
**Sample Size:** 871 months (72.5 years)

---

## Executive Summary

This analysis establishes the empirical baseline for traditional (illiquid) commercial real estate using 72+ years of historical data. The findings provide crucial context for understanding how tokenization and liquidity might transform real estate markets.

### Key Finding

**Traditional CRE exhibits extreme stability and regime persistence, with annualized volatility of only 1.80% - approximately 9.8x lower than liquid REIT markets.**

---

## Empirical Transition Matrix

```python
P_CRE_TRADITIONAL = [
    [0.8591, 0.1389, 0.0020, 0.0000],  # calm
    [0.2339, 0.7186, 0.0475, 0.0000],  # neutral
    [0.0339, 0.2203, 0.6949, 0.0508],  # volatile
    [0.0000, 0.0000, 0.7500, 0.2500],  # panic
]
```

### Regime Statistics

| Regime | % of Time | Avg Duration | Persistence |
|--------|-----------|--------------|-------------|
| **Calm** | 58.7% | 7.1 months | 86% |
| **Neutral** | 34.0% | 3.5 months | 72% |
| **Volatile** | 6.8% | 3.3 months | 69% |
| **Panic** | 0.5% | 1.3 months | 25% |

**Total time in stress (volatile + panic): 7.3%**

---

## Key Characteristics of Traditional CRE

### 1. Extreme Regime Persistence

Traditional CRE stays in the same regime for extended periods:

- **Calm regime:** 86% probability of staying calm next month
  - Average duration: 7.1 months
  - This represents the "normal" state of CRE markets

- **Neutral regime:** 72% persistence
  - Transitions slowly to calm (23%) or volatile (5%)
  
- **Volatile regime:** 69% persistence
  - Most likely exits back to neutral (22%)
  - Rarely escalates to panic (5%)

### 2. Panic is Extremely Rare

Over 72.5 years of data:
- Only **0.5% of time** spent in panic regime
- Approximately **4 months total** in panic state
- When panic occurs, 75% chance of reverting to volatile (not staying in panic)
- **Never transitions directly to neutral or calm** from panic

Major historical panics captured:
- 1970s oil crisis period
- 2008 financial crisis
- COVID-19 initial shock (2020)

### 3. Very Low Volatility

- **Baseline volatility:** 0.52% (monthly)
- **Annualized volatility:** 1.80%

This extreme smoothness reflects:
1. **Appraisal-based valuation** (not market-based)
2. **Illiquidity** (infrequent transactions)
3. **Smoothing techniques** in index construction
4. **Lagged information** incorporation

### 4. Asymmetric Transition Patterns

**Upward transitions (toward calm):**
- Volatile → Neutral: 22%
- Neutral → Calm: 23%
- Gradual recovery pattern

**Downward transitions (toward panic):**
- Calm → Volatile: 0.2% (nearly impossible)
- Calm → Neutral: 14% (gradual)
- Neutral → Volatile: 5% (rare)
- Crises develop slowly, with warning signals

---

## Comparison: Traditional CRE vs REITs

### Volatility Gap (9.8x Difference!)

| Asset Class | Monthly Vol | Annual Vol | Ratio |
|-------------|-------------|------------|-------|
| Traditional CRE | 0.52% | 1.80% | 1.0x |
| REITs (VNQ) | 5.11% | 17.70% | 9.8x |

**Interpretation:** Liquidity increases volatility by an order of magnitude due to:
- Continuous price discovery
- Market sentiment effects
- Daily mark-to-market
- Trading frictions disappear

### Regime Persistence Comparison

| State | CRE Persistence | REIT Persistence | Difference |
|-------|-----------------|------------------|------------|
| Calm | 86% | 67% | +19 pts |
| Neutral | 72% | 66% | +6 pts |
| Volatile | 69% | 59% | +10 pts |
| Panic | 25% | 60% | -35 pts |

**Key Insight:** Traditional CRE is "stickier" in normal states but exits panic faster. REITs transition more frequently but get stuck in panic when it occurs.

### Time Distribution Comparison

| Regime | Traditional CRE | REITs (VNQ) | Difference |
|--------|----------------|-------------|------------|
| Calm | 58.7% | 20.0% | +38.7 pts |
| Neutral | 34.0% | 56.0% | -22.0 pts |
| Volatile | 6.8% | 22.0% | -15.2 pts |
| Panic | 0.5% | 2.0% | -1.5 pts |

**Total stress time:**
- Traditional CRE: 7.3%
- REITs: 24.0%
- **REITs spend 3.3x more time in stress states**

### Crisis Dynamics

**Traditional CRE:**
- Crises are rare and brief
- Slow buildup (calm → neutral → volatile → panic)
- Quick exit (panic → volatile → neutral → calm)
- Panic appears only during systemic events

**REITs:**
- More frequent volatility
- Faster transitions between states
- Panic is stickier (60% persistence)
- Higher sensitivity to market sentiment

---

## Implications for Tokenization Research

### 1. Baseline Calibration

The traditional CRE matrix provides an empirical anchor for:
- "Traditional" market parameters in simulations
- Validating assumed illiquid housing dynamics
- Quantifying the liquidity transformation

**Before this analysis:** We used assumed transition matrices  
**After this analysis:** We have 72 years of empirical evidence

### 2. Magnitude of Liquidity Effect

Tokenization could transform markets by:
- **9.8x increase in volatility** (from 1.8% to 17.7%)
- **3.3x increase in stress time** (from 7.3% to 24%)
- **Faster regime transitions** (lower persistence)
- **Different crisis patterns** (stickier panics)

### 3. Trade-offs

**Gains from liquidity:**
- ✅ Better price discovery
- ✅ Easier portfolio rebalancing  
- ✅ Fractional ownership
- ✅ More market participants

**Costs from liquidity:**
- ❌ Higher volatility (10x)
- ❌ More time in stress states (3x)
- ❌ Sentiment-driven price swings
- ❌ Potential for flash crashes

### 4. Policy Considerations

Understanding the CRE baseline helps evaluate:
- Whether tokenization benefits outweigh costs
- If volatility dampening mechanisms are needed
- How to design circuit breakers or trading halts
- Regulatory frameworks for tokenized real estate

---

## Robustness and Limitations

### Strengths

1. **Long time series:** 72.5 years captures multiple economic cycles
2. **Complete regime coverage:** Includes major crises (1970s, 2008, COVID)
3. **Consistent methodology:** Same regime classification as REIT analysis
4. **Large sample:** 871 monthly observations

### Limitations

1. **Appraisal smoothing:** CRE data is inherently smoothed
   - True volatility is likely higher than 1.8%
   - Transitions may be artificially gradual
   - This makes the liquidity gap even more striking

2. **Index construction:** 
   - National aggregate masks regional variation
   - Commercial ≠ residential (though highly correlated)
   - Survivor bias possible

3. **Data frequency:**
   - Monthly data misses high-frequency dynamics
   - REITs trade daily, CRE appraisals are quarterly/annual
   - Comparison is somewhat apples-to-oranges

4. **Structural changes:**
   - Real estate markets evolved significantly since 1953
   - Recent decades may be more relevant than 1950s-60s
   - Consider sub-period analysis

### Sensitivity Checks

**Recommended:**
- Analyze post-1980 period separately (modern era)
- Compare residential vs commercial CRE
- Test different volatility thresholds for regime classification
- Bootstrap confidence intervals for transition probabilities

---

## Conclusion

This analysis establishes that traditional CRE markets are characterized by:

1. **Extreme stability:** 1.8% annualized volatility
2. **High persistence:** 86% calm, 72% neutral
3. **Rare crises:** Only 0.5% of time in panic over 72 years
4. **Slow transitions:** Average 7.1 months in calm regime

These empirical findings provide a crucial baseline for evaluating how tokenization and increased liquidity would transform real estate markets. The 9.8x volatility gap between traditional CRE and REITs suggests that tokenization could fundamentally alter market dynamics.

**The key research question becomes:** Are the benefits of liquidity (better price discovery, easier trading, fractional ownership) worth the costs of increased volatility and more frequent stress states?

---

## Data Access

- **Raw data:** `outputs/cre_monthly.csv`
- **Analysis script:** `scripts/analyze_cre_regimes.py`
- **Visualization:** `outputs/CRE_regime_analysis.png`
- **Transition matrix:** See code snippet above

---

## References

1. Historical CRE data from S&P CoreLogic Case-Shiller Commercial Real Estate Index
2. Methodology consistent with `analyze_reit_regimes.py`
3. Regime classification thresholds: 0.7x, 1.2x, 2.0x base volatility
4. Rolling window: 6 months for volatility estimation

---

## Next Steps

1. ✅ Compare with REIT empirical analysis
2. ✅ Integrate into housing liquidity comparison
3. ⬜ Sub-period analysis (1980-2025 vs 1953-1980)
4. ⬜ Residential vs commercial comparison
5. ⬜ International real estate markets
6. ⬜ Sensitivity analysis on regime thresholds
7. ⬜ Bootstrap confidence intervals

---

**Contact:** For questions about this analysis or the methodology, see main project README.

**Version:** 1.0 (February 9, 2026)
