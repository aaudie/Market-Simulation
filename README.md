# CRE Tokenisation Liquidity Study

**Research question:** To what extent does blockchain-based tokenisation improve liquidity, price discovery, and market stability in commercial real estate, and how do those effects differ from traditional illiquid CRE markets?

---

## Economic Premise

Traditional commercial real estate is one of the most illiquid asset classes in existence. Transactions take months, price discovery is infrequent, and when markets enter distress they tend to stay there — there are no market makers, no continuous quotes, and no mechanism for rapid coordination between buyers and sellers. The result is a market that spends a disproportionate share of time in volatile and panic regimes.

Blockchain-based tokenisation changes this structural feature. By representing fractional ownership of a property as an on-chain token, the asset inherits the liquidity properties of a traded security: continuous price discovery, instant settlement, and a visible order book. The closest existing analogue for what tokenised real estate *might* behave like is the REIT market — specifically, the VNQ Vanguard Real Estate ETF, which has 20 years of empirical data.

This study uses that analogy to answer a concrete question: **if CRE had the liquidity and regime dynamics of REITs, how much better would market outcomes be?**

---

## Methodology

The core model is a **4-state Markov chain** over market regimes: Calm, Neutral, Volatile, and Panic. Each state carries a volatility multiplier; the transition probabilities determine how long the market spends in each regime and how quickly it recovers from distress.

Two transition matrices are used:

**`P_TRADITIONAL`** — assumed illiquid CRE dynamics, characterised by high persistence in all states and a relatively easy path from calm into distress:

```
From \ To    Calm    Neutral  Volatile  Panic
Calm         0.850   0.140    0.010     0.000
Neutral      0.100   0.750    0.140     0.010
Volatile     0.020   0.180    0.700     0.100
Panic        0.010   0.090    0.300     0.600   ← gets stuck
```

**`P_TOKENIZED`** — empirically estimated from 253 months of VNQ ETF data (2005–2026). Represents a liquid, continuously-traded real estate market:

```
From \ To    Calm    Neutral  Volatile  Panic
Calm         0.817   0.174    0.009     0.000
Neutral      0.189   0.774    0.028     0.009
Volatile     0.050   0.200    0.750     0.000   ← can't fall into panic
Panic        0.000   0.000    0.100     0.900   ← rare but sticky
```

The key structural difference: in the tokenised matrix, panic is nearly unreachable from volatile (0.000 probability), so crises are rare. When they do occur (driven by systemic shocks like 2008 or COVID) they are severe and persistent — but the overall frequency is so much lower that the net outcome strongly favours liquidity.

The calibrated base parameters (from CRE monthly data, 1953–2025):
- Monthly drift μ = 0.08% (0.96% annualised)
- Monthly volatility σ = 0.52% (1.80% annualised)

---

## Three-Layer Quantitative Analysis

### Layer 1 — Analytical Markov Chain Results

Rather than relying purely on simulation, the analytical properties of the two chains are derived exactly using linear algebra. These results are mathematically certain given the matrices.

**Stationary distributions** (long-run % time in each regime):

| Regime | Traditional | Tokenised | Source |
|---|---|---|---|
| Calm | 29.0% | 45.8% | Exact (eigenvector) |
| Neutral | 37.6% | 42.2% | Exact |
| Volatile | 25.9% | 8.0% | Exact |
| Panic | 7.4% | 4.0% | Exact |
| **Healthy (C+N)** | **66.6%** | **88.1%** | |
| **Stressed (V+P)** | **33.4%** | **11.9%** | |

Bootstrap confidence intervals (10,000 resamples from observed regime counts) confirm these differences are well outside sampling uncertainty.

**Mean first passage times** — the expected number of months to first reach a state:

| Transition | Traditional | Tokenised |
|---|---|---|
| Calm → Panic | 43.7 months | **242.4 months** |
| Neutral → Panic | 37.6 months | **236.7 months** |
| Panic → Neutral | 7.4 months | 15.1 months |
| Panic → Calm | 21.1 months | 19.1 months |

The most striking result: in a tokenised market, reaching panic from calm requires on average **242 months (20 years)** of bad luck, versus only 44 months in the traditional market. Crises don't disappear — they become structurally rare.

**Expected sojourn times** (months per visit):

| Regime | Traditional | Tokenised |
|---|---|---|
| Calm | 6.7 mo | 5.5 mo |
| Neutral | 4.0 mo | 4.4 mo |
| Volatile | 3.3 mo | 4.0 mo |
| Panic | **2.5 mo** | **10.0 mo** |

The panic sojourn is longer in the tokenised case — consistent with the 2008 and COVID data embedded in VNQ. Crises are rarer but deeper. The net effect is still strongly positive because entry into crisis is so much harder.

---

### Layer 2 — Monte Carlo Analysis

To convert the analytical results into distributional statistics with confidence intervals and significance tests, 5,000 independent paths of 240 months (20 years) were simulated for each matrix. Each path draws regime sequences from the Markov chain and generates GBM prices conditioned on regime volatility.

**Results (mean ± 95% CI across 5,000 paths):**

| Metric | Traditional | Tokenised | p-value |
|---|---|---|---|
| Time in stress (%) | 48.4 [48.2, 48.6] | 16.2 [16.0, 16.4] | p < 0.001 |
| Panic episodes | 19.4 [19.3, 19.5] | 0.95 [0.92, 0.97] | p < 0.001 |
| Avg panic duration (mo) | 2.50 [2.49, 2.51] | 6.28 [6.05, 6.52] | p < 0.001 |
| Max drawdown (%) | −7.9 [−8.0, −7.8] | −5.4 [−5.4, −5.3] | p < 0.001 |
| Annualised volatility (%) | 2.56 [2.55, 2.56] | 1.91 [1.91, 1.92] | p < 0.001 |
| **Total return (%)** | **21.2 [20.8, 21.5]** | **21.1 [20.8, 21.4]** | **p = 0.922** |

The final row is a key economic finding: **total returns are statistically identical** (p = 0.922). The benefit of tokenisation is entirely structural — it reduces time in crisis by 66% and panic episodes by 95%, without sacrificing long-run return. Investors are not asked to accept lower performance in exchange for stability; they get stability for free.

---

### Layer 3 — Adoption Sensitivity Analysis

Rather than treating tokenisation as binary (on/off), this layer asks: *how much does each incremental increase in tokenisation adoption improve market outcomes?*

An interpolated matrix is defined as:

```
P(α) = (1 − α) × P_TRADITIONAL + α × P_TOKENIZED
```

where α ∈ [0, 1] represents the adoption level — 0 is fully traditional, 1 is fully tokenised. The stationary distribution and mean first passage times are computed analytically for every value of α, producing continuous sensitivity curves.

**Selected outcomes at key adoption levels:**

| α | Interpretation | Stress time | Panic→Neutral |
|---|---|---|---|
| 0.00 | Traditional CRE | 33.4% | 7.4 months |
| 0.25 | Early adoption | ~27% | ~9 months |
| 0.50 | Halfway | ~21% | ~11 months |
| 0.75 | Mature market | ~16% | ~13 months |
| 1.00 | Full REIT-like liquidity | 11.9% | 15.1 months |

The relationship is nonlinear: the largest gains occur between α = 0 and α = 0.40, where the probability of entering panic from volatile collapses toward zero. Beyond that point, further adoption provides diminishing marginal improvements to stress time while modestly increasing panic persistence (the REIT panic-stickiness effect).

**Empirical adoption grounding (RWA data):**

The study uses on-chain data on tokenised real estate TVL (total value bridged, May 2023 – March 2026) to empirically ground the adoption trajectory. A logistic sigmoid was fitted to the data:

- Saturation level (first wave): ~$468 million
- Inflection point: October 2025 (month 24.5 of the series)
- Monthly growth rate: k = 0.357 (doubling time ~1.9 months at peak growth)

The TVL grew from $2.1 million to $441 million over 35 months — a 20,800% increase. The fitted sigmoid shows this first wave of adoption approaching saturation by early 2026, with the current normalised adoption level α ≈ 0.97 for this market segment. **(Note to self: The first wave isn't truly approaching saturation through natural causes, it's slowed down only because we are facign stagflation so very little capital is available to be put into CRE markets let alone tokenized CRE markets)**

This is interpreted not as "CRE is 97% tokenised" (the total US CRE market is ~$20 trillion; $441 million is a rounding error) but rather as evidence that **the institutional infrastructure for tokenised real estate has matured rapidly**, and the regime dynamics embedded in REITs are now reachable as a structural model for what broader adoption would produce.

---

## Empirical Data Sources

| Dataset | Coverage | Use |
|---|---|---|
| CRE Monthly Index | 1953–2025, 871 months | Base calibration (μ, σ), traditional regime classification |
| VNQ ETF | 2005–2026, 253 months | Empirical tokenised transition matrix |
| RWA Bridged Token Value | May 2023–Mar 2026, 1,065 daily obs. | Adoption curve fitting |

---

## Running the Analysis

**Full pipeline (recommended):**
```bash
cd Market_Sim/Market_sim
pip install numpy pandas matplotlib scipy requests
python3 scripts/run_complete_analysis.py
```

This runs all six steps in sequence and saves all output charts to `outputs/`.

**Individual layers:**
```bash
# Layer 1: Analytical results (stationary distributions, passage times, bootstrap CIs)
python3 scripts/analytical_markov.py

# Layer 2: Monte Carlo (5,000 paths, distributional statistics, t-tests)
python3 scripts/monte_carlo_analysis.py

# Layer 3: Adoption sensitivity + RWA sigmoid fit
python3 scripts/adoption_sensitivity.py

# Empirical foundations
python3 scripts/analyze_cre_regimes.py
python3 scripts/analyze_reit_regimes.py

# Single-run benchmark comparison
python3 scripts/housing_liquidity_comparison.py
```

---

## Output Files

| File | Layer | Description |
|---|---|---|
| `layer1_analytical_markov.png` | 1 | Stationary distributions, MFPT heatmaps, sojourn times |
| `layer2_monte_carlo.png` | 2 | Distribution histograms for all metrics |
| `layer2_regime_occupancy.png` | 2 | Average regime occupancy over the 20-year horizon |
| `layer3_adoption_curve.png` | 3 | Observed RWA TVL with fitted logistic curve |
| `layer3_sensitivity_sweep.png` | 3 | Outcome sensitivity curves across all α values |
| `CRE_regime_analysis.png` | — | 72-year CRE regime baseline |
| `VNQ_regime_analysis.png` | — | VNQ empirical regime classification |
| `housing_liquidity_comparison.png` | — | Single-run price path comparison |

---

## Project Structure

```
Market_Sim/
├── Market_sim/
│   ├── scripts/
│   │   ├── run_complete_analysis.py      # Full pipeline runner
│   │   ├── analytical_markov.py          # Layer 1: exact analytical results
│   │   ├── monte_carlo_analysis.py       # Layer 2: distributional MC results
│   │   ├── adoption_sensitivity.py       # Layer 3: RWA sigmoid + α-sweep
│   │   ├── analyze_cre_regimes.py        # CRE baseline (72 years)
│   │   ├── analyze_reit_regimes.py       # VNQ empirical matrix
│   │   └── housing_liquidity_comparison.py  # Single-run benchmark
│   │
│   ├── sim/                              # Core simulation engine
│   │   ├── market_simulator.py           # Order book + GBM + regime switching
│   │   ├── regimes.py                    # Markov chain logic
│   │   ├── microstructure.py             # Order book matching engine
│   │   ├── calibration.py                # μ and σ estimation from history
│   │   ├── portfolio.py                  # Merton optimal weight
│   │   ├── data_loader.py                # CSV loaders
│   │   └── agents/rule_based.py          # Limit-order trading agents
│   │
│   ├── data/
│   │   ├── cre_monthly.csv                              # CRE 1953–2025
│   │   └── rwa-token-timeseries-export-*.csv            # RWA TVL 2023–2026
│   │
│   └── outputs/                          # All generated charts and reports
│
├── future_work/                          # Deferred components
│   ├── dmm/                              # Deep Markov Model (needs more data)
│   ├── qfclient/                         # Multi-provider market data client
│   └── README.md                         # Conditions for reintroduction
│
└── notebooks/
    └── markov_chains.ipynb               # Exploratory Markov analysis
```

---

## Limitations and Assumptions

1. **P_TRADITIONAL is assumed, not empirical.** No liquid daily price series exists for the illiquid CRE market as a whole. The traditional matrix encodes theoretical illiquidity — high panic persistence, moderate stress entry probabilities.

2. **VNQ as a proxy.** REITs are professionally managed, SEC-regulated, and have operated through two major crises (2008 and COVID). Real tokenised CRE may behave differently, particularly in early adoption phases before deep secondary markets form.

3. **Markov assumption (memorylessness).** The model assumes regime transitions depend only on the current state, not on how long the market has been there. Real markets exhibit path dependency.

4. **Returns are total-return-neutral by construction.** The simulation uses the same μ and σ base for both matrices. In practice, tokenisation might raise or lower fundamental returns through changes in cost of capital, liquidity premium, and transaction costs.

5. **RWA data reflects a single market segment.** The $441 million TVL represents one early-stage platform. The US CRE market is ~$20 trillion. The adoption curve captures first-wave institutional infrastructure maturation, not economy-wide penetration.

---

## Citation

```
CRE Tokenisation Liquidity Study
Analytical Markov chain and Monte Carlo analysis of regime dynamics
Data: VNQ (Vanguard Real Estate ETF) 2005–2026; CRE Monthly Index 1953–2025;
      RWA Bridged Token Value 2023–2026
Analysis date: March 2026
```
