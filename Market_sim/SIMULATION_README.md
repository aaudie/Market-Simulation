# Market Simulation — Full Technical Documentation

---

## 1. Problem Being Solved

Traditional commercial real estate (CRE) is structurally illiquid. Property cannot be sold in hours or days; transactions take weeks or months, bid-ask spreads are wide, and market participants operate with limited price transparency. This illiquidity means that when stress arrives, recovery is slow — panic states persist and there is no mechanism to rapidly return to calm.

**The central question this simulation addresses:**

> Does tokenization of commercial real estate improve market health — specifically, does it reduce the time spent in volatile and panic regimes, accelerate recovery from stress, and shift the long-run distribution of market states toward calm?

The simulation answers this by modelling the same underlying asset (CRE) under two market structures:

- **Traditional CRE** — illiquid, wide spreads, slow price discovery, empirically calibrated from 72+ years of data (1953–2025).
- **Tokenized CRE** — continuous trading, tighter spreads, faster price discovery, calibrated from REIT/VNQ data (2005–2026) as a liquid proxy.

Tokenization is not modelled as a sudden switch. It is treated as a gradual adoption process, and the simulation sweeps across every adoption level from 0% to 100% to show how market health metrics evolve continuously.

---

## 2. What Tokenization Does in the Simulation

Tokenization manifests through **three distinct mechanisms**:

### 2.1 The Transition Matrix Interpolation (Core Mechanism)

The most fundamental effect is encoded in the **regime transition matrix** `P(α)`:

```
P(α) = (1 − α) × P_TRADITIONAL  +  α × P_TOKENIZED
```

where `α ∈ [0, 1]` is the tokenization adoption level.

- At `α = 0`, the market behaves like illiquid traditional CRE.
- At `α = 1`, it behaves like a liquid REIT market.
- At intermediate values, the transition probabilities are a weighted blend.

This means tokenization directly changes **how likely the market is to stay in panic**, **how fast it escapes stress**, and **how often it sits in calm**. These are not assumed — they are derived analytically from the two empirically calibrated matrices.

### 2.2 The Adoption Sigmoid (Dynamic Over Time)

The adoption level `α` is not fixed — it evolves over simulation time following a **logistic sigmoid curve** fitted to real-world RWA (Real World Asset) tokenization TVL data (May 2023 – March 2026):

```
α(t) = 1 / (1 + exp(−k × (t − t₀)))
```

Fitted parameters from actual RWA data:
- `L` = saturation level (fitted to observed TVL trajectory)
- `k` = monthly growth rate
- `t₀` = inflection month (point of fastest growth)

This grounds the simulation in **empirical adoption data** rather than assumption.

### 2.3 The Anchor Weight Mechanism (Microstructure Level)

Inside the live microstructure simulation, tokenization manifests as the `anchor_weight` parameter:

```python
anchor_weight = max(anchor_floor, 1.0 − adoption)
```

- **High `anchor_weight` (traditional):** Traders place quotes tightly anchored to the fundamental (appraised) price. Price discovery is slow because all participants reference the same illiquid fundamental.
- **Low `anchor_weight` (tokenized):** Traders blend the fundamental price with the live microstructure price. The market self-discovers price through order flow rather than relying on appraisals.

In practice, as `adoption` rises over months, traders gradually shift from fundamental-anchored quoting to microstructure-driven quoting — exactly how a market transitions from OTC/appraisal-based trading to continuous exchange-based trading.

---

## 3. Exogenous vs Endogenous Variables

### Exogenous (given to the simulation, not determined inside it)

| Variable | Description | Source |
|----------|-------------|--------|
| `P_TRADITIONAL` | Transition matrix for illiquid CRE | Empirical — 72 years of CRE data (1953–2025) |
| `P_TOKENIZED` | Transition matrix for liquid CRE | Empirical — VNQ ETF (2005–2026); Bayesian CRE pooled from O, NNN, WPC, ADC |
| `mu_monthly` | Expected monthly drift of the fundamental price | Calibrated from CRE historical CSV via log-return mean |
| `sigma_monthly` (base) | Base monthly volatility | Calibrated from CRE historical CSV via log-return std dev |
| Historical price path | Actual CRE price series for the replay phase | Loaded from `data/cre_monthly.csv` |
| RWA TVL data | Real-world tokenization adoption trajectory | `data/rwa-token-timeseries-export-*.csv` |
| `adoption_speed` | How fast α rises per month | Set in `MarketSimulator` (default: 0.15) |
| `adoption_midpoint` | Month at which α ≈ 0.5 | Set in `MarketSimulator` (default: month 24) |
| Risk-free rate `r` | Used in Merton portfolio optimization | Hardcoded at 2% annual |
| Risk aversion `γ` | Merton utility parameter | Hardcoded at 3.0 |

### Endogenous (computed inside the simulation)

| Variable | Description | How it emerges |
|----------|-------------|----------------|
| `adoption` | Current tokenization level α(t) | Computed each month from sigmoid |
| `anchor_weight` | How much traders anchor to fundamental | `1 − adoption`, updated each candle roll |
| `regime` | Current market state (calm/neutral/volatile/panic) | Classified from realized rolling volatility |
| `current_sigma_monthly` | Effective volatility this period | `base_sigma × regime_multiplier` |
| Microstructure price | Live traded price from order book | Emerges from order matching between agents |
| Fundamental price | The GBM projection of underlying CRE value | Driven by `mu_monthly` and `current_sigma_monthly` |
| Order book state | Bids, asks, best bid/ask, imbalance | Emerges from agent quoting and trading activity |
| Regime sequence | The realized path of states over time | Emerges from transition matrix + volatility feedback |
| Stationary distribution π | Long-run fraction of time in each regime | Solved analytically from `P(α)` |
| Mean first passage times | Expected months to move between states | Solved analytically from `P(α)` and π |
| Merton optimal weight `w*` | Optimal portfolio allocation to CRE | `(μ − r) / (γ σ²)`, computed post-calibration |

---

## 4. Regime Classification: What Makes a State

Regimes are classified using **rolling realized volatility** compared to a **full-sample baseline volatility**.

### Step 1 — Compute log returns
```
r_t = log(P_t / P_{t-1})
```

### Step 2 — Compute rolling volatility
A 6-month rolling standard deviation of log returns is computed at each time step:
```
σ_realized(t) = std(r_{t-5}, r_{t-4}, ..., r_t)
```

### Step 3 — Compute baseline volatility
The full-sample standard deviation of all log returns is computed once:
```
σ_base = std(all log returns)
```

### Step 4 — Classify the regime

| Regime | Condition | Volatility Multiplier Applied |
|--------|-----------|-------------------------------|
| **Calm** | `σ_realized < 0.70 × σ_base` | 0.8× (quieter than normal) |
| **Neutral** | `0.70 × σ_base ≤ σ_realized < 1.20 × σ_base` | 1.0× (normal) |
| **Volatile** | `1.20 × σ_base ≤ σ_realized < 2.00 × σ_base` | 1.5× (elevated stress) |
| **Panic** | `σ_realized ≥ 2.00 × σ_base` | 2.2× (crisis-level stress) |

### Hybrid Regime (when Markov is enabled)

When the Markov chain is active, a **conservative hybrid** rule is used. The Markov chain provides the macro regime signal (exogenous persistence/shock history), while realized volatility can independently trigger a higher state. The rule is:

```
final_regime = more severe of { markov_regime, realized_regime }
```

The Markov chain can hold the market in a bad state even if realized vol temporarily drops (persistence). Realized vol can push into a worse state even if the Markov macro state is calm. The system never downgrades severity based on Markov alone.

---

## 5. Simulation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MarketSimulator                          │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  Fundamental     │    │  Regime Engine               │   │
│  │  Price (GBM)     │───▶│  - Rolling vol → regime      │   │
│  │  + History Replay│    │  - Markov chain (optional)   │   │
│  └──────────────────┘    │  - Hybrid combiner           │   │
│           │               └──────────────────────────────┘   │
│           │                           │                       │
│           ▼                           ▼                       │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │  Adoption Curve  │    │  current_sigma_monthly       │   │
│  │  α(t) → sigmoid  │    │  = base_sigma × multiplier   │   │
│  │  anchor_weight   │    └──────────────────────────────┘   │
│  └──────────────────┘                                         │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                 Order Book (Microstructure)          │     │
│  │  1000 rule-based market-making agents                │     │
│  │  - Quote bids/asks around blended ref price          │     │
│  │    ref = anchor_weight × fundamental + (1-w) × micro │     │
│  │  - Inventory-skewed spreads                          │     │
│  │  - 10% chance of market order per active tick        │     │
│  │  - Price-time priority matching                      │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Time Structure

Each simulation "month" consists of:
1. **Micro ticks** — agents place orders, order book matches, candlestick updates (`run_micro_ticks`)
2. **Candle roll** — month counter increments, adoption updates, fundamental steps, regime updates (`roll_candle`)

The simulation runs in two phases:
- **Replay phase** — fundamental price follows historical data exactly.
- **Projection phase** — fundamental price follows GBM with regime-adjusted `σ`.

---

## 6. The Three-Layer Quantitative Framework

### Layer 1 — Analytical Markov Chain (`analytical_markov.py`)

Exact mathematical results derived from the transition matrices. No simulation randomness.

| Output | Method |
|--------|--------|
| Stationary distribution π | Solve `π P = π`, `Σπ = 1` as a linear system |
| Mean first passage times M[i,j] | Fundamental matrix `Z = (I − P + Π)⁻¹` |
| Expected sojourn times | `1 / (1 − P[k,k])` per state |
| 95% bootstrap CIs on π | 10,000 Dirichlet resamples of transition counts |

Key comparisons between Traditional and Tokenized:
- What fraction of time is spent in stress states (volatile + panic)?
- How many months to escape panic back to neutral?
- How long does a typical panic episode last per visit?

### Layer 2 — Monte Carlo Analysis (`monte_carlo_analysis.py`)

5,000 simulated paths × 240 months (20 years) per path. Produces distributional results across the full ensemble.

Outputs: price distributions, regime occupancy over time, t-tests comparing Traditional vs Tokenized outcomes.

### Layer 3 — Adoption Sensitivity (`adoption_sensitivity.py`)

**Part A:** Fits a logistic sigmoid to real RWA tokenized real estate TVL data to estimate `α(t)` empirically. Determines where the market sits today and at +2yr / +5yr horizons on the adoption curve.

**Part B:** Sweeps α from 0 to 1 in 200 steps. At each α, computes the interpolated `P(α)` and solves analytically for all Markov metrics. Shows exactly how each outcome (stress time, panic sojourn, recovery speed) changes as tokenization adoption increases.

---

## 7. Bayesian CRE Transition Matrix (`bayesian_cre_transition.py`)

A purpose-built Bayesian estimator that produces a single **holistic** transition matrix from multiple CRE REIT sources.

### Why pooling multiple REITs?

A single ticker (e.g. VNQ) has noisy estimates because:
- Idiosyncratic shocks inflate certain row counts
- Some transitions (e.g. calm → panic) are rare and poorly estimated

By pooling transition counts across multiple REITs from the **same asset class**, the estimator uses the same latent market state (CRE cycle) observed through multiple instruments simultaneously.

### REIT Basket (Net Lease CRE)

| Ticker | Name | Rationale |
|--------|------|-----------|
| `O` | Realty Income | Largest net lease REIT, 20+ yr history |
| `NNN` | NNN REIT | Pure-play triple-net, deepest comparable history |
| `WPC` | W. P. Carey | Diversified net lease, stable long-term series |
| `ADC` | Agree Realty | Higher-quality tenant mix, independent confirmation |

All four operate the same business model (long-term triple-net commercial leases) and respond to the same macro/credit/cap-rate drivers. This makes their shared latent state a clean signal for the CRE asset class regime.

### Model

Each REIT's price series is independently classified into regimes using rolling volatility. The transition counts from all REITs are then **pooled**:

```
N_pooled[k,j] = Σ_i  count(ticker i:  state k → state j)
```

A **sticky Dirichlet prior** is placed on each row of the transition matrix:

```
alpha[k, k]   = 5.0   (diagonal — regimes tend to persist)
alpha[k, j≠k] = 1.0   (off-diagonal — uninformative)
```

The posterior is closed-form (Dirichlet-Multinomial conjugate):

```
P[k, ·] | data  ~  Dirichlet(alpha[k, ·] + N_pooled[k, ·])
```

The **posterior mean** is used as the holistic transition matrix. The model also produces **95% credible intervals** for every cell via 5,000 posterior samples.

### Output

Saves `outputs/bayesian_cre_transition.npz` with:
- `P_mean` — posterior mean matrix (drop-in for `RegimeMarkovChain`)
- `P_lower`, `P_upper` — 95% credible interval bounds
- `samples` — full posterior sample array for downstream Monte Carlo use
- `pooled_counts` — raw pooled transition counts

---

## 8. Data Sources

| File | Contents | Used By |
|------|----------|---------|
| `data/cre_monthly.csv` | Monthly CRE price index, 1953–2025 (872 observations) | Calibration, regime analysis, historical replay |
| `data/rwa-token-timeseries-export-*.csv` | Daily RWA tokenized real estate TVL, May 2023–Mar 2026 | Layer 3 adoption curve fitting |
| Twelve Data API | Live monthly OHLCV for individual REITs | `analyze_reit_regimes.py`, `bayesian_cre_transition.py` |

---

## 9. Agent Behaviour (Microstructure)

1,000 rule-based market-making agents populate the order book. Each agent:

1. **Activity gate** — only acts on a given tick with probability `trader_activity_rate` (default 5%). Prevents order spam.

2. **Reference price blending** — computes a blended reference:
   ```
   ref = anchor_weight × fundamental_price + (1 − anchor_weight) × micro_price
   ```
   This is the key mechanism by which tokenization adoption changes agent behaviour.

3. **Inventory-skewed quoting** — agents track their inventory position. A long inventory skews quotes downward to encourage selling; short inventory skews upward. This models realistic inventory management.

4. **Spread calculation** — default spread of 10 bps either side of the blended mid.

5. **Two-sided quotes** — each active agent posts both a bid and an ask.

6. **Market order injection** — with 10% probability per activity event, the agent submits a small market order (1–5 units) to provide directional flow and prevent the book from becoming stale.

---

## 10. Portfolio Optimization (Merton Framework)

After calibrating drift and volatility from historical data, the simulation computes the **Merton optimal portfolio weight**:

```
w* = (μ − r) / (γ σ²)
```

| Parameter | Description |
|-----------|-------------|
| `μ` | Calibrated annual expected return of CRE |
| `r` | Risk-free rate (2% annual) |
| `γ` | Risk aversion coefficient (default 3.0) |
| `σ` | Calibrated annual volatility of CRE |

This gives the theoretically optimal fraction of a portfolio to allocate to CRE under log-utility with constant relative risk aversion. It is computed separately for traditional and tokenized scenarios and used as a reference for interpreting simulation outcomes.

---

## 11. Running the Simulation

### Full analysis pipeline
```bash
cd Market_Sim/Market_sim
python3 scripts/run_complete_analysis.py
```
Runs all six steps in sequence and saves all output charts to `outputs/`.

### Individual components
```bash
# Traditional CRE baseline + regime analysis
python3 scripts/analyze_cre_regimes.py

# Single-ticker REIT regime analysis (VNQ)
python3 scripts/analyze_reit_regimes.py

# Bayesian pooled CRE transition matrix (O, NNN, WPC, ADC)
python3 scripts/bayesian_cre_transition.py

# Layer 1: Exact analytical Markov results
python3 scripts/analytical_markov.py

# Layer 2: Monte Carlo (5,000 paths × 20 years)
python3 scripts/monte_carlo_analysis.py

# Layer 3: RWA adoption curve + sensitivity sweep
python3 scripts/adoption_sensitivity.py

# Live microstructure demo (traditional vs tokenized)
python3 main.py
```

### Dependencies
```bash
pip install numpy pandas matplotlib scipy requests
```

---

## 12. Output Files

| File | Produced By | Contents |
|------|-------------|----------|
| `outputs/CRE_regime_analysis.png` | `analyze_cre_regimes.py` | 72-year CRE regime history, transition matrix, durations |
| `outputs/VNQ_regime_analysis.png` | `analyze_reit_regimes.py` | VNQ regime classification, empirical transition matrix |
| `outputs/bayesian_cre_transition.png` | `bayesian_cre_transition.py` | Posterior mean matrix, CI widths, per-REIT breakdown |
| `outputs/bayesian_cre_transition.npz` | `bayesian_cre_transition.py` | Posterior samples, P_mean, P_lower, P_upper |
| `outputs/layer1_analytical_markov.png` | `analytical_markov.py` | Stationary distributions, MFPT heatmaps, sojourn times |
| `outputs/layer2_monte_carlo.png` | `monte_carlo_analysis.py` | Price path distributions across 5,000 runs |
| `outputs/layer2_regime_occupancy.png` | `monte_carlo_analysis.py` | Regime occupancy fractions over simulation horizon |
| `outputs/layer3_adoption_curve.png` | `adoption_sensitivity.py` | Empirical RWA TVL with fitted logistic curve |
| `outputs/layer3_sensitivity_sweep.png` | `adoption_sensitivity.py` | All market metrics vs α from 0 to 1 |
| `outputs/housing_liquidity_comparison.png` | `housing_liquidity_comparison.py` | Single-path traditional vs tokenized price comparison |
| `outputs/hybrid_markov_model.npz` | Various | Saved hybrid model state |
