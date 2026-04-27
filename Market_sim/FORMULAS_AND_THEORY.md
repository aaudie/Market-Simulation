# Mathematical Foundations — Formulas, Variables & Academic Grounding

This document catalogues every key formula used in the simulation, defines its variables, explains the real-world intuition, and connects each to the academic literature that supports it.

---

## Table of Contents

1. [Geometric Brownian Motion (Fundamental Price)](#1-geometric-brownian-motion-fundamental-price)
2. [Historical Calibration (Drift & Volatility)](#2-historical-calibration-drift--volatility)
3. [Regime Classification via Rolling Volatility](#3-regime-classification-via-rolling-volatility)
4. [Regime-Dependent Volatility Scaling](#4-regime-dependent-volatility-scaling)
5. [Markov Chain Regime Dynamics](#5-markov-chain-regime-dynamics)
6. [Transition Matrix Interpolation (Tokenization Effect)](#6-transition-matrix-interpolation-tokenization-effect)
7. [Stationary Distribution](#7-stationary-distribution)
8. [Mean First Passage Times](#8-mean-first-passage-times)
9. [Expected Sojourn Times](#9-expected-sojourn-times)
10. [Logistic Adoption Curve](#10-logistic-adoption-curve)
11. [Anchor Weight Mechanism](#11-anchor-weight-mechanism)
12. [Bayesian Dirichlet–Multinomial Estimation](#12-bayesian-dirichletmultinomial-estimation)
13. [Merton Optimal Portfolio Weight](#13-merton-optimal-portfolio-weight)
14. [Box–Muller Transform](#14-boxmuller-transform)
15. [Microstructure: Order-Flow Imbalance](#15-microstructure-order-flow-imbalance)
16. [Microstructure: Inventory-Skewed Market Making](#16-microstructure-inventory-skewed-market-making)
17. [Monte Carlo Path Metrics](#17-monte-carlo-path-metrics)
18. [Bootstrap & Frequentist Inference](#18-bootstrap--frequentist-inference)
19. [Literature Summary Table](#19-literature-summary-table)

---

## 1. Geometric Brownian Motion (Fundamental Price)

### Formula

$$
\ln S_{t+1} = \ln S_t \;+\; \left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t \;+\; \sigma\sqrt{\Delta t}\;Z_t
$$

equivalently:

$$
S_{t+1} = S_t \;\exp\!\Big[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\;Z_t\Big]
$$

### Variables

| Symbol | Meaning | Units |
|--------|---------|-------|
| \(S_t\) | Fundamental asset price at month \(t\) | USD index |
| \(\mu\) | Expected monthly drift (calibrated from history) | per month |
| \(\sigma\) | Current monthly volatility (regime-adjusted) | per month |
| \(\Delta t\) | Time step (= 1 month in this simulation) | months |
| \(Z_t\) | Standard normal random draw | dimensionless |

### Real-World Logic

GBM is the workhorse model of asset pricing. It assumes that percentage price changes are independent and normally distributed in log-space, which produces the familiar skewed-right distribution of asset prices (prices can't go negative, but can grow without bound). The \(-\tfrac{1}{2}\sigma^2\) term is the **Itô correction**: it ensures that the expected value of \(S_{t+1}\) under the physical measure is \(S_t \cdot e^{\mu \Delta t}\), not inflated by the convexity of the exponential.

In this simulation, \(\sigma\) is not constant — it is scaled by the current regime (calm, neutral, volatile, panic), so the model becomes a **regime-switching GBM**, which is more realistic for real estate.

### Academic Support

- **Samuelson, P. A.** (1965). "Proof that properly anticipated prices fluctuate randomly." *Industrial Management Review*, 6(2), 41–49. — Foundational argument that competitive prices follow a random walk in log-space.
- **Black, F. & Scholes, M.** (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637–654. — Established GBM as the standard continuous-time asset price model.
- **Merton, R. C.** (1969). "Lifetime portfolio selection under uncertainty: The continuous-time case." *Review of Economics and Statistics*, 51(3), 247–257. — Uses GBM for the risky asset in continuous-time portfolio optimisation.

---

## 2. Historical Calibration (Drift & Volatility)

### Formulas

**Log returns:**

$$
r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

**Sample mean (monthly drift estimate):**

$$
\hat{\mu}_m = \frac{1}{n}\sum_{t=1}^{n} r_t
$$

**Sample variance (Bessel-corrected):**

$$
\hat{\sigma}_m^2 = \frac{1}{n-1}\sum_{t=1}^{n}(r_t - \hat{\mu}_m)^2
$$

**Annualisation:**

$$
\mu_a = 12 \cdot \hat{\mu}_m, \qquad \sigma_a = \sqrt{12}\;\hat{\sigma}_m
$$

### Variables

| Symbol | Meaning | Units |
|--------|---------|-------|
| \(P_t\) | Observed price at month \(t\) | USD index |
| \(r_t\) | Log return for month \(t\) | dimensionless |
| \(n\) | Number of observed returns | count |
| \(\hat{\mu}_m\) | Sample mean of monthly log returns | per month |
| \(\hat{\sigma}_m\) | Sample standard deviation of monthly log returns | per month |
| \(\mu_a\), \(\sigma_a\) | Annualised drift and volatility | per year |

### Real-World Logic

Log returns are the natural unit for asset analysis because they are additive over time (a 12-month log return is the sum of 12 monthly log returns) and they keep prices positive. The Bessel correction (\(n-1\) divisor) gives an unbiased estimate of the population variance from a finite sample.

Volatility scales with \(\sqrt{T}\) under the assumption that returns are serially uncorrelated — the "square root of time" rule. This is an approximation: CRE returns exhibit some autocorrelation due to appraisal smoothing, but \(\sqrt{T}\) scaling remains the standard first-order approach.

### Academic Support

- **Campbell, J. Y., Lo, A. W. & MacKinlay, A. C.** (1997). *The Econometrics of Financial Markets*. Princeton University Press. — Standard reference for return calculation, annualisation, and statistical estimation of asset dynamics.
- **Geltner, D.** (1991). "Smoothing in appraisal-based returns." *Journal of Real Estate Finance and Economics*, 4(3), 327–345. — Documents how appraisal-based CRE indices exhibit smoothed returns, motivating careful calibration.

---

## 3. Regime Classification via Rolling Volatility

### Formulas

**Step 1 — Rolling realised volatility** over a window of \(w\) months:

$$
\sigma_{\text{realized}}(t) = \sqrt{\frac{1}{w-1}\sum_{i=t-w+1}^{t}(r_i - \bar{r})^2}
$$

**Step 2 — Baseline volatility** (full-sample):

$$
\sigma_{\text{base}} = \text{std}(\text{all log returns})
$$

**Step 3 — Threshold classification:**

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| **Calm** | \(\sigma_{\text{realized}} < 0.70 \cdot \sigma_{\text{base}}\) | Below-normal volatility |
| **Neutral** | \(0.70 \leq \sigma_{\text{realized}} / \sigma_{\text{base}} < 1.20\) | Normal range |
| **Volatile** | \(1.20 \leq \sigma_{\text{realized}} / \sigma_{\text{base}} < 2.00\) | Elevated stress |
| **Panic** | \(\sigma_{\text{realized}} \geq 2.00 \cdot \sigma_{\text{base}}\) | Crisis-level volatility |

### Variables

| Symbol | Meaning | Units |
|--------|---------|-------|
| \(\sigma_{\text{realized}}(t)\) | Rolling standard deviation of recent log returns at month \(t\) | per month |
| \(\sigma_{\text{base}}\) | Full-sample standard deviation (long-run baseline) | per month |
| \(w\) | Rolling window size (default = 6 months) | months |
| \(\bar{r}\) | Mean of returns within the rolling window | per month |

### Real-World Logic

Financial markets alternate between calm and turbulent periods. Rather than assuming volatility is constant, the simulation measures recent realised volatility and compares it to the historical baseline. The thresholds (0.7×, 1.2×, 2.0×) define progressively more stressed conditions. This is conceptually identical to how VIX levels are interpreted: low-VIX environments are calm; spikes above 2× historical norms indicate crisis.

The 6-month window is appropriate for real estate because CRE cycles are slower than equity markets — monthly data with a half-year lookback captures regime shifts without excessive noise.

### Academic Support

- **Hamilton, J. D.** (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357–384. — Pioneered regime-switching models in economics; the threshold-based classification here is a simplified discrete analogue.
- **Guidolin, M. & Timmermann, A.** (2006). "An econometric model of nonlinear dynamics in the joint distribution of stock and bond returns." *Journal of Applied Econometrics*, 21(1), 1–22. — Demonstrates that multi-regime models capture fat tails and volatility clustering.
- **Ang, A. & Bekaert, G.** (2002). "Regime switches in interest rates." *Journal of Business & Economic Statistics*, 20(2), 163–182. — Formalises regime-switching volatility in financial time series.

---

## 4. Regime-Dependent Volatility Scaling

### Formula

$$
\sigma_{\text{effective}} = \sigma_{\text{base}} \times m(R)
$$

where \(m(R)\) is the **volatility multiplier** for regime \(R\):

| Regime | Multiplier \(m(R)\) |
|--------|---------------------|
| Calm | 0.8 |
| Neutral | 1.0 |
| Volatile | 1.5 |
| Panic | 2.2 |

### Variables

| Symbol | Meaning |
|--------|---------|
| \(\sigma_{\text{base}}\) | Calibrated baseline monthly volatility |
| \(m(R)\) | Piecewise multiplier determined by current regime \(R\) |
| \(\sigma_{\text{effective}}\) | Volatility fed into the GBM price step |

### Real-World Logic

The multipliers encode the empirical fact that volatility is not uniformly distributed — crisis periods exhibit 2–3× the volatility of calm periods. The 2.2× panic multiplier is consistent with observed CRE drawdowns during the 2008–2009 Global Financial Crisis, where monthly price swings were roughly double the long-run norm. The 0.8× calm multiplier reflects compression during stable growth periods (e.g., mid-2010s).

### Academic Support

- **Engle, R. F.** (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987–1007. — Established that financial volatility is time-varying and clustered.
- **Bollerslev, T.** (1986). "Generalized autoregressive conditional heteroscedasticity." *Journal of Econometrics*, 31(3), 307–327. — GARCH models formalise the idea that volatility depends on recent market conditions.

---

## 5. Markov Chain Regime Dynamics

### Formula

The regime at time \(t+1\) is drawn from a categorical distribution defined by the current state's row in the **transition matrix** \(P\):

$$
\Pr(R_{t+1} = j \mid R_t = i) = P_{ij}
$$

where \(P\) is a \(K \times K\) row-stochastic matrix (\(\sum_j P_{ij} = 1\) for all \(i\)), and \(K = 4\) states.

### Hybrid Rule

When the Markov chain is active alongside realised-volatility classification, a conservative hybrid applies:

$$
R_{\text{final}} = \max\!\big(\text{severity}(R_{\text{Markov}}),\;\text{severity}(R_{\text{realized}})\big)
$$

The regime is always set to the **more severe** of the two signals.

### Variables

| Symbol | Meaning |
|--------|---------|
| \(R_t\) | Regime state at month \(t\) (calm, neutral, volatile, panic) |
| \(P_{ij}\) | Probability of transitioning from state \(i\) to state \(j\) |
| \(K\) | Number of states (= 4) |

### Real-World Logic

Markets exhibit **persistence** (calm periods tend to stay calm; panics don't end instantly) and **asymmetric transitions** (it's easier to enter panic than to leave it). A Markov chain captures both features naturally through its transition probabilities. The diagonal entries of \(P\) encode persistence; the off-diagonal entries encode the likelihood and direction of regime shifts.

The hybrid combiner is conservative by design: it never downgrades severity based on the Markov macro signal alone. This mirrors how risk managers operate — you don't declare a crisis over just because a model says so; you wait for observed volatility to actually decline.

### Academic Support

- **Hamilton, J. D.** (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357–384. — The foundational paper on Markov-switching models in economics.
- **Kim, C.-J. & Nelson, C. R.** (1999). *State-Space Models with Regime Switching*. MIT Press. — Comprehensive treatment of Markov regime-switching in macroeconomics and finance.
- **Ang, A. & Timmermann, A.** (2012). "Regime changes and financial markets." *Annual Review of Financial Economics*, 4, 313–337. — Survey of regime-switching in asset returns, portfolio allocation, and risk management.

---

## 6. Transition Matrix Interpolation (Tokenization Effect)

### Formula

$$
P(\alpha) = (1 - \alpha) \cdot P_{\text{Traditional}} + \alpha \cdot P_{\text{Tokenized}}
$$

### Variables

| Symbol | Meaning | Range |
|--------|---------|-------|
| \(\alpha\) | Tokenization adoption level | \([0, 1]\) |
| \(P_{\text{Traditional}}\) | Empirical transition matrix for illiquid CRE | fixed |
| \(P_{\text{Tokenized}}\) | Empirical transition matrix for liquid CRE (REIT/VNQ proxy) | fixed |
| \(P(\alpha)\) | Blended transition matrix at adoption level \(\alpha\) | varies |

### Real-World Logic

This is the core mechanism linking tokenization to market health. \(P_{\text{Traditional}}\) is calibrated from 72 years of illiquid CRE data; \(P_{\text{Tokenized}}\) is calibrated from liquid REIT data. As \(\alpha\) increases (more tokenization adoption), the market's transition dynamics shift from illiquid to liquid behaviour.

The convex combination is valid because the set of row-stochastic matrices is convex — any weighted average of two valid transition matrices is itself a valid transition matrix with rows summing to 1. This is a standard technique in the mixture-of-Markov-chains literature.

Economically, this captures the idea that **partial** adoption of tokenization produces **partial** improvement in market dynamics — you don't need full adoption to see benefits, and the relationship is smooth rather than all-or-nothing.

### Academic Support

- **Capponi, A. & Jia, R.** (2021). "The adoption of blockchain-based decentralized exchanges." Working paper, Columbia University. — Models gradual adoption of decentralised trading infrastructure and its effect on market microstructure.
- **Yermack, D.** (2017). "Corporate governance and blockchains." *Review of Finance*, 21(1), 7–31. — Discusses how tokenization changes asset market structure, supporting the idea of gradual structural transition.
- **Baum, A. & Hartzell, D.** (2012). *Global Property Investment: Strategies, Structures, Decisions*. Wiley. — Documents the structural differences between direct (illiquid) and securitised (liquid) real estate markets.

---

## 7. Stationary Distribution

### Formula

The stationary (equilibrium) distribution \(\pi\) satisfies:

$$
\pi P = \pi, \qquad \sum_{i=1}^{K} \pi_i = 1
$$

Solved by reformulating as a linear system:

$$
(P^\top - I)\pi = 0
$$

Replace the last row with the normalisation constraint \(\sum \pi_i = 1\), then solve \(A\pi = b\).

### Variables

| Symbol | Meaning |
|--------|---------|
| \(\pi\) | Row vector of long-run probabilities for each state |
| \(\pi_i\) | Fraction of time spent in state \(i\) over an infinite horizon |
| \(P\) | Transition matrix |
| \(I\) | Identity matrix |

### Real-World Logic

\(\pi\) answers the question: "If this market ran forever under these dynamics, what fraction of time would it spend in calm, neutral, volatile, and panic?" This is the most fundamental summary of market health. A market with high \(\pi_{\text{calm}}\) and low \(\pi_{\text{panic}}\) is a healthier market.

The simulation computes \(\pi\) for every adoption level \(\alpha \in [0, 1]\), showing exactly how the long-run distribution of market states improves with tokenization.

### Academic Support

- **Norris, J. R.** (1997). *Markov Chains*. Cambridge University Press. — Definitive mathematical treatment; proves existence and uniqueness of the stationary distribution for irreducible, aperiodic chains.
- **Kemeny, J. G. & Snell, J. L.** (1976). *Finite Markov Chains*. Springer. — Classical text covering stationary distributions, mean first passage times, and fundamental matrices.

---

## 8. Mean First Passage Times

### Formula

Using the **fundamental matrix** \(Z\):

$$
Z = (I - P + \Pi)^{-1}
$$

where \(\Pi\) is the matrix with every row equal to \(\pi\).

**Diagonal (mean return time):**

$$
M_{ii} = \frac{1}{\pi_i}
$$

**Off-diagonal (expected months from state \(i\) to first reach state \(j\)):**

$$
M_{ij} = \frac{Z_{jj} - Z_{ij}}{\pi_j}
$$

### Variables

| Symbol | Meaning |
|--------|---------|
| \(M_{ij}\) | Expected number of steps to reach state \(j\) for the first time, starting from state \(i\) |
| \(Z\) | Fundamental matrix of the Markov chain |
| \(\Pi\) | Matrix where each row is the stationary distribution \(\pi\) |

### Real-World Logic

Mean first passage times answer questions like:

- **"How many months does it take to recover from panic to neutral?"** — This is \(M_{\text{panic}, \text{neutral}}\). A lower value means faster crisis recovery.
- **"How long until a calm market first hits panic?"** — This is \(M_{\text{calm}, \text{panic}}\). A higher value means the market is more resilient.

These are the most operationally meaningful metrics for risk managers and investors: they quantify how long stress lasts and how robust the market is against shocks.

### Academic Support

- **Kemeny, J. G. & Snell, J. L.** (1976). *Finite Markov Chains*. Springer. — Derives the fundamental matrix approach to mean first passage times.
- **Grinstead, C. M. & Snell, J. L.** (1997). *Introduction to Probability*. AMS. — Accessible derivation of MFPT from first principles.

---

## 9. Expected Sojourn Times

### Formula

The expected number of consecutive months spent in state \(k\) per visit follows a **geometric distribution**:

$$
E[\tau_k] = \frac{1}{1 - P_{kk}}
$$

### Variables

| Symbol | Meaning |
|--------|---------|
| \(E[\tau_k]\) | Expected consecutive months in state \(k\) before transitioning out |
| \(P_{kk}\) | Self-transition (persistence) probability for state \(k\) |

### Real-World Logic

If a market regime persists with probability 0.90 each month (\(P_{kk} = 0.90\)), the expected duration of that regime is \(1/(1-0.90) = 10\) months. High persistence in panic (\(P_{\text{panic,panic}}\)) is bad — it means the market gets stuck in crisis. One of the key findings of the simulation is that tokenization reduces panic persistence and therefore shortens panic sojourn times.

This is the Markov chain analogue of asking "How long does a typical recession last?" — a question central to macroeconomic policy and investment planning.

### Academic Support

- **Ross, S. M.** (2014). *Introduction to Probability Models*, 11th ed. Academic Press. — Derives the geometric distribution of sojourn times in discrete-time Markov chains.

---

## 10. Logistic Adoption Curve

### Formula

$$
\alpha(t) = \frac{L}{1 + e^{-k(t - t_0)}}
$$

In the simulator (normalised to [0, 1]):

$$
\alpha(t) = \frac{1}{1 + e^{-k_s(t - t_m)}}
$$

**Doubling time** of the adoption S-curve:

$$
T_{\text{double}} = \frac{\ln 2}{k}
$$

### Variables

| Symbol | Meaning | Source |
|--------|---------|-------|
| \(L\) | Saturation level (maximum TVL) | Fitted to RWA data |
| \(k\) | Growth rate (steepness of the sigmoid) | Fitted to RWA data |
| \(t_0\) | Inflection point (month of fastest growth) | Fitted to RWA data |
| \(k_s\) | `adoption_speed` in the simulator (default 0.15) | Configuration |
| \(t_m\) | `adoption_midpoint` in the simulator (default month 24) | Configuration |

### Real-World Logic

Technology adoption universally follows an S-curve: slow early uptake, rapid growth through the middle, saturation at the end. The simulation fits this curve to **actual RWA (Real World Asset) tokenized real estate TVL data** from May 2023 to March 2026 using nonlinear least squares (`scipy.optimize.curve_fit`).

The fitted parameters reveal where the market currently sits on the adoption curve (early growth phase as of 2026) and project where it will be in 2, 5, and 10 years. This grounds the simulation in empirical adoption data rather than assumption.

### Academic Support

- **Rogers, E. M.** (2003). *Diffusion of Innovations*, 5th ed. Free Press. — The canonical work on S-curve technology adoption; defines innovators, early adopters, early majority, late majority, laggards.
- **Bass, F. M.** (1969). "A new product growth for model consumer durables." *Management Science*, 15(5), 215–227. — Formalises the Bass diffusion model, a generalisation of the logistic curve that separates innovation and imitation effects.
- **Catalini, C. & Gans, J. S.** (2020). "Some simple economics of the blockchain." *Communications of the ACM*, 63(7), 80–90. — Applies adoption economics to blockchain technology.

---

## 11. Anchor Weight Mechanism

### Formula

$$
w_{\text{anchor}} = \max\!\big(w_{\text{floor}},\; 1 - \alpha(t)\big)
$$

**Reference price blending** (used by every trading agent):

$$
P_{\text{ref}} = w_{\text{anchor}} \cdot P_{\text{fundamental}} + (1 - w_{\text{anchor}}) \cdot P_{\text{micro}}
$$

### Variables

| Symbol | Meaning | Default |
|--------|---------|---------|
| \(w_{\text{anchor}}\) | Weight placed on the fundamental (appraisal) price | starts at 1.0 |
| \(w_{\text{floor}}\) | Minimum anchor weight (fundamental never fully ignored) | 0.05 |
| \(\alpha(t)\) | Current tokenization adoption level | 0 → 1 |
| \(P_{\text{fundamental}}\) | Fundamental CRE price (GBM or historical) | — |
| \(P_{\text{micro}}\) | Live microstructure price (from order book) | — |

### Real-World Logic

In traditional CRE markets, prices are determined by periodic appraisals — valuers assess what a property is "worth" and transactions reference this appraised value. Traders are heavily anchored to the fundamental (appraised) price because there is no continuous market to discover price otherwise.

As tokenization introduces continuous trading, price discovery shifts from appraisal-based to market-based. The anchor weight mechanism captures this transition: at low adoption, agents quote tightly around the appraised value; at high adoption, they blend in the live market price, allowing the order book to self-discover price through supply and demand.

The floor (\(w_{\text{floor}} = 0.05\)) ensures that even fully tokenized markets retain a minimal connection to fundamentals, reflecting the fact that real estate always has an underlying physical value.

### Academic Support

- **Geltner, D. & Miller, N. G.** (2007). *Commercial Real Estate Analysis and Investments*, 2nd ed. South-Western. — Documents the appraisal-based pricing mechanism in traditional CRE and its implications for price discovery.
- **Baum, A.** (2020). "Tokenisation — the future of real estate investment?" *The Journal of Portfolio Management*, 46(special real estate issue). — Argues that tokenization shifts CRE from appraisal-based to market-based pricing.
- **Tversky, A. & Kahneman, D.** (1974). "Judgment under uncertainty: Heuristics and biases." *Science*, 185(4157), 1124–1131. — Defines the anchoring heuristic; the anchor weight formula is a literal implementation of this cognitive bias in price formation.

---

## 12. Bayesian Dirichlet–Multinomial Estimation

### Model

Each row of the transition matrix is modelled independently:

$$
P_{k,\cdot} \;\sim\; \text{Dirichlet}(\boldsymbol{\alpha}_k)
$$

**Sticky Dirichlet prior:**

$$
\alpha_{kk} = 5.0 \quad\text{(diagonal)}, \qquad \alpha_{kj} = 1.0 \quad\text{(off-diagonal, } j \neq k\text{)}
$$

**Pooled transition counts:**

$$
N_{kj} = \sum_{i=1}^{R} \text{count}(\text{REIT } i\text{: state } k \to \text{state } j)
$$

**Posterior (conjugate update):**

$$
P_{k,\cdot} \mid \text{data} \;\sim\; \text{Dirichlet}\!\big(\boldsymbol{\alpha}_k + \mathbf{N}_{k,\cdot}\big)
$$

**Posterior mean (point estimate):**

$$
\hat{P}_{kj} = \frac{\alpha_{kj} + N_{kj}}{\sum_{\ell}\!(\alpha_{k\ell} + N_{k\ell})}
$$

**Credible intervals:** Computed from 5,000 posterior Dirichlet samples using equal-tailed quantiles.

### Variables

| Symbol | Meaning |
|--------|---------|
| \(\boldsymbol{\alpha}_k\) | Prior concentration vector for row \(k\) |
| \(N_{kj}\) | Pooled count of observed \(k \to j\) transitions across all REITs |
| \(R\) | Number of REITs in the basket (= 4: O, NNN, WPC, ADC) |
| \(\hat{P}_{kj}\) | Posterior mean probability of transitioning from \(k\) to \(j\) |

### Real-World Logic

A single REIT has noisy transition estimates because some state transitions (e.g., calm → panic) are rare. By pooling observations across multiple REITs from the **same asset class** (net lease CRE), the estimator effectively uses multiple independent readings of the same underlying market state, reducing estimation variance.

The sticky Dirichlet prior encodes the well-documented empirical regularity that financial regimes persist — calm months are usually followed by calm months, and panics don't end overnight. The prior nudges the posterior toward persistent dynamics without overwhelming the data; with hundreds of observed transitions, the data dominates.

### Academic Support

- **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B.** (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press. — Definitive reference for Dirichlet-Multinomial conjugacy and hierarchical Bayesian modelling.
- **Fox, E. B., Sudderth, E. B., Jordan, M. I. & Willsky, A. S.** (2011). "A sticky HDP-HMM with application to speaker diarization." *Annals of Applied Statistics*, 5(2A), 1020–1056. — Introduces the sticky Dirichlet prior for Hidden Markov Models to encourage regime persistence; directly motivates the prior choice used here.
- **Robert, C. P. & Casella, G.** (2004). *Monte Carlo Statistical Methods*, 2nd ed. Springer. — Covers posterior sampling from Dirichlet distributions and credible interval computation.

---

## 13. Merton Optimal Portfolio Weight

### Formula

$$
w^* = \frac{\mu - r}{\gamma \cdot \sigma^2}
$$

### Variables

| Symbol | Meaning | Value in Simulation |
|--------|---------|---------------------|
| \(w^*\) | Optimal fraction of wealth allocated to the risky asset (CRE) | computed |
| \(\mu\) | Expected annual return of CRE | calibrated |
| \(r\) | Risk-free annual rate | 2% |
| \(\gamma\) | Coefficient of relative risk aversion | 3.0 |
| \(\sigma\) | Annual volatility of CRE | calibrated |

### Real-World Logic

Merton's formula answers: "Given the asset's expected return and risk, and my personal risk tolerance, what fraction of my portfolio should I allocate to this asset?" A higher risk premium (\(\mu - r\)) or lower volatility increases the optimal allocation; higher risk aversion decreases it.

In the simulation, this is computed for both traditional and tokenized scenarios. If tokenization reduces \(\sigma\) (through faster regime recovery and less time in panic), the optimal allocation increases — meaning tokenized CRE deserves a larger portfolio share. This connects market microstructure improvements to tangible portfolio decisions.

### Academic Support

- **Merton, R. C.** (1969). "Lifetime portfolio selection under uncertainty: The continuous-time case." *Review of Economics and Statistics*, 51(3), 247–257. — Derives the formula for continuous-time portfolio optimisation under CRRA utility.
- **Merton, R. C.** (1971). "Optimum consumption and portfolio rules in a continuous-time model." *Journal of Economic Theory*, 3(4), 373–413. — Extends the framework to include consumption and more general dynamics.
- **Campbell, J. Y. & Viceira, L. M.** (2002). *Strategic Asset Allocation: Portfolio Choice for Long-Term Investors*. Oxford University Press. — Modern treatment of Merton-style portfolio theory applied to multi-asset allocation.

---

## 14. Box–Muller Transform

### Formula

Given two independent uniform random variables \(U_1, U_2 \sim \text{Uniform}(0, 1)\):

$$
Z = \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2)
$$

yields a standard normal random variable \(Z \sim N(0, 1)\).

### Variables

| Symbol | Meaning |
|--------|---------|
| \(U_1, U_2\) | Independent draws from Uniform(0, 1) |
| \(Z\) | Standard normal output |

### Real-World Logic

The GBM price model requires normal random shocks, but pseudo-random number generators produce uniform variates. The Box–Muller transform is an exact (not approximate) method for converting pairs of uniform draws into standard normal draws. It is used instead of library functions to maintain full reproducibility and avoid platform-specific behaviour in the custom random number generator.

### Academic Support

- **Box, G. E. P. & Muller, M. E.** (1958). "A note on the generation of random normal deviates." *Annals of Mathematical Statistics*, 29(2), 610–611. — Original paper.

---

## 15. Microstructure: Order-Flow Imbalance

### Formula

$$
\text{Imbalance} = \frac{V_{\text{buy}} - V_{\text{sell}}}{V_{\text{buy}} + V_{\text{sell}}}
$$

### Variables

| Symbol | Meaning |
|--------|---------|
| \(V_{\text{buy}}\) | Total cumulative buy volume executed |
| \(V_{\text{sell}}\) | Total cumulative sell volume executed |
| Imbalance | Normalised measure of directional pressure, range \([-1, +1]\) |

### Real-World Logic

Order-flow imbalance is a key indicator of short-term price pressure. A positive imbalance (more buying than selling) suggests upward price pressure; a negative imbalance suggests downward pressure. Market makers and algorithmic traders monitor this signal in real time to adjust their quotes.

In the simulation, imbalance is computed after every trade execution and reflects the emergent directional bias of the 1,000 trading agents.

### Academic Support

- **Chordia, T., Roll, R. & Subrahmanyam, A.** (2002). "Order imbalance, liquidity, and market returns." *Journal of Financial Economics*, 65(1), 111–130. — Empirically demonstrates the relationship between order imbalance and price movements.
- **Cont, R., Kukanov, A. & Stoikov, S.** (2014). "The price impact of order book events." *Journal of Financial Econometrics*, 12(1), 47–88. — Formalises the link between order-flow imbalance and price formation.

---

## 16. Microstructure: Inventory-Skewed Market Making

### Formulas

**Inventory normalisation:**

$$
\text{inv}_{\text{norm}} = \frac{\text{inv}}{\text{max\_inv}} \in [-1, +1]
$$

**Inventory-skewed mid price:**

$$
P_{\text{mid}} = P_{\text{ref}} \cdot \left(1 - s_{\text{skew}} \cdot 10^{-4} \cdot \text{inv}_{\text{norm}}\right)
$$

**Bid and ask:**

$$
P_{\text{bid}} = P_{\text{mid}} - \tfrac{1}{2}\,P_{\text{ref}} \cdot s_{\text{spread}} \cdot 10^{-4}
$$

$$
P_{\text{ask}} = P_{\text{mid}} + \tfrac{1}{2}\,P_{\text{ref}} \cdot s_{\text{spread}} \cdot 10^{-4}
$$

**Urgency-adjusted size:**

$$
q = \max\!\big(1,\;\lfloor q_{\text{base}} \cdot (1 + 0.5 \cdot |\text{inv}_{\text{norm}}|) \rfloor\big)
$$

### Variables

| Symbol | Meaning | Default |
|--------|---------|---------|
| inv | Current inventory position | 0 |
| max_inv | Maximum inventory capacity | 10 units |
| \(\text{inv}_{\text{norm}}\) | Normalised inventory \(\in [-1, 1]\) | — |
| \(s_{\text{spread}}\) | Spread in basis points | 10 bps |
| \(s_{\text{skew}}\) | Inventory skew in basis points | 5 bps |
| \(q_{\text{base}}\) | Base order quantity | 2 units |
| \(P_{\text{ref}}\) | Blended reference price (see [Anchor Weight](#11-anchor-weight-mechanism)) | — |

### Real-World Logic

This implements the classic **Avellaneda–Stoikov** market-making strategy. When a market maker accumulates a long inventory (bought more than sold), they face directional risk. To reduce this risk, they skew their quotes downward — making their ask cheaper to attract sellers and their bid less attractive to discourage more buyers. The reverse applies for short inventory.

The 10 bps spread represents the half-spread typical of liquid REIT markets. The urgency scaling means that when inventory builds up, the market maker posts larger quantities to accelerate rebalancing — a behaviour observed in real-world electronic market makers.

### Academic Support

- **Avellaneda, M. & Stoikov, S.** (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217–224. — Derives the optimal inventory-skewed quoting strategy for market makers; directly motivates the mid-price skewing formula.
- **Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J.** (2013). "Dealing with the inventory risk: A solution to the market making problem." *Mathematics and Financial Economics*, 7(4), 477–507. — Extends the Avellaneda–Stoikov framework with more general inventory penalties.
- **Ho, T. & Stoll, H. R.** (1981). "Optimal dealer pricing under transactions and return uncertainty." *Journal of Financial Economics*, 9(1), 47–73. — Original model of inventory-based spread adjustment by market makers.

---

## 17. Monte Carlo Path Metrics

### Formulas

**Stress time (% of months):**

$$
\text{Stress\%} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\!\left[R_t \in \{\text{volatile}, \text{panic}\}\right] \times 100
$$

**Maximum drawdown:**

$$
\text{DD}_t = \frac{S_t - \max_{s \leq t} S_s}{\max_{s \leq t} S_s}, \qquad \text{MaxDD} = \min_{t} \text{DD}_t
$$

**Annualised volatility from monthly log returns:**

$$
\sigma_{\text{ann}} = \text{std}(r_1, \ldots, r_T) \cdot \sqrt{12}
$$

**Total return:**

$$
R_{\text{total}} = \left(\frac{S_T}{S_0} - 1\right) \times 100\%
$$

**Panic episode duration:** Average length of consecutive runs of panic states:

$$
\bar{d}_{\text{panic}} = \frac{1}{E}\sum_{e=1}^{E}(\text{end}_e - \text{start}_e + 1)
$$

### Variables

| Symbol | Meaning |
|--------|---------|
| \(T\) | Simulation horizon (240 months = 20 years) |
| \(S_t\) | Simulated price at month \(t\) |
| \(\text{MaxDD}\) | Maximum peak-to-trough percentage decline |
| \(E\) | Number of distinct panic episodes |

### Real-World Logic

These are standard risk and performance metrics used by institutional investors and regulators:

- **Stress time** measures the fraction of market life spent in unfavourable conditions — analogous to "percentage of time in recession" in macroeconomics.
- **Maximum drawdown** is the worst peak-to-trough decline, the single most important downside risk measure for illiquid assets. CRE investors particularly care about drawdowns because they cannot quickly exit positions.
- **Annualised volatility** standardises risk for comparison across asset classes and time horizons.
- **Panic duration** measures how long crises persist per episode — critical for liquidity planning and capital reserves.

### Academic Support

- **Magdon-Ismail, M. & Atiya, A.** (2004). "Maximum drawdown." *Risk Magazine*, October. — Analytical properties of maximum drawdown as a risk measure.
- **Ang, A., Chen, J. & Xing, Y.** (2006). "Downside risk." *Review of Financial Studies*, 19(4), 1191–1239. — Shows that downside risk measures (including drawdown) are priced in cross-sectional returns.

---

## 18. Bootstrap & Frequentist Inference

### Bootstrap Confidence Intervals on Stationary Distribution

For each bootstrap replicate \(b = 1, \ldots, B\):

1. For each row \(i\), draw a new transition row from \(\text{Dirichlet}(\text{counts}[i] \cdot P[i,:] + 0.5)\)
2. Recompute \(\pi^{(b)}\) from the bootstrapped matrix

Then:

$$
\text{CI}_{95\%} = \left[\text{quantile}_{2.5\%}(\pi^{(b)}),\;\text{quantile}_{97.5\%}(\pi^{(b)})\right]
$$

### Welch's t-test (Monte Carlo comparisons)

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

Standard error of the mean:

$$
\text{SE} = \frac{s}{\sqrt{n}}, \qquad \text{CI}_{95\%} = \bar{X} \pm t_{0.975, n-1} \cdot \text{SE}
$$

### Variables

| Symbol | Meaning |
|--------|---------|
| \(B\) | Number of bootstrap replicates (10,000) |
| \(\bar{X}_1, \bar{X}_2\) | Sample means for traditional and tokenized paths |
| \(s_1, s_2\) | Sample standard deviations |
| \(n_1, n_2\) | Number of paths per scenario (5,000 each) |

### Real-World Logic

With 5,000 Monte Carlo paths per scenario, any difference between traditional and tokenized outcomes could be due to random noise. The Welch t-test provides formal statistical significance (\(p < 0.001\) means the difference is not due to chance). The bootstrap on the stationary distribution quantifies uncertainty in the analytical results that arises from finite estimation of the transition matrices themselves.

### Academic Support

- **Efron, B. & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. — Foundational reference for bootstrap methods.
- **Welch, B. L.** (1947). "The generalization of 'Student's' problem when several different population variances are involved." *Biometrika*, 34(1–2), 28–35. — Original paper on the unequal-variance t-test.

---

## 19. Literature Summary Table

| Formula / Concept | Primary References |
|---|---|
| Geometric Brownian Motion | Samuelson (1965); Black & Scholes (1973); Merton (1969) |
| Log-return calibration | Campbell, Lo & MacKinlay (1997); Geltner (1991) |
| Regime-switching models | Hamilton (1989); Ang & Bekaert (2002); Ang & Timmermann (2012) |
| Time-varying volatility | Engle (1982); Bollerslev (1986) |
| Markov chains (theory) | Norris (1997); Kemeny & Snell (1976) |
| Mean first passage times | Kemeny & Snell (1976); Grinstead & Snell (1997) |
| Logistic adoption / diffusion | Rogers (2003); Bass (1969); Catalini & Gans (2020) |
| Tokenization & blockchain markets | Yermack (2017); Capponi & Jia (2021); Baum (2020) |
| Appraisal-based CRE pricing | Geltner & Miller (2007); Geltner (1991) |
| Bayesian Dirichlet–Multinomial | Gelman et al. (2013); Fox et al. (2011) |
| Sticky HDP-HMM priors | Fox et al. (2011) |
| Merton portfolio theory | Merton (1969, 1971); Campbell & Viceira (2002) |
| Box–Muller transform | Box & Muller (1958) |
| Order-flow imbalance | Chordia, Roll & Subrahmanyam (2002); Cont, Kukanov & Stoikov (2014) |
| Inventory-based market making | Avellaneda & Stoikov (2008); Ho & Stoll (1981) |
| Maximum drawdown | Magdon-Ismail & Atiya (2004); Ang, Chen & Xing (2006) |
| Bootstrap methods | Efron & Tibshirani (1993) |
| Welch t-test | Welch (1947) |
| Anchoring heuristic | Tversky & Kahneman (1974) |

---

*This document covers every formula implemented in the simulation codebase, from price dynamics and regime classification to Bayesian estimation and microstructure. Each formula is grounded in established academic theory and connected to the real-world phenomena it models.*
