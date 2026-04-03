# Future Work

This folder contains components that were built during earlier exploration phases but are **not used in the current thesis simulation**. They are preserved here because they represent meaningful research directions that could be revisited with more data or compute.

---

## Why these are here

The current thesis approach is built on:
- Analytical Markov chain analysis (stationary distributions, mean first passage times)
- Monte Carlo simulation with statistical confidence intervals
- Sensitivity/adoption-curve analysis via matrix interpolation

The items below are either data-hungry (DMM), infrastructure-heavy (qfclient), or outside the core research question (RL/forex data).

---

## Contents

### `dmm/` — Deep Markov Model
A PyTorch neural network that learns regime transitions and emissions from data.

**Why deferred:** The DMM requires tens of thousands of time-step observations to train reliably. With ~253 months of VNQ data and ~871 months of CRE data, the model consistently underperforms the empirical transition matrices. The posterior collapse issue documented in `dmm/docs/FIXING_POSTERIOR_COLLAPSE.md` is a direct consequence of insufficient data.

**When to revisit:** If real tokenized CRE transaction-level data becomes available (daily or weekly), the DMM becomes viable and would provide a genuinely data-driven alternative to the hand-specified matrices.

### `qfclient/` — Multi-provider Market Data Client
A Python package wrapping Alpha Vantage, Twelve Data, Yahoo Finance, and other providers.

**Why deferred:** Only needed to feed data into the DMM training pipeline. The current analysis uses the CRE CSV directly and the empirically-derived VNQ transition matrix. Reintroduce when live or higher-frequency data is needed.

### `scripts/` — Deferred Analysis Scripts
- `residential_equity_regimes.py` — Regime analysis of the Residential Equity index. Outside the core thesis comparison (traditional CRE vs tokenized CRE). Could become a third comparison arm in extended work.

### `data/` — Non-CRE Datasets
- `EURUSD_Candlestick_1_Hour_BID_*.csv` — Forex data used by the RL_Trading project
- `Equinix.csv` — Equinix REIT stock data for RL training
- `Residential_Equity.csv` — Input for the residential equity regime script
- `trade_history_output.csv` — RL agent trading logs

### `outputs/` — DMM and Exploratory Charts
Model checkpoints and training charts from DMM experiments:
- `deep_markov_model_*.pt` — Saved PyTorch model weights
- `dmm_*.png` — Training loss and regime assignment plots
- `data_quality_inspection.png` — Data diagnostics from DMM pipeline
- `residential_equity_regimes.png` — Output of the deferred residential script

### `docs/` — Development Notes
Progress and fix logs from the earlier development phase. Kept for reference:
- `FINAL_SIMPLIFICATION.md` — Simplification decisions made during refactoring
- `FIXES_APPLIED.md` — Bug fixes applied to the DMM and simulator
- `QFCLIENT_SETUP.md` — API key and provider configuration guide
- `TRAINING_ISSUE_FIXED.md` — DMM posterior collapse diagnosis and fix
- `TRAINING_PROGRESS.md` — Training run logs
- `ORGANIZATION_SUMMARY.md` — Earlier project reorganization notes

---

## Recommended next steps for these components

1. **DMM**: Obtain daily REIT NAV or transaction-level tokenized real estate data. Minimum ~5,000 observations recommended for reliable training.
2. **Multi-asset expansion**: Use `qfclient` to pull IYR, XLRE, RMZ alongside VNQ and average their transition matrices for a more robust tokenized baseline.
3. **International comparison**: Run the same regime-switching analysis on European REIT indices (e.g., IPRP.L) to test whether findings generalize beyond the US.
4. **Residential arm**: Bring back `residential_equity_regimes.py` as a third simulation track once the primary CRE vs tokenized comparison is published.
