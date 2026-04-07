"""
Bayesian Pooled CRE Transition Matrix Estimator
================================================

Pools regime observations across multiple commercial real estate (CRE) REITs
to estimate a single, holistic Bayesian transition matrix for the CRE asset class.

Model
-----
All REITs are assumed to share the same latent market state S_t at each time t.
Each REIT i emits a regime label independently given the shared state:

    S_t | S_{t-1} = k  ~  Categorical(P_{k,1}, ..., P_{k,K})
    P_{k,·}            ~  Dirichlet(alpha_{k,·})          [prior]

Since regime labels are observed (classified from rolling volatility), this
reduces to a conjugate Dirichlet-Multinomial posterior:

    P_{k,·} | data  ~  Dirichlet(alpha_{k,·} + N_{k,·})

where N_{k,j} is the total count of (state k -> state j) transitions pooled
across all REITs and all time periods.

Sticky Prior
------------
The prior concentrations alpha are set higher on the diagonal to encode that
regimes tend to persist:

    alpha[k, k]   = PRIOR_PERSISTENCE   (e.g. 5.0)
    alpha[k, j≠k] = PRIOR_OFFDIAG       (e.g. 1.0)

REIT Basket (Net Lease CRE – homogeneous peer group)
-----------------------------------------------------
Ticker  Name                  Why included
------  --------------------  ---------------------------
O       Realty Income         Largest net lease REIT, deep history
NNN     NNN REIT              Pure-play triple-net, ~30yr history
WPC     W. P. Carey           Diversified net lease, long history
ADC     Agree Realty          Smaller / higher-quality tenants

All four are NYSE-listed, operate the same business model (long-term
triple-net leases on single-tenant commercial properties), and respond
to the same macro / credit / cap-rate drivers. This makes their shared
latent state a clean signal for the CRE asset class.

Outputs
-------
- Holistic transition matrix P  (posterior mean)
- 95% credible intervals for each cell
- Posterior samples for downstream Monte Carlo use
- Regime statistics: time-in-state, expected duration, stability
- Drop-in Python format for RegimeMarkovChain
- Visualization saved to outputs/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import dirichlet

# ---------------------------------------------------------------------------
# Path setup so we can import from the sim package
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from sim.data_loader import load_twelvedata_api

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

STATES: List[str] = ["calm", "neutral", "volatile", "panic"]
K: int = len(STATES)

# Bayesian prior: sticky Dirichlet (diagonal favours persistence)
PRIOR_PERSISTENCE: float = 5.0   # alpha_{k,k}
PRIOR_OFFDIAG: float = 1.0       # alpha_{k,j≠k}

# Rolling volatility window (months)
VOL_WINDOW: int = 6

# Posterior credible interval width
CI_LEVEL: float = 0.95
N_POSTERIOR_SAMPLES: int = 5_000

# Data range
START_DATE: str = "2005-01-01 00:00:00"
END_DATE: str   = "2026-01-01 00:00:00"

# Net Lease CRE REIT basket (homogeneous peer group)
CRE_REITS: Dict[str, str] = {
    "O":   "Realty Income",
    "NNN": "NNN REIT",
    "WPC": "W. P. Carey",
    "ADC": "Agree Realty",
}


# =============================================================================
# Volatility-Based Regime Classification
# =============================================================================

def rolling_volatility(returns: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """Compute rolling standard deviation of log returns."""
    n = len(returns)
    vols = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = returns[start : i + 1]
        vols[i] = np.std(chunk, ddof=1) if len(chunk) >= 2 else 0.0
    return vols


def classify_regime(vol: float, base_vol: float) -> str:
    """Map realized volatility to a regime label using fixed thresholds."""
    if vol < 0.70 * base_vol:
        return "calm"
    elif vol < 1.20 * base_vol:
        return "neutral"
    elif vol < 2.00 * base_vol:
        return "volatile"
    else:
        return "panic"


def regimes_from_prices(prices: List[float]) -> List[str]:
    """
    Derive regime sequence from a price series.

    Steps:
        1. Compute log returns.
        2. Compute rolling volatility.
        3. Classify each observation using the full-sample baseline vol.
    """
    prices_arr = np.array(prices, dtype=float)
    log_ret = np.log(prices_arr[1:] / prices_arr[:-1])
    base_vol = np.std(log_ret, ddof=1)
    vols = rolling_volatility(log_ret, window=VOL_WINDOW)
    return [classify_regime(v, base_vol) for v in vols]


# =============================================================================
# Transition Count Accumulation
# =============================================================================

def count_transitions(regimes: List[str]) -> np.ndarray:
    """
    Count consecutive-pair transitions in a regime sequence.

    Returns a (K x K) integer array where entry [i, j] is the number
    of observed i -> j transitions.
    """
    idx = {s: i for i, s in enumerate(STATES)}
    counts = np.zeros((K, K), dtype=float)
    for t in range(len(regimes) - 1):
        i = idx.get(regimes[t])
        j = idx.get(regimes[t + 1])
        if i is not None and j is not None:
            counts[i, j] += 1.0
    return counts


# =============================================================================
# Bayesian Posterior
# =============================================================================

def build_prior() -> np.ndarray:
    """
    Build the Dirichlet prior concentration matrix (K x K).

    Sticky prior: larger values on the diagonal encode that regimes
    tend to persist from one period to the next.
    """
    alpha = np.full((K, K), PRIOR_OFFDIAG)
    np.fill_diagonal(alpha, PRIOR_PERSISTENCE)
    return alpha


def posterior_mean(alpha_prior: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Compute the posterior mean transition matrix.

    Under a Dirichlet-Multinomial model the posterior for row k is
        Dirichlet(alpha_prior[k] + counts[k])
    and the posterior mean of each cell is:
        E[P_{k,j}] = (alpha[k,j] + counts[k,j]) / sum_j(alpha[k,j] + counts[k,j])
    """
    alpha_post = alpha_prior + counts
    row_sums = alpha_post.sum(axis=1, keepdims=True)
    return alpha_post / row_sums


def posterior_samples(
    alpha_prior: np.ndarray,
    counts: np.ndarray,
    n_samples: int = N_POSTERIOR_SAMPLES,
) -> np.ndarray:
    """
    Draw samples from the posterior Dirichlet for each row.

    Returns array of shape (n_samples, K, K).
    """
    alpha_post = alpha_prior + counts
    samples = np.zeros((n_samples, K, K))
    for k in range(K):
        samples[:, k, :] = dirichlet.rvs(alpha_post[k], size=n_samples)
    return samples


def credible_intervals(
    samples: np.ndarray, level: float = CI_LEVEL
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equal-tailed credible intervals from posterior samples.

    Returns (lower, upper) arrays of shape (K, K).
    """
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    lower = np.quantile(samples, lo, axis=0)
    upper = np.quantile(samples, hi, axis=0)
    return lower, upper


# =============================================================================
# Data Loading
# =============================================================================

def load_reit(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch monthly price history for a REIT from the Twelve Data API.

    Returns a DataFrame with columns ['date', 'price'], or None on failure.
    """
    print(f"  Fetching {ticker} ...", end="", flush=True)
    try:
        history = load_twelvedata_api(
            symbol=ticker,
            interval="1month",
            start_date=START_DATE,
            end_date=END_DATE,
        )
        if len(history) < 24:
            print(f"  [SKIP – only {len(history)} months]")
            return None
        df = pd.DataFrame(
            [{"date": p.date, "price": p.price} for p in history]
        )
        print(f"  {len(df)} months  OK")
        return df
    except Exception as exc:
        print(f"  [FAILED: {exc}]")
        return None


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    P_mean: np.ndarray,
    P_lower: np.ndarray,
    P_upper: np.ndarray,
    pooled_counts: np.ndarray,
    ticker_stats: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Produce a comprehensive figure summarising the Bayesian estimation."""

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Bayesian Pooled CRE Transition Matrix\n"
        "Net Lease REIT Basket: O · NNN · WPC · ADC",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    state_labels = [s.capitalize() for s in STATES]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    # ------------------------------------------------------------------
    # 1. Posterior mean heatmap
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(P_mean, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax1.set_title("Posterior Mean  P(k → j)", fontweight="bold")
    ax1.set_xticks(range(K))
    ax1.set_yticks(range(K))
    ax1.set_xticklabels(state_labels)
    ax1.set_yticklabels(state_labels)
    ax1.set_xlabel("To state")
    ax1.set_ylabel("From state")
    for i in range(K):
        for j in range(K):
            ax1.text(
                j, i, f"{P_mean[i, j]:.3f}",
                ha="center", va="center",
                fontsize=9,
                color="white" if P_mean[i, j] > 0.5 else "black",
            )
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # ------------------------------------------------------------------
    # 2. Credible interval width heatmap
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ci_width = P_upper - P_lower
    im2 = ax2.imshow(ci_width, cmap="Blues", vmin=0, aspect="auto")
    ax2.set_title(f"95% CI Width  (uncertainty)", fontweight="bold")
    ax2.set_xticks(range(K))
    ax2.set_yticks(range(K))
    ax2.set_xticklabels(state_labels)
    ax2.set_yticklabels(state_labels)
    ax2.set_xlabel("To state")
    ax2.set_ylabel("From state")
    for i in range(K):
        for j in range(K):
            ax2.text(
                j, i, f"{ci_width[i, j]:.3f}",
                ha="center", va="center", fontsize=9,
            )
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # ------------------------------------------------------------------
    # 3. Pooled transition counts
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(pooled_counts, cmap="Greens", aspect="auto")
    ax3.set_title("Pooled Transition Counts\n(all REITs combined)", fontweight="bold")
    ax3.set_xticks(range(K))
    ax3.set_yticks(range(K))
    ax3.set_xticklabels(state_labels)
    ax3.set_yticklabels(state_labels)
    ax3.set_xlabel("To state")
    ax3.set_ylabel("From state")
    for i in range(K):
        for j in range(K):
            ax3.text(
                j, i, f"{int(pooled_counts[i, j])}",
                ha="center", va="center", fontsize=9,
            )
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # ------------------------------------------------------------------
    # 4. Diagonal persistence bars (with CI error bars)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    diag_mean = np.diag(P_mean)
    diag_lo   = np.diag(P_lower)
    diag_hi   = np.diag(P_upper)
    bars = ax4.bar(
        range(K), diag_mean, color=colors, alpha=0.8,
        yerr=[np.maximum(diag_mean - diag_lo, 0), np.maximum(diag_hi - diag_mean, 0)],
        capsize=6, error_kw={"linewidth": 1.5},
    )
    ax4.set_title("Regime Persistence  P(k→k)", fontweight="bold")
    ax4.set_xticks(range(K))
    ax4.set_xticklabels(state_labels)
    ax4.set_ylabel("Probability")
    ax4.set_ylim(0, 1.05)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, diag_mean):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02, f"{v:.3f}",
            ha="center", fontsize=9,
        )

    # ------------------------------------------------------------------
    # 5. Expected regime duration
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    dur_mean   = 1.0 / (1.0 - np.diag(P_mean).clip(0, 0.9999))
    # Higher P_stay → longer duration: P_upper → dur_ci_hi, P_lower → dur_ci_lo
    dur_ci_lo  = 1.0 / (1.0 - np.diag(P_lower).clip(0, 0.9999))
    dur_ci_hi  = 1.0 / (1.0 - np.diag(P_upper).clip(0, 0.9999))
    bars2 = ax5.bar(
        range(K), dur_mean, color=colors, alpha=0.8,
        yerr=[np.maximum(dur_mean - dur_ci_lo, 0), np.maximum(dur_ci_hi - dur_mean, 0)],
        capsize=6, error_kw={"linewidth": 1.5},
    )
    ax5.set_title("Expected Duration  E[τ_k]  (months)", fontweight="bold")
    ax5.set_xticks(range(K))
    ax5.set_xticklabels(state_labels)
    ax5.set_ylabel("Months")
    ax5.grid(True, alpha=0.3, axis="y")
    y_pad = 0.2 + np.maximum(dur_ci_hi - dur_mean, 0)
    for bar, v, pad in zip(bars2, dur_mean, y_pad):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            v + pad + 0.1, f"{v:.1f}",
            ha="center", fontsize=9,
        )

    # ------------------------------------------------------------------
    # 6. Time in each regime across REITs (stacked bar)
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    tickers = list(ticker_stats.keys())
    bottom = np.zeros(len(tickers))
    for s_idx, state in enumerate(STATES):
        pcts = [ticker_stats[t]["regime_pct"].get(state, 0.0) for t in tickers]
        ax6.bar(
            tickers, pcts, bottom=bottom,
            color=colors[s_idx], alpha=0.85, label=state.capitalize(),
        )
        bottom += np.array(pcts)
    ax6.set_title("Time in Each Regime per REIT", fontweight="bold")
    ax6.set_ylabel("Percentage (%)")
    ax6.set_ylim(0, 105)
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "bayesian_cre_transition.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Visualization saved: {output_path}")
    plt.show()


# =============================================================================
# Reporting
# =============================================================================

def print_matrix(label: str, M: np.ndarray) -> None:
    """Pretty-print a K×K matrix with state labels."""
    print(f"\n{label}")
    header = f"{'From \\ To':<12}" + "".join(f"{s.capitalize():<12}" for s in STATES)
    print(header)
    print("-" * len(header))
    for i, from_state in enumerate(STATES):
        row = f"{from_state.capitalize():<12}" + "".join(
            f"{M[i, j]:<12.4f}" for j in range(K)
        )
        print(row)


def print_regime_durations(P: np.ndarray) -> None:
    """Print expected duration (months) in each regime."""
    print("\nExpected Regime Duration (months):")
    for k, state in enumerate(STATES):
        p_stay = P[k, k]
        duration = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")
        print(f"  {state.capitalize():<10}: {duration:.1f} months  (P_stay={p_stay:.3f})")


def print_python_format(P: np.ndarray) -> None:
    """Print the matrix in drop-in format for RegimeMarkovChain."""
    print("\n# ── Drop-in format for RegimeMarkovChain ──────────────────────────")
    print("BAYESIAN_CRE_TRANSITION_MATRIX = [")
    for i, state in enumerate(STATES):
        row = [round(P[i, j], 6) for j in range(K)]
        print(f"    {row},  # from {state}")
    print("]")
    print("# States: calm=0, neutral=1, volatile=2, panic=3")


# =============================================================================
# Main
# =============================================================================

def main() -> Dict:
    """
    Full pipeline:
        1. Fetch monthly prices for each REIT in the basket.
        2. Classify each REIT's history into regime sequences.
        3. Pool transition counts across all REITs.
        4. Apply Dirichlet prior and compute Bayesian posterior.
        5. Report results and save visualization.

    Returns
    -------
    dict with keys:
        P_mean     – posterior mean transition matrix (K x K ndarray)
        P_lower    – 95% CI lower bound
        P_upper    – 95% CI upper bound
        samples    – posterior samples  (N_POSTERIOR_SAMPLES x K x K)
        counts     – pooled raw counts  (K x K)
    """
    print("=" * 70)
    print("  BAYESIAN POOLED CRE TRANSITION MATRIX ESTIMATOR")
    print("=" * 70)

    print(f"\nREIT basket ({len(CRE_REITS)} tickers):")
    for ticker, name in CRE_REITS.items():
        print(f"  {ticker:<6} {name}")

    print(f"\nPrior: sticky Dirichlet  α_diag={PRIOR_PERSISTENCE}  α_offdiag={PRIOR_OFFDIAG}")
    print(f"Data:  monthly, {START_DATE[:10]} → {END_DATE[:10]}")

    # ------------------------------------------------------------------
    # Step 1: Fetch and process each REIT
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("Fetching data ...")
    print("─" * 70)

    alpha_prior = build_prior()
    pooled_counts = np.zeros((K, K), dtype=float)
    ticker_stats: Dict[str, Dict] = {}
    loaded_tickers: List[str] = []

    for ticker, name in CRE_REITS.items():
        df = load_reit(ticker)
        if df is None:
            continue

        prices = df["price"].tolist()
        regimes = regimes_from_prices(prices)
        counts = count_transitions(regimes)
        pooled_counts += counts
        loaded_tickers.append(ticker)

        # Per-ticker stats
        total = len(regimes)
        regime_pct = {
            s: 100.0 * regimes.count(s) / total for s in STATES
        }
        ticker_stats[ticker] = {
            "n_months": len(df),
            "n_regimes": total,
            "regime_pct": regime_pct,
            "counts": counts,
        }

    if not loaded_tickers:
        print("\nNo REIT data could be loaded. Check API key / network.")
        return {}

    print(f"\nLoaded {len(loaded_tickers)}/{len(CRE_REITS)} REITs: "
          f"{', '.join(loaded_tickers)}")
    print(f"Total pooled transition observations: {int(pooled_counts.sum())}")

    # ------------------------------------------------------------------
    # Step 2: Bayesian posterior
    # ------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("Computing Bayesian posterior ...")
    print("─" * 70)

    P_mean = posterior_mean(alpha_prior, pooled_counts)
    samples = posterior_samples(alpha_prior, pooled_counts, n_samples=N_POSTERIOR_SAMPLES)
    P_lower, P_upper = credible_intervals(samples, level=CI_LEVEL)

    # ------------------------------------------------------------------
    # Step 3: Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print_matrix("Posterior Mean Transition Matrix:", P_mean)

    print(f"\n95% Credible Interval (lower):")
    print_matrix("", P_lower)
    print(f"\n95% Credible Interval (upper):")
    print_matrix("", P_upper)

    print("\n" + "─" * 70)
    print_regime_durations(P_mean)

    print("\n" + "─" * 70)
    print("\nPer-REIT regime time distribution (%):")
    print(f"{'Ticker':<8}" + "".join(f"{s.capitalize():<12}" for s in STATES))
    print("-" * (8 + 12 * K))
    for ticker in loaded_tickers:
        pct = ticker_stats[ticker]["regime_pct"]
        row = f"{ticker:<8}" + "".join(f"{pct.get(s, 0):<12.1f}" for s in STATES)
        print(row)

    print("\n" + "─" * 70)
    print(f"\nPooled transition counts across all REITs:")
    print_matrix("", pooled_counts)

    print("\n" + "=" * 70)
    print_python_format(P_mean)
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 4: Visualize
    # ------------------------------------------------------------------
    output_dir = parent_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    try:
        plot_results(P_mean, P_lower, P_upper, pooled_counts, ticker_stats, output_dir)
    except Exception as exc:
        print(f"\nVisualization failed: {exc}")

    # ------------------------------------------------------------------
    # Step 5: Save posterior matrix as npz for downstream use
    # ------------------------------------------------------------------
    npz_path = output_dir / "bayesian_cre_transition.npz"
    np.savez(
        npz_path,
        P_mean=P_mean,
        P_lower=P_lower,
        P_upper=P_upper,
        samples=samples,
        pooled_counts=pooled_counts,
        states=np.array(STATES),
    )
    print(f"\n  Posterior saved: {npz_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return {
        "P_mean": P_mean,
        "P_lower": P_lower,
        "P_upper": P_upper,
        "samples": samples,
        "counts": pooled_counts,
    }


if __name__ == "__main__":
    results = main()
