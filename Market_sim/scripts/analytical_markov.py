"""
Layer 1: Analytical Markov Chain Analysis

Computes exact results from the empirical transition matrices:
  1. Stationary distributions (long-run % time in each regime)
  2. Mean first passage times (expected months to reach each state from each state)
  3. Expected sojourn times (expected consecutive months in a state per visit)
  4. Bootstrap confidence intervals on the stationary distributions

These are mathematically exact results — no simulation needed.

Data sources:
  - P_TRADITIONAL: assumed illiquid CRE matrix
  - P_TOKENIZED:   empirical from VNQ ETF (2005-2026, 253 months)
  - Observation counts inferred from regime frequency reports in FINDINGS_SUMMARY.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# TRANSITION MATRICES & STATE METADATA
# =============================================================================

STATES = ["Calm", "Neutral", "Volatile", "Panic"]
STATE_COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

P_TRADITIONAL = np.array([
    [0.85, 0.14, 0.01, 0.00],
    [0.10, 0.75, 0.14, 0.01],
    [0.02, 0.18, 0.70, 0.10],
    [0.01, 0.09, 0.30, 0.60],
])

P_TOKENIZED = np.array([
    [0.8174, 0.1739, 0.0087, 0.0000],
    [0.1887, 0.7736, 0.0283, 0.0094],
    [0.0500, 0.2000, 0.7500, 0.0000],
    [0.0000, 0.0000, 0.1000, 0.9000],
])

# Approximate observation counts used for bootstrap CI estimation.
# Traditional CRE: 871 months, regime split from CRE_BASELINE_ANALYSIS.md
#   calm=58.7%, neutral=34.0%, volatile=6.8%, panic=0.5%
COUNTS_TRADITIONAL = np.array([511, 296, 59, 4])

# Tokenized/VNQ: 253 months, regime split from FINDINGS_SUMMARY.md
#   calm=46.0%, neutral=42.1%, volatile=7.9%, panic=4.0%
COUNTS_TOKENIZED = np.array([116, 107, 20, 10])


# =============================================================================
# ANALYTICAL COMPUTATIONS
# =============================================================================

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution π satisfying π P = π, Σπ_i = 1.

    Approach: rewrite as a linear system (P^T - I)π = 0, replace the last
    equation with the normalization constraint Σπ_i = 1, then solve.
    """
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    return pi


def mean_first_passage_times(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Compute the mean first passage time matrix M[i, j] = expected number of
    steps to first reach state j starting from state i.

    Uses the fundamental matrix Z = (I - P + Π)^{-1} where Π has each row
    equal to the stationary distribution.

    Diagonal: M[i,i] = 1 / π_i  (mean return time)
    Off-diagonal: M[i,j] = (Z[j,j] - Z[i,j]) / π_j
    """
    n = P.shape[0]
    Pi = np.tile(pi, (n, 1))
    Z = np.linalg.inv(np.eye(n) - P + Pi)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0 / pi[j]
            else:
                M[i, j] = (Z[j, j] - Z[i, j]) / pi[j]
    return M


def expected_sojourn_times(P: np.ndarray) -> np.ndarray:
    """
    Expected consecutive months spent in state i before leaving.
    = 1 / (1 - P[i, i])
    """
    return np.array([1.0 / (1.0 - P[i, i]) for i in range(P.shape[0])])


def bootstrap_stationary_ci(
    P: np.ndarray,
    counts: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> tuple:
    """
    Bootstrap confidence intervals for the stationary distribution.

    For each row i of P, the transition counts follow a Dirichlet-multinomial.
    We resample using Dirichlet(alpha = observed_row_counts + 0.5) as a
    Bayesian pseudocount smoothing approach, then recompute the stationary
    distribution for each bootstrap replicate.

    Returns (lower, upper) arrays of shape (4,).
    """
    n = P.shape[0]
    boot_pi = np.zeros((n_boot, n))
    rng = np.random.default_rng(42)

    for b in range(n_boot):
        P_boot = np.zeros((n, n))
        for i in range(n):
            alpha = counts[i] * P[i] + 0.5
            alpha = np.maximum(alpha, 0.01)
            P_boot[i] = rng.dirichlet(alpha)
        boot_pi[b] = stationary_distribution(P_boot)

    alpha_tail = (1.0 - ci) / 2.0
    lower = np.percentile(boot_pi, alpha_tail * 100, axis=0)
    upper = np.percentile(boot_pi, (1.0 - alpha_tail) * 100, axis=0)
    return lower, upper


# =============================================================================
# CONSOLE REPORTING
# =============================================================================

def _print_matrix(P: np.ndarray, name: str) -> None:
    print(f"\n{name}:")
    header = f"{'From \\ To':<12}" + "".join(f"{s:>10}" for s in STATES)
    print(header)
    print("-" * len(header))
    for i, state in enumerate(STATES):
        row = "".join(f"{P[i, j]:>10.4f}" for j in range(4))
        print(f"{state:<12}{row}")


def print_full_results(
    label: str,
    pi: np.ndarray,
    M: np.ndarray,
    sojourn: np.ndarray,
    pi_lo: np.ndarray,
    pi_hi: np.ndarray,
) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  Stationary Distribution (long-run % time in each regime):")
    print(f"  {'State':<12}{'π':<10}{'95% CI Lower':<16}{'95% CI Upper':<16}{'Sojourn (mo)'}")
    print(f"  {'-'*60}")
    for i, s in enumerate(STATES):
        print(
            f"  {s:<12}{pi[i]*100:>7.2f}%   "
            f"[{pi_lo[i]*100:>6.2f}%, {pi_hi[i]*100:>6.2f}%]"
            f"   {sojourn[i]:>8.2f} mo"
        )

    print(f"\n  Aggregates:")
    print(f"    Time in Calm/Neutral (healthy): {(pi[0]+pi[1])*100:.2f}%")
    print(f"    Time in Volatile/Panic (stress): {(pi[2]+pi[3])*100:.2f}%")

    print(f"\n  Mean First Passage Times (months):")
    header = f"  {'From \\ To':<12}" + "".join(f"{s:>10}" for s in STATES)
    print(header)
    print(f"  {'-'*56}")
    for i, state in enumerate(STATES):
        row = "".join(f"{M[i, j]:>10.1f}" for j in range(4))
        print(f"  {state:<12}{row}")

    print(f"\n  Key first-passage highlights:")
    print(f"    Calm  → Panic:   {M[0, 3]:>6.1f} months expected")
    print(f"    Neutral → Panic: {M[1, 3]:>6.1f} months expected")
    print(f"    Panic → Neutral: {M[3, 1]:>6.1f} months expected")
    print(f"    Panic → Calm:    {M[3, 0]:>6.1f} months expected")


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(
    pi_trad: np.ndarray, pi_tok: np.ndarray,
    pi_trad_lo: np.ndarray, pi_trad_hi: np.ndarray,
    pi_tok_lo: np.ndarray, pi_tok_hi: np.ndarray,
    M_trad: np.ndarray, M_tok: np.ndarray,
    sojourn_trad: np.ndarray, sojourn_tok: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Layer 1: Analytical Markov Chain Results\nTraditional vs Tokenized CRE",
        fontsize=15, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    x = np.arange(4)
    w = 0.35

    # ── Panel 1: Stationary distribution with bootstrap CIs ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    yerr_t = [pi_trad * 100 - pi_trad_lo * 100, pi_trad_hi * 100 - pi_trad * 100]
    yerr_k = [pi_tok * 100 - pi_tok_lo * 100,   pi_tok_hi * 100 - pi_tok * 100]
    ax1.bar(x - w/2, pi_trad * 100, w, label="Traditional", color="#2c3e50",
            alpha=0.85, yerr=yerr_t, capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "#666"})
    ax1.bar(x + w/2, pi_tok * 100,  w, label="Tokenized",   color="#16a085",
            alpha=0.85, yerr=yerr_k, capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "#666"})
    ax1.set_title("Stationary Distribution\n(± 95% Bootstrap CI)", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(STATES, fontsize=9)
    ax1.set_ylabel("Long-run time (%)"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Panel 2: Healthy vs Stressed aggregation ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cats = ["Calm/Neutral\n(Healthy)", "Volatile/Panic\n(Stressed)"]
    t_agg = [(pi_trad[0]+pi_trad[1])*100, (pi_trad[2]+pi_trad[3])*100]
    k_agg = [(pi_tok[0]+pi_tok[1])*100,   (pi_tok[2]+pi_tok[3])*100]
    x2 = np.arange(2)
    b1 = ax2.bar(x2 - w/2, t_agg, w, label="Traditional", color="#e74c3c", alpha=0.85)
    b2 = ax2.bar(x2 + w/2, k_agg, w, label="Tokenized",   color="#27ae60", alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Healthy vs Stressed\n(Analytical)", fontweight="bold")
    ax2.set_xticks(x2); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylabel("Long-run time (%)"); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: MFPT heatmap — Traditional ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    vmax = max(M_trad.max(), M_tok.max())
    im3 = ax3.imshow(np.clip(M_trad, 0, vmax), cmap="YlOrRd", aspect="auto", vmax=vmax)
    ax3.set_xticks(range(4)); ax3.set_xticklabels(STATES, fontsize=8)
    ax3.set_yticks(range(4)); ax3.set_yticklabels(STATES, fontsize=8)
    ax3.set_title("Mean First Passage Times\nTraditional (months)", fontweight="bold")
    ax3.set_xlabel("To State"); ax3.set_ylabel("From State")
    for i in range(4):
        for j in range(4):
            v = M_trad[i, j]
            c = "white" if v > vmax * 0.55 else "black"
            ax3.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im3, ax=ax3, shrink=0.8, label="months")

    # ── Panel 4: MFPT heatmap — Tokenized ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(np.clip(M_tok, 0, vmax), cmap="YlGn", aspect="auto", vmax=vmax)
    ax4.set_xticks(range(4)); ax4.set_xticklabels(STATES, fontsize=8)
    ax4.set_yticks(range(4)); ax4.set_yticklabels(STATES, fontsize=8)
    ax4.set_title("Mean First Passage Times\nTokenized (months)", fontweight="bold")
    ax4.set_xlabel("To State"); ax4.set_ylabel("From State")
    for i in range(4):
        for j in range(4):
            v = M_tok[i, j]
            c = "white" if v > vmax * 0.55 else "black"
            ax4.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im4, ax=ax4, shrink=0.8, label="months")

    # ── Panel 5: Sojourn times ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x - w/2, sojourn_trad, w, label="Traditional", color="#2c3e50", alpha=0.85)
    ax5.bar(x + w/2, sojourn_tok,  w, label="Tokenized",   color="#16a085", alpha=0.85)
    for i in range(4):
        ax5.text(i - w/2, sojourn_trad[i] + 0.05, f"{sojourn_trad[i]:.1f}",
                 ha="center", va="bottom", fontsize=8)
        ax5.text(i + w/2, sojourn_tok[i] + 0.05, f"{sojourn_tok[i]:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax5.set_title("Expected Sojourn Time\n(months per visit)", fontweight="bold")
    ax5.set_xticks(x); ax5.set_xticklabels(STATES, fontsize=9)
    ax5.set_ylabel("Months"); ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis="y")

    # ── Panel 6: Key passage times bar ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    passages = [
        ("Calm→Panic",   M_trad[0, 3], M_tok[0, 3]),
        ("Neutral→Panic",M_trad[1, 3], M_tok[1, 3]),
        ("Panic→Neutral",M_trad[3, 1], M_tok[3, 1]),
        ("Panic→Calm",   M_trad[3, 0], M_tok[3, 0]),
    ]
    pl = [p[0] for p in passages]
    pt = [p[1] for p in passages]
    pk = [p[2] for p in passages]
    xp = np.arange(len(pl))
    ax6.bar(xp - w/2, pt, w, label="Traditional", color="#2c3e50", alpha=0.85)
    ax6.bar(xp + w/2, pk, w, label="Tokenized",   color="#16a085", alpha=0.85)
    ax6.set_title("Key Mean First Passage Times", fontweight="bold")
    ax6.set_xticks(xp); ax6.set_xticklabels(pl, rotation=28, ha="right", fontsize=8)
    ax6.set_ylabel("Expected months"); ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis="y")

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_path = outputs_dir / "layer1_analytical_markov.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Plot saved: {out_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  LAYER 1: ANALYTICAL MARKOV CHAIN ANALYSIS")
    print("  Exact results from empirical transition matrices")
    print("=" * 70)

    _print_matrix(P_TRADITIONAL, "P_TRADITIONAL (Assumed illiquid CRE)")
    _print_matrix(P_TOKENIZED,   "P_TOKENIZED (Empirical VNQ ETF, 2005-2026)")

    pi_trad    = stationary_distribution(P_TRADITIONAL)
    pi_tok     = stationary_distribution(P_TOKENIZED)
    M_trad     = mean_first_passage_times(P_TRADITIONAL, pi_trad)
    M_tok      = mean_first_passage_times(P_TOKENIZED,   pi_tok)
    soj_trad   = expected_sojourn_times(P_TRADITIONAL)
    soj_tok    = expected_sojourn_times(P_TOKENIZED)

    print("\n  Computing bootstrap confidence intervals (10 000 resamples)…")
    pi_trad_lo, pi_trad_hi = bootstrap_stationary_ci(P_TRADITIONAL, COUNTS_TRADITIONAL)
    pi_tok_lo,  pi_tok_hi  = bootstrap_stationary_ci(P_TOKENIZED,   COUNTS_TOKENIZED)

    print_full_results("TRADITIONAL CRE (Illiquid)",
                       pi_trad, M_trad, soj_trad, pi_trad_lo, pi_trad_hi)
    print_full_results("TOKENIZED CRE (REIT-like, Empirical VNQ)",
                       pi_tok,  M_tok,  soj_tok,  pi_tok_lo,  pi_tok_hi)

    print(f"\n{'='*70}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*70}")
    stress_t = (pi_trad[2] + pi_trad[3]) * 100
    stress_k = (pi_tok[2]  + pi_tok[3])  * 100
    print(f"  Stress-time reduction:       {stress_t:.2f}% → {stress_k:.2f}%  "
          f"(−{stress_t - stress_k:.2f} pp)")
    print(f"  Calm→Panic passage:          {M_trad[0,3]:.1f} → {M_tok[0,3]:.1f} months")
    print(f"  Panic→Neutral passage:       {M_trad[3,1]:.1f} → {M_tok[3,1]:.1f} months")
    print(f"  Panic sojourn:               {soj_trad[3]:.1f} → {soj_tok[3]:.1f} months/visit")

    plot_results(
        pi_trad, pi_tok,
        pi_trad_lo, pi_trad_hi,
        pi_tok_lo,  pi_tok_hi,
        M_trad, M_tok,
        soj_trad, soj_tok,
    )
    print("\n  Layer 1 complete.")


if __name__ == "__main__":
    main()
