"""
Layer 3: Adoption Sensitivity Analysis

This script does two things:

  Part A — Empirical adoption curve (RWA data)
    Fits a logistic sigmoid to the observed RWA "Bridged Token Value"
    time-series (May 2023 – March 2026).  The fitted parameters (growth
    rate k, inflection point t0, saturation L) are used to:
      • Plot the empirical adoption trajectory alongside the fitted curve.
      • Estimate the current "maturity" level α_current = V_now / L.

  Part B — Sensitivity sweep (α from 0 → 1)
    Defines the interpolated transition matrix
        P(α) = (1 − α) × P_TRADITIONAL + α × P_TOKENIZED
    and computes the stationary distribution analytically for each α ∈ [0, 1].
    Plots how key market-health metrics evolve as tokenization increases,
    and marks where the fitted adoption curve places the market today and
    at several future horizons.

Data:
  rwa-token-timeseries-export-*.csv  (Bridged Token Value, daily)
  P_TOKENIZED: Bayesian posterior mean from outputs/bayesian_cre_transition.npz
               (run bayesian_cre_transition.py first)
"""

import sys
import csv
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# =============================================================================
# CONFIGURATION
# =============================================================================

STATES = ["Calm", "Neutral", "Volatile", "Panic"]

P_TRADITIONAL = np.array([
    [0.85, 0.14, 0.01, 0.00],
    [0.10, 0.75, 0.14, 0.01],
    [0.02, 0.18, 0.70, 0.10],
    [0.01, 0.09, 0.30, 0.60],
])

_BAYESIAN_NPZ = Path(__file__).resolve().parent.parent / "outputs" / "bayesian_cre_transition.npz"
if not _BAYESIAN_NPZ.exists():
    raise SystemExit(
        f"Missing {_BAYESIAN_NPZ.resolve()}\n"
        "Generate it with: python3 scripts/bayesian_cre_transition.py"
    )
P_TOKENIZED = np.asarray(np.load(_BAYESIAN_NPZ)["P_mean"], dtype=float)

# Path to the RWA dataset (copy in data/ directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR   = _SCRIPT_DIR.parent / "data"
RWA_CSV     = _DATA_DIR / "rwa-token-timeseries-export-1774989038960.csv"


# =============================================================================
# PART A — RWA ADOPTION CURVE
# =============================================================================

def load_rwa_monthly(csv_path: Path) -> tuple:
    """
    Load the RWA CSV and resample to monthly end-of-period values.

    Returns (months_from_start, tvl_values, month_labels) where
    months_from_start[0] = 0 corresponds to the first month in the file.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "date":  datetime.strptime(r["Date"], "%Y-%m-%d").date(),
                "value": float(r["Real Estate"]),
            })

    # Group by year-month, take last value in each month
    monthly: dict = {}
    for r in rows:
        ym = (r["date"].year, r["date"].month)
        monthly[ym] = r["value"]

    sorted_months = sorted(monthly.keys())
    t0_ym         = sorted_months[0]
    t0_date       = date(t0_ym[0], t0_ym[1], 1)

    months_rel = []
    tvl        = []
    labels     = []

    for ym in sorted_months:
        d = date(ym[0], ym[1], 1)
        m = (ym[0] - t0_ym[0]) * 12 + (ym[1] - t0_ym[1])
        months_rel.append(m)
        tvl.append(monthly[ym])
        labels.append(f"{ym[0]}-{ym[1]:02d}")

    return np.array(months_rel, dtype=float), np.array(tvl, dtype=float), labels, t0_date


def logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    """Standard logistic function: L / (1 + exp(-k*(t - t0)))"""
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_adoption_sigmoid(t: np.ndarray, v: np.ndarray) -> tuple:
    """
    Fit the logistic model to (t, v) and return (popt, pcov).

    Initial guesses:
      L  = 2 × current maximum  (market hasn't saturated)
      k  = 0.12 monthly growth
      t0 = month index near the observed inflection (~Feb 2025 = month 21)
    """
    L_guess  = v.max() * 2.5
    k_guess  = 0.12
    t0_guess = 21.0          # Feb 2025 in the 35-month series

    popt, pcov = curve_fit(
        logistic, t, v,
        p0=[L_guess, k_guess, t0_guess],
        maxfev=20_000,
        bounds=([v.max(), 0.001, 0.0], [v.max() * 50, 2.0, 50.0]),
    )
    return popt, pcov


# =============================================================================
# PART B — SENSITIVITY SWEEP
# =============================================================================

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Solve π P = π, Σπ = 1."""
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def expected_sojourn(P: np.ndarray) -> np.ndarray:
    return np.array([1.0 / (1.0 - P[i, i]) for i in range(P.shape[0])])


def mean_first_passage_times(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    Pi = np.tile(pi, (n, 1))
    Z  = np.linalg.inv(np.eye(n) - P + Pi)
    M  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 / pi[j] if i == j else (Z[j, j] - Z[i, j]) / pi[j]
    return M


def sensitivity_sweep(n_alpha: int = 201) -> dict:
    """
    Compute analytical Markov metrics for P(α) = (1-α)P_TRAD + α P_TOK
    over α ∈ [0, 1].

    Returns a dict of metric arrays indexed by alpha.
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)

    stress_pct       = np.zeros(n_alpha)
    calm_neutral_pct = np.zeros(n_alpha)
    panic_sojourn    = np.zeros(n_alpha)
    panic_to_neutral = np.zeros(n_alpha)
    calm_to_panic    = np.zeros(n_alpha)

    for idx, a in enumerate(alphas):
        P   = (1.0 - a) * P_TRADITIONAL + a * P_TOKENIZED
        pi  = stationary_distribution(P)
        M   = mean_first_passage_times(P, pi)
        soj = expected_sojourn(P)

        stress_pct[idx]       = (pi[2] + pi[3]) * 100.0
        calm_neutral_pct[idx] = (pi[0] + pi[1]) * 100.0
        panic_sojourn[idx]    = soj[3]
        panic_to_neutral[idx] = M[3, 1]
        calm_to_panic[idx]    = M[0, 3]

    return {
        "alphas":          alphas,
        "stress_pct":      stress_pct,
        "calm_neutral_pct":calm_neutral_pct,
        "panic_sojourn":   panic_sojourn,
        "panic_to_neutral":panic_to_neutral,
        "calm_to_panic":   calm_to_panic,
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_adoption_curve(
    t_obs: np.ndarray,
    v_obs: np.ndarray,
    labels: list,
    popt: tuple,
    t0_date: date,
) -> None:
    L, k, t0_fit = popt
    t_fine = np.linspace(0, t_obs.max() + 24, 500)
    v_fine = logistic(t_fine, L, k, t0_fit)

    # Convert month index to calendar dates for x-axis labels
    def to_date(m):
        yr  = t0_date.year + (t0_date.month - 1 + int(m)) // 12
        mo  = (t0_date.month - 1 + int(m)) % 12 + 1
        return date(yr, mo, 1)

    tick_months = np.arange(0, t_obs.max() + 25, 6)
    tick_labels = [to_date(m).strftime("%b %Y") for m in tick_months]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.scatter(t_obs, v_obs / 1e6, color="#2c3e50", s=30, zorder=5,
               label="Observed monthly TVL")
    ax.plot(t_fine, v_fine / 1e6, color="#e74c3c", linewidth=2.5,
            label=f"Fitted logistic  (L=${L/1e6:.0f}M, k={k:.3f}/mo, t₀=mo {t0_fit:.1f})")
    ax.axhline(L / 1e6, color="#e74c3c", linewidth=1.0, linestyle=":",
               alpha=0.6, label=f"Saturation L = ${L/1e6:.0f} M")

    ax.set_xticks(tick_months)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Part A — Empirical RWA Tokenised Real Estate TVL\n"
                 "with Fitted Logistic Adoption Curve", fontweight="bold")
    ax.set_ylabel("TVL (USD millions)")
    ax.set_xlabel("Calendar month")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mark current position
    now_m = t_obs[-1]
    now_v = v_obs[-1]
    alpha_now = logistic(now_m, 1.0, k, t0_fit)   # normalised ∈ (0,1)
    ax.annotate(
        f"Today  α ≈ {alpha_now:.2f}\n${now_v/1e6:.1f} M",
        xy=(now_m, now_v / 1e6),
        xytext=(now_m - 6, now_v / 1e6 + L / 1e6 * 0.08),
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        fontsize=9, color="#e74c3c", fontweight="bold",
    )

    plt.tight_layout()
    outputs_dir = _SCRIPT_DIR.parent / "outputs"
    out = outputs_dir / "layer3_adoption_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {out}")
    plt.show()


def plot_sensitivity(sweep: dict, popt: tuple, t_obs: np.ndarray) -> None:
    """Sensitivity curves + empirical adoption markers."""
    L, k, t0_fit = popt
    alphas = sweep["alphas"]

    # Compute α at today (end of observed series) and at +2yr / +5yr
    now_m   = t_obs[-1]
    future_marks = {
        "Today (Mar 2026)":   (now_m,      "#e74c3c"),
        "+2 yr (Mar 2028)":   (now_m + 24, "#f39c12"),
        "+5 yr (Mar 2031)":   (now_m + 60, "#27ae60"),
    }
    alpha_marks = {
        lbl: logistic(m, 1.0, k, t0_fit)
        for lbl, (m, _) in future_marks.items()
    }

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(
        "Layer 3: Sensitivity Analysis — Market Outcomes as a Function of Tokenisation Adoption\n"
        "P(α) = (1 − α) × P_TRADITIONAL + α × P_TOKENIZED",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    panels = [
        ("stress_pct",       "Time in Stress (Volatile+Panic) (%)", "Stress time (%)",      "lower is better"),
        ("calm_neutral_pct", "Time in Calm/Neutral (%)",            "Healthy time (%)",     "higher is better"),
        ("panic_sojourn",    "Expected Sojourn in Panic (months)",   "Sojourn (months)",    "lower is better"),
        ("panic_to_neutral", "Panic → Neutral Passage Time (months)","Passage time (mo)",   "lower is better"),
        ("calm_to_panic",    "Calm → Panic Passage Time (months)",   "Passage time (mo)",   "higher is better"),
    ]

    for idx, (key, title, ylabel, direction) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        y  = sweep[key]

        ax.plot(alphas, y, color="#2980b9", linewidth=2.5)
        ax.fill_between(alphas, y, y.min() if "lower" in direction else y.max(),
                        alpha=0.12, color="#2980b9")

        # Mark end-points
        ax.scatter([0, 1], [y[0], y[-1]], color=["#e74c3c", "#27ae60"],
                   s=80, zorder=6, label=["α=0 (Traditional)", "α=1 (Tokenized)"])
        ax.text(0.01, y[0], f" α=0\n {y[0]:.1f}", fontsize=7.5, va="center", color="#e74c3c")
        ax.text(0.99, y[-1], f"α=1 \n{y[-1]:.1f} ", fontsize=7.5, va="center",
                ha="right", color="#27ae60")

        # Adoption markers
        for lbl, (_, col) in future_marks.items():
            a_m = alpha_marks[lbl]
            if 0 <= a_m <= 1:
                y_m = float(interp1d(alphas, y)(a_m))
                ax.axvline(a_m, color=col, linewidth=1.2, linestyle="--", alpha=0.8)
                ax.scatter([a_m], [y_m], color=col, s=60, zorder=7)
                ax.text(a_m + 0.01, y_m, f" {lbl.split('(')[0].strip()}\n {y_m:.1f}",
                        fontsize=6.5, color=col)

        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel("Adoption level α", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # ── Panel 6: summary table ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    rows = [["Scenario", "α", "Stress %", "MFPT Panic→N"]]
    rows.append(["Traditional (α=0)", "0.00",
                 f"{sweep['stress_pct'][0]:.1f}%",
                 f"{sweep['panic_to_neutral'][0]:.1f} mo"])
    for lbl, (m, _) in future_marks.items():
        a = alpha_marks[lbl]
        if 0 <= a <= 1:
            s = float(interp1d(alphas, sweep["stress_pct"])(a))
            p = float(interp1d(alphas, sweep["panic_to_neutral"])(a))
            rows.append([lbl, f"{a:.2f}", f"{s:.1f}%", f"{p:.1f} mo"])
    rows.append(["Tokenized (α=1)", "1.00",
                 f"{sweep['stress_pct'][-1]:.1f}%",
                 f"{sweep['panic_to_neutral'][-1]:.1f} mo"])

    tbl = ax6.table(cellText=rows, cellLoc="center", loc="center",
                    colWidths=[0.38, 0.12, 0.22, 0.28])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.2)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    ax6.set_title("Summary Table", fontweight="bold", pad=10)

    outputs_dir = _SCRIPT_DIR.parent / "outputs"
    out = outputs_dir / "layer3_sensitivity_sweep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {out}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  LAYER 3: ADOPTION SENSITIVITY ANALYSIS")
    print("=" * 70)

    # ── Part A: RWA adoption curve ────────────────────────────────────────────
    print(f"\n  [Part A] Loading RWA data from:\n    {RWA_CSV}")
    if not RWA_CSV.exists():
        print(f"\n  WARNING: RWA CSV not found at {RWA_CSV}")
        print("  Proceeding with sensitivity sweep only (Part B).")
        popt = (1e9, 0.12, 21.0)     # fallback guesses
        t_obs = np.array([34.0])
        run_part_a = False
    else:
        t_obs, v_obs, labels, t0_date = load_rwa_monthly(RWA_CSV)
        print(f"  Loaded {len(t_obs)} monthly observations")
        print(f"  Date range: {labels[0]} → {labels[-1]}")
        print(f"  TVL range:  ${v_obs.min()/1e6:.2f}M → ${v_obs.max()/1e6:.2f}M")

        print("\n  Fitting logistic adoption curve…")
        popt, pcov = fit_adoption_sigmoid(t_obs, v_obs)
        L_fit, k_fit, t0_fit = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"  Fitted parameters:")
        print(f"    L  (saturation TVL): ${L_fit/1e6:.1f} M  ± ${perr[0]/1e6:.1f} M")
        print(f"    k  (monthly growth): {k_fit:.4f}  ± {perr[1]:.4f}")
        print(f"    t₀ (inflection mo):  {t0_fit:.1f}  ± {perr[2]:.1f}")

        now_alpha = logistic(t_obs[-1], 1.0, k_fit, t0_fit)
        print(f"\n  Current adoption level (normalised): α ≈ {now_alpha:.3f}")
        print(f"  Implied doublingtime:  ≈ {np.log(2)/k_fit:.1f} months")

        plot_adoption_curve(t_obs, v_obs, labels, popt, t0_date)
        run_part_a = True

    # ── Part B: sensitivity sweep ─────────────────────────────────────────────
    print("\n  [Part B] Running sensitivity sweep (α = 0 → 1)…")
    sweep = sensitivity_sweep(n_alpha=201)

    print(f"\n  Key metrics at α endpoints:")
    print(f"  {'Metric':<35} {'α=0 (Trad.)':>14}  {'α=1 (Tok.)':>14}")
    print(f"  {'-'*65}")
    for key, label in [
        ("stress_pct",       "Stress time (%)"),
        ("calm_neutral_pct", "Calm/Neutral time (%)"),
        ("panic_sojourn",    "Panic sojourn (months)"),
        ("panic_to_neutral", "Panic→Neutral passage (months)"),
        ("calm_to_panic",    "Calm→Panic passage (months)"),
    ]:
        print(f"  {label:<35} {sweep[key][0]:>14.2f}  {sweep[key][-1]:>14.2f}")

    L_fit, k_fit, t0_fit = popt
    now_m = t_obs[-1]
    for horizon_label, dm in [("Today (Mar 2026)", 0),
                               ("+2yr (Mar 2028)",  24),
                               ("+5yr (Mar 2031)",  60)]:
        a = logistic(now_m + dm, 1.0, k_fit, t0_fit)
        s = float(np.interp(a, sweep["alphas"], sweep["stress_pct"]))
        p = float(np.interp(a, sweep["alphas"], sweep["panic_to_neutral"]))
        print(f"\n  {horizon_label}:  α ≈ {a:.3f}  →  stress={s:.1f}%,  "
              f"panic→neutral={p:.1f} mo")

    plot_sensitivity(sweep, popt, t_obs)
    print("\n  Layer 3 complete.")


if __name__ == "__main__":
    main()
