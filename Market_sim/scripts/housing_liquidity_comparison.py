"""
Housing Liquidity Comparison: Traditional vs Tokenized (REIT-like)

This simulation demonstrates what a US housing market with higher liquidity 
(similar to REITs) would look like compared to traditional illiquid housing markets.

Key differences:
- Traditional: Slow regime transitions, gets stuck in panic/volatile states (assumed)
- Tokenized: Bayesian posterior mean from pooled basket (O / NNN / WPC / ADC / VNQ);
  loaded from outputs/bayesian_cre_transition.npz (run bayesian_cre_transition.py first)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports from sim package
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sim.market_simulator import MarketSimulator
from sim.data_loader import load_cre_csv
from sim.calibration import calibrate_from_history
from sim.types import ScenarioParams


# =============================================================================
# TRANSITION MATRICES
# =============================================================================

# Traditional Housing: Illiquid, slow to recover from stress
P_TRADITIONAL = [
    [0.85, 0.14, 0.01, 0.00],  # calm -> stays calm mostly
    [0.10, 0.75, 0.14, 0.01],  # neutral -> moderate persistence
    [0.02, 0.18, 0.70, 0.10],  # volatile -> hard to escape
    [0.01, 0.09, 0.30, 0.60],  # panic -> very persistent (GETS STUCK)
]

# Tokenized endpoint: Bayesian posterior mean only (outputs/bayesian_cre_transition.npz).
_BAYESIAN_NPZ = Path(__file__).resolve().parent.parent / "outputs" / "bayesian_cre_transition.npz"
if not _BAYESIAN_NPZ.exists():
    raise SystemExit(
        f"Missing {_BAYESIAN_NPZ.resolve()}\n"
        "Generate it with: python3 scripts/bayesian_cre_transition.py"
    )
P_TOKENIZED = np.load(_BAYESIAN_NPZ)["P_mean"].tolist()
print(f"  [housing_sim] P_TOKENIZED loaded from Bayesian posterior: {_BAYESIAN_NPZ.name}")


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(prices: List[float], regimes: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive market metrics"""
    prices_arr = np.array(prices)
    
    # Returns
    returns = np.diff(np.log(prices_arr))
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(12)  # Annualized
    
    # Drawdown
    cummax = np.maximum.accumulate(prices_arr)
    drawdown = (prices_arr - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    # Recovery time (time to recover from max drawdown)
    max_dd_idx = np.argmin(drawdown)
    recovery_idx = max_dd_idx
    for i in range(max_dd_idx, len(prices_arr)):
        if prices_arr[i] >= cummax[max_dd_idx]:
            recovery_idx = i
            break
    recovery_time = recovery_idx - max_dd_idx
    
    # Regime statistics
    regime_counts = {r: regimes.count(r) for r in ["calm", "neutral", "volatile", "panic"]}
    regime_percentages = {r: 100.0 * count / len(regimes) for r, count in regime_counts.items()}
    
    # Time in stress (volatile + panic)
    stress_time = regime_percentages["volatile"] + regime_percentages["panic"]
    
    # Average panic duration
    panic_durations = []
    in_panic = False
    panic_start = 0
    for i, regime in enumerate(regimes):
        if regime == "panic" and not in_panic:
            in_panic = True
            panic_start = i
        elif regime != "panic" and in_panic:
            in_panic = False
            panic_durations.append(i - panic_start)
    avg_panic_duration = np.mean(panic_durations) if panic_durations else 0
    
    return {
        "final_price": prices_arr[-1],
        "total_return": (prices_arr[-1] / prices_arr[0] - 1) * 100,
        "volatility": volatility * 100,
        "max_drawdown": max_drawdown * 100,
        "recovery_time": recovery_time,
        "stress_time_pct": stress_time,
        "regime_percentages": regime_percentages,
        "avg_panic_duration": avg_panic_duration,
        "num_panic_episodes": len(panic_durations),
    }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation(
    csv_path: str,
    months_ahead: int,
    ticks_per_candle: int,
    transition_matrix: List[List[float]],
    scenario_name: str,
    adoption_interpolated_markov: bool = False,
    tokenized_endpoint_matrix: List[List[float]] | None = None,
    seed: int = 42,
    verbose: bool = True,
    use_micro_feedback: bool = False,
    regime_micro_weight: float = 0.25,
    fundamental_micro_feedback: float = 0.10,
) -> Tuple[List[float], List[float], List[float], List[str], int]:
    """
    Run a single simulation with specified transition matrix
    
    Returns:
        months: List of month indices
        fund_prices: List of fundamental prices
        micro_prices: List of microstructure prices
        regimes: List of regime labels
        history_len: Number of replay months before projection begins
    """
    # Create new simulator instance
    sim = MarketSimulator()
    sim.rng.seed(seed)
    
    # Load and calibrate
    history = load_cre_csv(csv_path)
    calib = calibrate_from_history(history)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {scenario_name} simulation")
        print(f"{'='*60}")
        print(f"Historical data: {len(history)} months")
        print(f"Calibrated μ (monthly): {calib.mu_monthly:.4f}")
        print(f"Calibrated σ (monthly): {calib.sigma_monthly:.4f}")
        print(f"Annualized return: {calib.mu_annual*100:.2f}%")
        print(f"Annualized volatility: {calib.sigma_annual*100:.2f}%")
    
    # Create scenario (keep base parameters same for fair comparison)
    scenario = ScenarioParams(
        name=scenario_name,
        mu_monthly=calib.mu_monthly,
        sigma_monthly=calib.sigma_monthly,
    )
    
    # Attach history and enable Markov regimes
    sim.attach_history_and_scenario(history, scenario)
    if use_micro_feedback:
        sim.enable_microstructure_feedback(
            regime_micro_weight=regime_micro_weight,
            fundamental_micro_feedback=fundamental_micro_feedback,
        )
    if adoption_interpolated_markov:
        # Adoption should ramp during the projection phase, not during history replay.
        # Centre the sigmoid on the midpoint of the forward window so adoption goes
        # 0→1 across the actual projection period rather than saturating at 1.0
        # before projection even begins.
        projection_start = len(history)
        sim.adoption_midpoint = projection_start + months_ahead // 2
        endpoint = tokenized_endpoint_matrix if tokenized_endpoint_matrix is not None else P_TOKENIZED
        sim.enable_adoption_markov_regimes(transition_matrix, endpoint, start_state="neutral")
    else:
        sim.enable_markov_regimes(transition_matrix, start_state="neutral")
    
    # Run simulation
    months = []
    micro_prices = []
    fund_prices = []
    regimes = []
    
    total_months = len(history) + months_ahead
    
    for m in range(total_months):
        # Run micro ticks
        sim.run_micro_ticks(ticks_per_candle)
        
        # Record data
        months.append(m)
        micro_prices.append(sim.order_book.last_price)
        fund_prices.append(sim.target_price)   # GBM fundamental — smooth monthly signal
        regimes.append(sim.regime)
        
        # Progress indicator
        if verbose and m % 100 == 0:
            print(f"  Month {m}/{total_months} | Regime: {sim.regime:8s} | Price: ${sim.order_book.last_price:,.0f}")
        
        # Roll to next month
        sim.roll_candle()
    
    if verbose:
        print(f"Simulation complete: {len(months)} months")
    
    return months, fund_prices, micro_prices, regimes, len(history)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(
    months: List[int],
    traditional_prices: List[float],
    tokenized_prices: List[float],
    traditional_regimes: List[str],
    tokenized_regimes: List[str],
    history_len: int,
    traditional_metrics_full: Dict,
    tokenized_metrics_full: Dict,
    traditional_metrics_forward: Dict,
    tokenized_metrics_forward: Dict,
):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Color map for regimes
    regime_colors = {
        "calm": "#2ecc71",      # green
        "neutral": "#3498db",   # blue
        "volatile": "#f39c12",  # orange
        "panic": "#e74c3c"      # red
    }
    
    # =========================
    # Plot 1: Price Comparison
    # =========================
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(months, traditional_prices, label="Traditional Housing", 
             linewidth=2, alpha=0.8, color="#2c3e50")
    ax1.plot(months, tokenized_prices, label="Tokenized Housing (REIT-like)", 
             linewidth=2, alpha=0.8, color="#16a085")
    projection_start = max(0, history_len - 1)
    ax1.axvline(
        projection_start,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label="Projection starts",
    )
    ymax = max(max(traditional_prices), max(tokenized_prices))
    ax1.text(
        projection_start + 3,
        ymax * 0.98,
        "History | Projection",
        color="#7f8c8d",
        fontsize=8,
        va="top",
    )
    ax1.set_title("Price Evolution: Traditional vs Tokenized Housing\n(GBM fundamental price, monthly)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # =========================
    # Plot 2: Projection-Only Price Evolution
    # =========================
    ax2 = plt.subplot(3, 2, 2)
    projection_months = months[history_len:]
    traditional_projection_prices = traditional_prices[history_len:]
    tokenized_projection_prices = tokenized_prices[history_len:]
    projection_x = np.arange(len(projection_months))
    ax2.plot(
        projection_x,
        traditional_projection_prices,
        label="Traditional Housing",
        linewidth=2,
        alpha=0.8,
        color="#2c3e50",
    )
    ax2.plot(
        projection_x,
        tokenized_projection_prices,
        label="Tokenized Housing (REIT-like)",
        linewidth=2,
        alpha=0.8,
        color="#16a085",
    )
    ax2.set_title("Price Evolution: Projection Window Only", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # =========================
    # Plot 3: Regime Distribution Comparison
    # =========================
    ax3 = plt.subplot(3, 2, 3)
    regimes_list = ["calm", "neutral", "volatile", "panic"]
    x = np.arange(len(regimes_list))
    width = 0.35
    
    trad_pcts = [traditional_metrics_full["regime_percentages"][r] for r in regimes_list]
    tok_pcts = [tokenized_metrics_full["regime_percentages"][r] for r in regimes_list]
    
    bars1 = ax3.bar(x - width/2, trad_pcts, width, label='Traditional', 
                    color='#2c3e50', alpha=0.8)
    bars2 = ax3.bar(x + width/2, tok_pcts, width, label='Tokenized', 
                    color='#16a085', alpha=0.8)
    
    ax3.set_title("Time Spent in Each Regime (%)", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Percentage (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([r.capitalize() for r in regimes_list])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # =========================
    # Plot 4: Drawdown Comparison
    # =========================
    ax4 = plt.subplot(3, 2, 4)
    
    # Calculate drawdowns
    trad_arr = np.array(traditional_prices)
    tok_arr = np.array(tokenized_prices)
    
    trad_cummax = np.maximum.accumulate(trad_arr)
    tok_cummax = np.maximum.accumulate(tok_arr)
    
    trad_dd = (trad_arr - trad_cummax) / trad_cummax * 100
    tok_dd = (tok_arr - tok_cummax) / tok_cummax * 100
    
    ax4.fill_between(months, trad_dd, 0, alpha=0.5, color='#e74c3c', label='Traditional')
    ax4.fill_between(months, tok_dd, 0, alpha=0.5, color='#2ecc71', label='Tokenized')
    ax4.set_title("Drawdown Comparison", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Drawdown (%)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # =========================
    # Plot 5: Full-Period Metrics Table
    # =========================
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('off')
    
    # Create full-period metrics table
    full_period_data = [
        ["Metric", "Traditional", "Tokenized", "Improvement"],
        ["Total Return (%)", 
         f"{traditional_metrics_full['total_return']:.2f}%",
         f"{tokenized_metrics_full['total_return']:.2f}%",
         f"{tokenized_metrics_full['total_return'] - traditional_metrics_full['total_return']:+.2f}%"],
        
        ["Volatility (annual %)", 
         f"{traditional_metrics_full['volatility']:.2f}%",
         f"{tokenized_metrics_full['volatility']:.2f}%",
         f"{tokenized_metrics_full['volatility'] - traditional_metrics_full['volatility']:+.2f}%"],
        
        ["Max Drawdown (%)", 
         f"{traditional_metrics_full['max_drawdown']:.2f}%",
         f"{tokenized_metrics_full['max_drawdown']:.2f}%",
         f"{tokenized_metrics_full['max_drawdown'] - traditional_metrics_full['max_drawdown']:+.2f}%"],
        
        ["Recovery Time (months)", 
         f"{traditional_metrics_full['recovery_time']:.0f}",
         f"{tokenized_metrics_full['recovery_time']:.0f}",
         f"{tokenized_metrics_full['recovery_time'] - traditional_metrics_full['recovery_time']:+.0f}"],
        
        ["Time in Stress (%)", 
         f"{traditional_metrics_full['stress_time_pct']:.1f}%",
         f"{tokenized_metrics_full['stress_time_pct']:.1f}%",
         f"{tokenized_metrics_full['stress_time_pct'] - traditional_metrics_full['stress_time_pct']:+.1f}%"],
        
        ["Panic Episodes", 
         f"{traditional_metrics_full['num_panic_episodes']:.0f}",
         f"{tokenized_metrics_full['num_panic_episodes']:.0f}",
         f"{tokenized_metrics_full['num_panic_episodes'] - traditional_metrics_full['num_panic_episodes']:+.0f}"],
        
        ["Avg Panic Duration (mo)", 
         f"{traditional_metrics_full['avg_panic_duration']:.1f}",
         f"{tokenized_metrics_full['avg_panic_duration']:.1f}",
         f"{tokenized_metrics_full['avg_panic_duration'] - traditional_metrics_full['avg_panic_duration']:+.1f}"],
    ]

    table_full = ax5.table(cellText=full_period_data, cellLoc='center', loc='center',
                           colWidths=[0.35, 0.2, 0.2, 0.25])
    table_full.auto_set_font_size(False)
    table_full.set_fontsize(8)
    table_full.scale(1, 1.4)

    for i in range(4):
        table_full[(0, i)].set_facecolor('#34495e')
        table_full[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(full_period_data)):
        for j in range(4):
            if j == 3:
                cell_text = full_period_data[i][j]
                if cell_text.startswith('+') or cell_text.startswith('-'):
                    if "Return" in full_period_data[i][0]:
                        color = '#d5f4e6' if '+' in cell_text else '#fadbd8'
                    else:
                        color = '#d5f4e6' if '-' in cell_text else '#fadbd8'
                    table_full[(i, j)].set_facecolor(color)

    ax5.set_title("Key Metrics Comparison — Full Period", fontsize=14, fontweight='bold', pad=20)

    # =========================
    # Plot 6: Projection-Only Metrics Table
    # =========================
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')

    projection_only_data = [
        ["Metric", "Traditional", "Tokenized", "Improvement"],
        ["Total Return (%)", 
         f"{traditional_metrics_forward['total_return']:.2f}%",
         f"{tokenized_metrics_forward['total_return']:.2f}%",
         f"{tokenized_metrics_forward['total_return'] - traditional_metrics_forward['total_return']:+.2f}%"],
        
        ["Volatility (annual %)", 
         f"{traditional_metrics_forward['volatility']:.2f}%",
         f"{tokenized_metrics_forward['volatility']:.2f}%",
         f"{tokenized_metrics_forward['volatility'] - traditional_metrics_forward['volatility']:+.2f}%"],
        
        ["Max Drawdown (%)", 
         f"{traditional_metrics_forward['max_drawdown']:.2f}%",
         f"{tokenized_metrics_forward['max_drawdown']:.2f}%",
         f"{tokenized_metrics_forward['max_drawdown'] - traditional_metrics_forward['max_drawdown']:+.2f}%"],
        
        ["Recovery Time (months)", 
         f"{traditional_metrics_forward['recovery_time']:.0f}",
         f"{tokenized_metrics_forward['recovery_time']:.0f}",
         f"{tokenized_metrics_forward['recovery_time'] - traditional_metrics_forward['recovery_time']:+.0f}"],
        
        ["Time in Stress (%)", 
         f"{traditional_metrics_forward['stress_time_pct']:.1f}%",
         f"{tokenized_metrics_forward['stress_time_pct']:.1f}%",
         f"{tokenized_metrics_forward['stress_time_pct'] - traditional_metrics_forward['stress_time_pct']:+.1f}%"],
        
        ["Panic Episodes", 
         f"{traditional_metrics_forward['num_panic_episodes']:.0f}",
         f"{tokenized_metrics_forward['num_panic_episodes']:.0f}",
         f"{tokenized_metrics_forward['num_panic_episodes'] - traditional_metrics_forward['num_panic_episodes']:+.0f}"],
        
        ["Avg Panic Duration (mo)", 
         f"{traditional_metrics_forward['avg_panic_duration']:.1f}",
         f"{tokenized_metrics_forward['avg_panic_duration']:.1f}",
         f"{tokenized_metrics_forward['avg_panic_duration'] - traditional_metrics_forward['avg_panic_duration']:+.1f}"],
    ]
    
    table_projection = ax6.table(cellText=projection_only_data, cellLoc='center', loc='center',
                                 colWidths=[0.35, 0.2, 0.2, 0.25])
    table_projection.auto_set_font_size(False)
    table_projection.set_fontsize(8)
    table_projection.scale(1, 1.4)
    
    # Style header row
    for i in range(4):
        table_projection[(0, i)].set_facecolor('#34495e')
        table_projection[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(projection_only_data)):
        for j in range(4):
            if j == 3:  # Improvement column
                # Color code improvements (green=good, red=bad)
                cell_text = projection_only_data[i][j]
                if cell_text.startswith('+') or cell_text.startswith('-'):
                    # For most metrics, lower is better
                    if "Return" in projection_only_data[i][0]:  # Return: higher is better
                        color = '#d5f4e6' if '+' in cell_text else '#fadbd8'
                    else:  # Other metrics: lower is better
                        color = '#d5f4e6' if '-' in cell_text else '#fadbd8'
                    table_projection[(i, j)].set_facecolor(color)
    
    ax6.set_title("Key Metrics Comparison — Projection Only", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save to outputs directory
    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir.parent / "outputs"
    output_path = outputs_dir / 'housing_liquidity_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Configuration
    # Get path to data directory relative to this script
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    CSV_PATH = str(data_dir / "cre_monthly.csv")
    MONTHS_AHEAD = 120  # Project 10 years into future
    TICKS_PER_CANDLE = 50
    SEED = 42
    USE_MICRO_FEEDBACK = True
    REGIME_MICRO_WEIGHT = 0.25
    FUNDAMENTAL_MICRO_FEEDBACK = 0.10
    
    print("\n" + "="*70)
    print(" HOUSING MARKET LIQUIDITY COMPARISON")
    print(" Traditional (Illiquid) vs Tokenized (REIT-like)")
    print(" Tokenized run uses adoption-weighted Markov interpolation to Bayesian endpoint")
    print("="*70)
    
    # Run Traditional simulation
    months, traditional_prices, traditional_micro_prices, traditional_regimes, history_len = run_simulation(
        csv_path=CSV_PATH,
        months_ahead=MONTHS_AHEAD,
        ticks_per_candle=TICKS_PER_CANDLE,
        transition_matrix=P_TRADITIONAL,
        scenario_name="Traditional Housing (Illiquid)",
        seed=SEED,
        use_micro_feedback=USE_MICRO_FEEDBACK,
        regime_micro_weight=REGIME_MICRO_WEIGHT,
        fundamental_micro_feedback=FUNDAMENTAL_MICRO_FEEDBACK,
    )
    
    # Run Tokenized simulation
    _, tokenized_prices, tokenized_micro_prices, tokenized_regimes, _ = run_simulation(
        csv_path=CSV_PATH,
        months_ahead=MONTHS_AHEAD,
        ticks_per_candle=TICKS_PER_CANDLE,
        transition_matrix=P_TRADITIONAL,
        scenario_name="Tokenized Housing (REIT-like)",
        adoption_interpolated_markov=True,
        tokenized_endpoint_matrix=P_TOKENIZED,
        seed=SEED,
        use_micro_feedback=USE_MICRO_FEEDBACK,
        regime_micro_weight=REGIME_MICRO_WEIGHT,
        fundamental_micro_feedback=FUNDAMENTAL_MICRO_FEEDBACK,
    )
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    traditional_metrics_full = calculate_metrics(traditional_prices, traditional_regimes)
    tokenized_metrics_full = calculate_metrics(tokenized_prices, tokenized_regimes)

    # Forward-only metrics isolate treatment effects (history segment is identical).
    traditional_prices_fwd = traditional_prices[history_len:]
    tokenized_prices_fwd = tokenized_prices[history_len:]
    traditional_regimes_fwd = traditional_regimes[history_len:]
    tokenized_regimes_fwd = tokenized_regimes[history_len:]

    traditional_metrics_forward = calculate_metrics(traditional_prices_fwd, traditional_regimes_fwd)
    tokenized_metrics_forward = calculate_metrics(tokenized_prices_fwd, tokenized_regimes_fwd)
    traditional_micro_metrics_forward = calculate_metrics(traditional_micro_prices[history_len:], traditional_regimes_fwd)
    tokenized_micro_metrics_forward = calculate_metrics(tokenized_micro_prices[history_len:], tokenized_regimes_fwd)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nHistory replay months: {history_len} | Forward projection months: {MONTHS_AHEAD}")

    print("\nFULL PERIOD (History + Projection):")
    print(f"  Traditional Return: {traditional_metrics_full['total_return']:.2f}% | Tokenized Return: {tokenized_metrics_full['total_return']:.2f}%")
    print(f"  Traditional Vol: {traditional_metrics_full['volatility']:.2f}% | Tokenized Vol: {tokenized_metrics_full['volatility']:.2f}%")
    print(f"  Traditional Stress: {traditional_metrics_full['stress_time_pct']:.1f}% | Tokenized Stress: {tokenized_metrics_full['stress_time_pct']:.1f}%")

    print("\nPROJECTION ONLY (Forward Window):")
    print(f"  Traditional Return: {traditional_metrics_forward['total_return']:.2f}% | Tokenized Return: {tokenized_metrics_forward['total_return']:.2f}%")
    print(f"  Traditional Vol: {traditional_metrics_forward['volatility']:.2f}% | Tokenized Vol: {tokenized_metrics_forward['volatility']:.2f}%")
    print(f"  Traditional Stress: {traditional_metrics_forward['stress_time_pct']:.1f}% | Tokenized Stress: {tokenized_metrics_forward['stress_time_pct']:.1f}%")
    print("  [Micro Price Metrics — Forward Window]")
    print(f"  Traditional Return: {traditional_micro_metrics_forward['total_return']:.2f}% | Tokenized Return: {tokenized_micro_metrics_forward['total_return']:.2f}%")
    print(f"  Traditional Vol: {traditional_micro_metrics_forward['volatility']:.2f}% | Tokenized Vol: {tokenized_micro_metrics_forward['volatility']:.2f}%")
    
    print("\nKEY INSIGHTS (Projection Only):")
    recovery_improvement = traditional_metrics_forward['recovery_time'] - tokenized_metrics_forward['recovery_time']
    panic_improvement = traditional_metrics_forward['avg_panic_duration'] - tokenized_metrics_forward['avg_panic_duration']
    stress_improvement = traditional_metrics_forward['stress_time_pct'] - tokenized_metrics_forward['stress_time_pct']

    recovery_word = "faster" if recovery_improvement >= 0 else "slower"
    panic_word = "shorter" if panic_improvement >= 0 else "longer"
    stress_word = "less" if stress_improvement >= 0 else "more"
    print(f"  • Recovery Time: {abs(recovery_improvement):.0f} months {recovery_word} in tokenized market")
    print(f"  • Panic Duration: {abs(panic_improvement):.1f} months {panic_word} in tokenized market")
    print(f"  • Time in Stress: {abs(stress_improvement):.1f}% {stress_word} in tokenized market")
    print(f"  • Liquidity benefit: Faster price discovery and crisis resolution")
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create plots
    plot_comparison(
        months=months,
        traditional_prices=traditional_prices,
        tokenized_prices=tokenized_prices,
        traditional_regimes=traditional_regimes,
        tokenized_regimes=tokenized_regimes,
        history_len=history_len,
        traditional_metrics_full=traditional_metrics_full,
        tokenized_metrics_full=tokenized_metrics_full,
        traditional_metrics_forward=traditional_metrics_forward,
        tokenized_metrics_forward=tokenized_metrics_forward,
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nSingle-path comparison complete.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()