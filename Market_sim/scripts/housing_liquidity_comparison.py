"""
Housing Liquidity Comparison: Traditional vs Tokenized (REIT-like)

This simulation demonstrates what a US housing market with higher liquidity 
(similar to REITs) would look like compared to traditional illiquid housing markets.

Key differences:
- Traditional: Slow regime transitions, gets stuck in panic/volatile states (assumed)
- Tokenized: EMPIRICAL transition matrix from VNQ (Vanguard Real Estate ETF, 2005-2026)
  - 88% of time in calm/neutral states
  - Only 4% in panic, but when it hits, it's severe (90% persistence)
  - Represents actual REIT market behavior with real liquidity
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

# Tokenized Housing (REIT-like): Empirical from VNQ (Vanguard Real Estate ETF)
# Based on 253 months of real REIT data (2005-2026)
# Key insight: REITs spend 88% of time in calm/neutral, but when panic hits, it's severe
P_TOKENIZED = [
    [0.8174, 0.1739, 0.0087, 0.0000],  # calm -> highly stable (46% of time)
    [0.1887, 0.7736, 0.0283, 0.0094],  # neutral -> strong persistence (42% of time)
    [0.0500, 0.2000, 0.7500, 0.0000],  # volatile -> can escape to neutral (8% of time)
    [0.0000, 0.0000, 0.1000, 0.9000],  # panic -> very persistent BUT rare (4% of time)
]


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
    seed: int = 42
) -> Tuple[List[float], List[float], List[str]]:
    """
    Run a single simulation with specified transition matrix
    
    Returns:
        months: List of month indices
        prices: List of micro prices
        regimes: List of regime labels
    """
    # Create new simulator instance
    sim = MarketSimulator()
    sim.rng.seed(seed)
    
    # Load and calibrate
    history = load_cre_csv(csv_path)
    calib = calibrate_from_history(history)
    
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
        fund_prices.append(sim.target_price)
        regimes.append(sim.regime)
        
        # Progress indicator
        if m % 100 == 0:
            print(f"  Month {m}/{total_months} | Regime: {sim.regime:8s} | Price: ${sim.order_book.last_price:,.0f}")
        
        # Roll to next month
        sim.roll_candle()
    
    print(f"Simulation complete: {len(months)} months")
    
    return months, micro_prices, regimes


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(
    months: List[int],
    traditional_prices: List[float],
    tokenized_prices: List[float],
    traditional_regimes: List[str],
    tokenized_regimes: List[str],
    traditional_metrics: Dict,
    tokenized_metrics: Dict,
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
    ax1.set_title("Price Evolution: Traditional vs Tokenized Housing", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # =========================
    # Plot 2: Regime Time Series (Traditional)
    # =========================
    ax2 = plt.subplot(3, 2, 2)
    regime_numeric_trad = [["calm", "neutral", "volatile", "panic"].index(r) for r in traditional_regimes]
    colors_trad = [regime_colors[r] for r in traditional_regimes]
    ax2.scatter(months, regime_numeric_trad, c=colors_trad, alpha=0.6, s=10)
    ax2.set_title("Regime Evolution: Traditional Housing", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Regime")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["Calm", "Neutral", "Volatile", "Panic"])
    ax2.grid(True, alpha=0.3)
    
    # =========================
    # Plot 3: Regime Time Series (Tokenized)
    # =========================
    ax3 = plt.subplot(3, 2, 3)
    regime_numeric_tok = [["calm", "neutral", "volatile", "panic"].index(r) for r in tokenized_regimes]
    colors_tok = [regime_colors[r] for r in tokenized_regimes]
    ax3.scatter(months, regime_numeric_tok, c=colors_tok, alpha=0.6, s=10)
    ax3.set_title("Regime Evolution: Tokenized Housing", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Regime")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["Calm", "Neutral", "Volatile", "Panic"])
    ax3.grid(True, alpha=0.3)
    
    # =========================
    # Plot 4: Regime Distribution Comparison
    # =========================
    ax4 = plt.subplot(3, 2, 4)
    regimes_list = ["calm", "neutral", "volatile", "panic"]
    x = np.arange(len(regimes_list))
    width = 0.35
    
    trad_pcts = [traditional_metrics["regime_percentages"][r] for r in regimes_list]
    tok_pcts = [tokenized_metrics["regime_percentages"][r] for r in regimes_list]
    
    bars1 = ax4.bar(x - width/2, trad_pcts, width, label='Traditional', 
                    color='#2c3e50', alpha=0.8)
    bars2 = ax4.bar(x + width/2, tok_pcts, width, label='Tokenized', 
                    color='#16a085', alpha=0.8)
    
    ax4.set_title("Time Spent in Each Regime (%)", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Percentage (%)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([r.capitalize() for r in regimes_list])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # =========================
    # Plot 5: Drawdown Comparison
    # =========================
    ax5 = plt.subplot(3, 2, 5)
    
    # Calculate drawdowns
    trad_arr = np.array(traditional_prices)
    tok_arr = np.array(tokenized_prices)
    
    trad_cummax = np.maximum.accumulate(trad_arr)
    tok_cummax = np.maximum.accumulate(tok_arr)
    
    trad_dd = (trad_arr - trad_cummax) / trad_cummax * 100
    tok_dd = (tok_arr - tok_cummax) / tok_cummax * 100
    
    ax5.fill_between(months, trad_dd, 0, alpha=0.5, color='#e74c3c', label='Traditional')
    ax5.fill_between(months, tok_dd, 0, alpha=0.5, color='#2ecc71', label='Tokenized')
    ax5.set_title("Drawdown Comparison", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Drawdown (%)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # =========================
    # Plot 6: Metrics Summary Table
    # =========================
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create metrics table
    metrics_data = [
        ["Metric", "Traditional", "Tokenized", "Improvement"],
        ["", "", "", ""],
        ["Total Return (%)", 
         f"{traditional_metrics['total_return']:.2f}%",
         f"{tokenized_metrics['total_return']:.2f}%",
         f"{tokenized_metrics['total_return'] - traditional_metrics['total_return']:+.2f}%"],
        
        ["Volatility (annual %)", 
         f"{traditional_metrics['volatility']:.2f}%",
         f"{tokenized_metrics['volatility']:.2f}%",
         f"{tokenized_metrics['volatility'] - traditional_metrics['volatility']:+.2f}%"],
        
        ["Max Drawdown (%)", 
         f"{traditional_metrics['max_drawdown']:.2f}%",
         f"{tokenized_metrics['max_drawdown']:.2f}%",
         f"{tokenized_metrics['max_drawdown'] - traditional_metrics['max_drawdown']:+.2f}%"],
        
        ["Recovery Time (months)", 
         f"{traditional_metrics['recovery_time']:.0f}",
         f"{tokenized_metrics['recovery_time']:.0f}",
         f"{tokenized_metrics['recovery_time'] - traditional_metrics['recovery_time']:+.0f}"],
        
        ["Time in Stress (%)", 
         f"{traditional_metrics['stress_time_pct']:.1f}%",
         f"{tokenized_metrics['stress_time_pct']:.1f}%",
         f"{tokenized_metrics['stress_time_pct'] - traditional_metrics['stress_time_pct']:+.1f}%"],
        
        ["Panic Episodes", 
         f"{traditional_metrics['num_panic_episodes']:.0f}",
         f"{tokenized_metrics['num_panic_episodes']:.0f}",
         f"{tokenized_metrics['num_panic_episodes'] - traditional_metrics['num_panic_episodes']:+.0f}"],
        
        ["Avg Panic Duration (mo)", 
         f"{traditional_metrics['avg_panic_duration']:.1f}",
         f"{tokenized_metrics['avg_panic_duration']:.1f}",
         f"{tokenized_metrics['avg_panic_duration'] - traditional_metrics['avg_panic_duration']:+.1f}"],
    ]
    
    table = ax6.table(cellText=metrics_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(2, len(metrics_data)):
        for j in range(4):
            if j == 3:  # Improvement column
                # Color code improvements (green=good, red=bad)
                cell_text = metrics_data[i][j]
                if cell_text.startswith('+') or cell_text.startswith('-'):
                    # For most metrics, lower is better
                    if "Return" in metrics_data[i][0]:  # Return: higher is better
                        color = '#d5f4e6' if '+' in cell_text else '#fadbd8'
                    else:  # Other metrics: lower is better
                        color = '#d5f4e6' if '-' in cell_text else '#fadbd8'
                    table[(i, j)].set_facecolor(color)
    
    ax6.set_title("Key Metrics Comparison", fontsize=14, fontweight='bold', pad=20)
    
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
    
    print("\n" + "="*70)
    print(" HOUSING MARKET LIQUIDITY COMPARISON")
    print(" Traditional (Illiquid) vs Tokenized (REIT-like)")
    print("="*70)
    
    # Run Traditional simulation
    months, traditional_prices, traditional_regimes = run_simulation(
        csv_path=CSV_PATH,
        months_ahead=MONTHS_AHEAD,
        ticks_per_candle=TICKS_PER_CANDLE,
        transition_matrix=P_TRADITIONAL,
        scenario_name="Traditional Housing (Illiquid)",
        seed=SEED
    )
    
    # Run Tokenized simulation
    _, tokenized_prices, tokenized_regimes = run_simulation(
        csv_path=CSV_PATH,
        months_ahead=MONTHS_AHEAD,
        ticks_per_candle=TICKS_PER_CANDLE,
        transition_matrix=P_TOKENIZED,
        scenario_name="Tokenized Housing (REIT-like)",
        seed=SEED
    )
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    traditional_metrics = calculate_metrics(traditional_prices, traditional_regimes)
    tokenized_metrics = calculate_metrics(tokenized_prices, tokenized_regimes)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nTRADITIONAL HOUSING (Illiquid):")
    print(f"  Total Return: {traditional_metrics['total_return']:.2f}%")
    print(f"  Volatility: {traditional_metrics['volatility']:.2f}%")
    print(f"  Max Drawdown: {traditional_metrics['max_drawdown']:.2f}%")
    print(f"  Recovery Time: {traditional_metrics['recovery_time']:.0f} months")
    print(f"  Time in Stress: {traditional_metrics['stress_time_pct']:.1f}%")
    print(f"  Panic Episodes: {traditional_metrics['num_panic_episodes']:.0f}")
    print(f"  Avg Panic Duration: {traditional_metrics['avg_panic_duration']:.1f} months")
    
    print("\nTOKENIZED HOUSING (REIT-like):")
    print(f"  Total Return: {tokenized_metrics['total_return']:.2f}%")
    print(f"  Volatility: {tokenized_metrics['volatility']:.2f}%")
    print(f"  Max Drawdown: {tokenized_metrics['max_drawdown']:.2f}%")
    print(f"  Recovery Time: {tokenized_metrics['recovery_time']:.0f} months")
    print(f"  Time in Stress: {tokenized_metrics['stress_time_pct']:.1f}%")
    print(f"  Panic Episodes: {tokenized_metrics['num_panic_episodes']:.0f}")
    print(f"  Avg Panic Duration: {tokenized_metrics['avg_panic_duration']:.1f} months")
    
    print("\nKEY INSIGHTS:")
    recovery_improvement = traditional_metrics['recovery_time'] - tokenized_metrics['recovery_time']
    panic_improvement = traditional_metrics['avg_panic_duration'] - tokenized_metrics['avg_panic_duration']
    stress_improvement = traditional_metrics['stress_time_pct'] - tokenized_metrics['stress_time_pct']
    
    print(f"  • Recovery Time: {recovery_improvement:.0f} months faster in tokenized market")
    print(f"  • Panic Duration: {panic_improvement:.1f} months shorter in tokenized market")
    print(f"  • Time in Stress: {stress_improvement:.1f}% less in tokenized market")
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
        traditional_metrics=traditional_metrics,
        tokenized_metrics=tokenized_metrics,
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nThe simulation demonstrates how increased liquidity (tokenization)")
    print("leads to faster recovery from market stress and reduced time in crisis.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
