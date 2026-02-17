"""
Market Simulator Main Entry Point

Demonstrates tokenized vs traditional CRE market simulation with regime dynamics.

This demo uses EMPIRICAL transition matrices derived from:
- Traditional CRE: 72+ years of historical data (1953-2025)
- Tokenized/REIT: 20+ years of VNQ ETF data (2005-2026)

Run 'python3 scripts/run_complete_analysis.py' to regenerate these matrices.
"""

import os
import matplotlib.pyplot as plt

from sim.market_simulator import MarketSimulator
from sim.data_loader import load_cre_csv
from sim.calibration import calibrate_from_history
from sim.portfolio import merton_optimal_weight
from sim.types import ScenarioParams


def run_simulation_runtime(seconds: float, ticks_per_second: int) -> None:
    """
    Run a simple runtime simulation with no historical data attached.
    
    Args:
        seconds: Duration of simulation in seconds
        ticks_per_second: Number of ticks to execute per second
    """
    sim = MarketSimulator()
    steps = int(seconds * ticks_per_second)
    sim.run_micro_ticks(steps)
    print(f"Done. Last micro price: {sim.order_book.last_price}")


def demo_cre_with_tokenization(
    path_to_csv: str, 
    months_ahead: int = 24, 
    ticks_per_candle: int = 50
) -> None:
    """
    Simulate tokenized CRE market and compare with traditional.
    
    Args:
        path_to_csv: Path to historical CRE data CSV file
        months_ahead: Number of months to project into future
        ticks_per_candle: Number of micro ticks per monthly candle
    """
    sim = MarketSimulator()
    history = load_cre_csv(path_to_csv)
    calib = calibrate_from_history(history)

    # Calculate Merton optimal portfolio weight
    r_annual = 0.02
    gamma = 3.0
    w_star = merton_optimal_weight(calib.mu_annual, r_annual, gamma, calib.sigma_annual)
    print(f"Merton optimal weight: {w_star:.4f}")

    # Create bullish tokenization scenario
    scenario = ScenarioParams(
        name="tokenized_bullish",
        mu_monthly=calib.mu_monthly * 1.10,
        sigma_monthly=calib.sigma_monthly * 0.90,
    )

    sim.attach_history_and_scenario(history, scenario)

    # Store time series for plotting
    months = []
    micro_series = []
    fund_series = []

    # Run simulation
    total_months = len(history) + months_ahead
    for m in range(total_months):
        sim.run_micro_ticks(ticks_per_candle)

        micro = sim.order_book.last_price
        fund = sim.target_price

        months.append(m)
        micro_series.append(micro)
        fund_series.append(fund)

        print(
            f"Month={m:04d} | micro={micro:,.0f} | fund={fund:,.0f} | "
            f"regime={sim.regime:8s} | sigma={sim.current_sigma_monthly:.4f} | "
            f"Candle OHLC=({sim.live_candlestick.open}, {sim.live_candlestick.high}, "
            f"{sim.live_candlestick.low}, {sim.live_candlestick.close}) "
            f"Vol={sim.live_candlestick.volume}"
        )

        sim.roll_candle()

    # Plot results
    plt.figure(figsize=(11, 5))
    plt.plot(months, fund_series, label="Traditional CRE (Fund Price)", linewidth=2)
    plt.plot(months, micro_series, label="Tokenized CRE (Micro Price)", linewidth=2, alpha=0.8)
    plt.title("Traditional vs Tokenized CRE Market", fontsize=14, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Transition Matrices for Regime Dynamics (EMPIRICAL)
# =============================================================================

# Traditional CRE: Empirical from 72+ years of data (1953-2025)
# Source: analyze_cre_regimes.py analysis of cre_monthly.csv
# Characteristics: High persistence (86% calm), very low volatility (1.8% annual)
# Only 0.5% of time in panic over 72 years
P_TRADITIONAL = [
    [0.8591, 0.1389, 0.0020, 0.0000],  # calm    (58.7% of time, avg 7.1 months)
    [0.2339, 0.7186, 0.0475, 0.0000],  # neutral (34.0% of time, avg 3.5 months)
    [0.0339, 0.2203, 0.6949, 0.0508],  # volatile (6.8% of time, avg 3.3 months)
    [0.0000, 0.0000, 0.7500, 0.2500],  # panic    (0.5% of time, avg 1.3 months)
]

# Tokenized/REIT: Empirical from VNQ data (2005-2026)
# Source: analyze_reit_regimes.py analysis of VNQ ETF
# Characteristics: Lower persistence (82% calm), high volatility (22.2% annual)
# 9.8x more volatile than traditional CRE
P_TOKENIZED = [
    [0.8174, 0.1739, 0.0087, 0.0000],  # calm    (46.0% of time, avg 5.3 months)
    [0.1887, 0.7736, 0.0283, 0.0094],  # neutral (42.1% of time, avg 4.4 months)
    [0.0500, 0.2000, 0.7500, 0.0000],  # volatile (7.9% of time, avg 4.0 months)
    [0.0000, 0.0000, 0.1000, 0.9000],  # panic    (4.0% of time, avg 10.0 months - sticky!)
]


def main():
    """Main execution function."""
    print("Running basic simulation...")
    run_simulation_runtime(5.0, 10)
    
    print("\nRunning CRE tokenization demo...")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "cre_monthly.csv")
    demo_cre_with_tokenization(csv_path)


if __name__ == "__main__":
    main()