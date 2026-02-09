"""
Market Simulator Main Entry Point

Demonstrates tokenized vs traditional CRE market simulation with regime dynamics.
"""

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
# Transition Matrices for Regime Dynamics
# =============================================================================

# Traditional housing: slow recovery from stress
P_TRADITIONAL = [
    [0.85, 0.14, 0.01, 0.00],  # calm
    [0.10, 0.75, 0.14, 0.01],  # neutral
    [0.02, 0.18, 0.70, 0.10],  # volatile
    [0.01, 0.09, 0.30, 0.60],  # panic (gets stuck)
]

# Tokenized housing: faster crisis resolution
P_TOKENIZED = [
    [0.90, 0.10, 0.00, 0.00],  # calm
    [0.08, 0.80, 0.10, 0.02],  # neutral
    [0.03, 0.25, 0.65, 0.07],  # volatile
    [0.02, 0.30, 0.40, 0.28],  # panic resolves faster
]


def main():
    """Main execution function."""
    print("Running basic simulation...")
    run_simulation_runtime(5.0, 10)
    
    print("\nRunning CRE tokenization demo...")
    demo_cre_with_tokenization("cre_monthly.csv")


if __name__ == "__main__":
    main()