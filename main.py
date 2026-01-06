from sim.market_simulator import MarketSimulator
from sim.data_loader import load_cre_csv
from sim.calibration import calibrate_from_history
from sim.portfolio import merton_optimal_weight
from sim.types import ScenarioParams
import matplotlib.pyplot as plt

sim = MarketSimulator()
def run_simulation_runtime(seconds: float, ticks_per_second: int) -> None:
    # simple runtime demo (no history/scenario attached
    steps = int(seconds * ticks_per_second)
    sim.run_micro_ticks(steps)
    print("Done. Last micro price:", sim.order_book.last_price)

def demo_cre_with_tokenization(path_to_csv: str, months_ahead: int = 24, ticks_per_candle: int = 50) -> None:
    history = load_cre_csv(path_to_csv)
    calib = calibrate_from_history(history)

    # Example parameters
    r_annual = 0.02
    gamma = 3.0
    w_star = merton_optimal_weight(calib.mu_annual, r_annual, gamma, calib.sigma_annual)
    print("Merton w*:", w_star)

    scenario = ScenarioParams(
        name="tokenized_bullish",
        mu_monthly=calib.mu_monthly * 1.10,
        sigma_monthly=calib.sigma_monthly * 0.90,
    )

    sim.attach_history_and_scenario(history, scenario)

    # --- NEW: store time series for plotting ---
    months = []
    micro_series = []
    fund_series = []

    total_months = len(history) + months_ahead
    for m in range(total_months):
        sim.run_micro_ticks(ticks_per_candle)

        micro = sim.order_book.last_price
        fund = sim.target_price

        # --- NEW: collect for plotting ---
        months.append(m)
        micro_series.append(micro)
        fund_series.append(fund)

        print(
            f"Month={m:04d} | micro={micro} | fund={fund} | "
            f"regime={sim.regime} sigma={sim.current_sigma_monthly:.4f} | "
            f"Candle O={sim.live_candlestick.open} H={sim.live_candlestick.high} "
            f"L={sim.live_candlestick.low} C={sim.live_candlestick.close} "
            f"Vol={sim.live_candlestick.volume}"
        )

        sim.roll_candle()
    # for m in range(total_months):
    #     fund_price = sim.target_price  # snapshot "this month's" fundamental before rolling

    #     sim.run_micro_ticks(ticks_per_candle)

    #     # capture the candle for this month, then advance the month state
    #     candle = sim.roll_candle()

    #     micro_price = candle.close

    #     print(
    #         f"Month={m:04d} | micro={micro_price} | fund={fund_price:.0f} | "
    #         f"regime={sim.current_regime} sigma={sim.current_sigma_monthly:.4f} | "
    #         f"Candle O={candle.open} H={candle.high} L={candle.low} C={candle.close} Vol={candle.volume}"
    #     )

    # --- NEW: plot at end ---
    plt.figure(figsize=(11, 5))
    plt.plot(months, fund_series, label="Traditional CRE (Fund/target_price)")
    plt.plot(months, micro_series, label="Tokenized CRE (Micro/last_price)")
    plt.title("Traditional vs Tokenized CRE")
    plt.xlabel("Month")
    plt.ylabel("Price(USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




P_traditional = [
    [0.85, 0.14, 0.01, 0.00],  # calm
    [0.10, 0.75, 0.14, 0.01],  # neutral
    [0.02, 0.18, 0.70, 0.10],  # volatile
    [0.01, 0.09, 0.30, 0.60],  # panic
]

P_tokenized = [
    [0.90, 0.10, 0.00, 0.00],
    [0.08, 0.80, 0.10, 0.02],
    [0.03, 0.25, 0.65, 0.07],
    [0.02, 0.30, 0.40, 0.28],  # panic resolves faster
]

sim.enable_markov_regimes(P_tokenized, start_state="neutral")


if __name__ == "__main__":
    run_simulation_runtime(5.0, 10)
    demo_cre_with_tokenization("cre_monthly.csv")