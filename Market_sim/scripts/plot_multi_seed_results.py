"""
Plot PNG charts from multi-seed liquidity sweep results.

This script reads outputs/multi_seed_liquidity_sweep_results.csv and writes:
  1) multi_seed_price_evolution_proxy.png
  2) multi_seed_key_metrics.png

Note:
The CSV contains projection-window summary metrics per seed, not full monthly
price paths. Therefore, "price evolution" is represented as forward endpoint
return/price-index distributions across trader counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "seed",
        "trader_count",
        "traditional_return_fwd",
        "tokenized_return_fwd",
        "spread_return_fwd",
        "traditional_volatility_fwd",
        "tokenized_volatility_fwd",
        "spread_volatility_fwd",
        "traditional_micro_return_fwd",
        "tokenized_micro_return_fwd",
        "spread_micro_return_fwd",
        "traditional_stress_fwd",
        "tokenized_stress_fwd",
        "spread_stress_fwd",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "traditional_sharpe_fwd" not in df.columns:
        trad_vol = df["traditional_volatility_fwd"].replace(0.0, np.nan)
        df["traditional_sharpe_fwd"] = df["traditional_return_fwd"] / trad_vol
        df["traditional_sharpe_fwd"] = df["traditional_sharpe_fwd"].fillna(0.0)
    if "tokenized_sharpe_fwd" not in df.columns:
        tok_vol = df["tokenized_volatility_fwd"].replace(0.0, np.nan)
        df["tokenized_sharpe_fwd"] = df["tokenized_return_fwd"] / tok_vol
        df["tokenized_sharpe_fwd"] = df["tokenized_sharpe_fwd"].fillna(0.0)
    if "spread_sharpe_fwd" not in df.columns:
        df["spread_sharpe_fwd"] = df["tokenized_sharpe_fwd"] - df["traditional_sharpe_fwd"]
    return df


def _plot_price_proxy(df: pd.DataFrame, output_path: Path) -> None:
    trader_counts = sorted(df["trader_count"].unique().tolist())

    # Price-index proxy from forward return.
    # Start from index=100 and map forward return to endpoint.
    df = df.copy()
    df["traditional_price_index"] = 100.0 * (1.0 + df["traditional_return_fwd"] / 100.0)
    df["tokenized_price_index"] = 100.0 * (1.0 + df["tokenized_return_fwd"] / 100.0)
    grouped = df.groupby("trader_count", sort=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    tok_means = grouped["tokenized_return_fwd"].mean().reindex(trader_counts).values
    trad_means = grouped["traditional_return_fwd"].mean().reindex(trader_counts).values
    tok_lo = grouped["tokenized_return_fwd"].quantile(0.025).reindex(trader_counts).values
    tok_hi = grouped["tokenized_return_fwd"].quantile(0.975).reindex(trader_counts).values
    trad_lo = grouped["traditional_return_fwd"].quantile(0.025).reindex(trader_counts).values
    trad_hi = grouped["traditional_return_fwd"].quantile(0.975).reindex(trader_counts).values

    ax1.plot(trader_counts, trad_means, marker="o", linewidth=2, label="Traditional mean return")
    ax1.fill_between(trader_counts, trad_lo, trad_hi, alpha=0.2, label="Traditional 95% band")
    ax1.plot(trader_counts, tok_means, marker="o", linewidth=2, label="Tokenized mean return")
    ax1.fill_between(trader_counts, tok_lo, tok_hi, alpha=0.2, label="Tokenized 95% band")
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Forward Return vs Trader Count (Multi-Seed)")
    ax1.set_xlabel("Tokenized trader bots")
    ax1.set_ylabel("Forward return (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    tok_idx_mean = grouped["tokenized_price_index"].mean().reindex(trader_counts).values
    trad_idx_mean = grouped["traditional_price_index"].mean().reindex(trader_counts).values
    tok_idx_lo = grouped["tokenized_price_index"].quantile(0.025).reindex(trader_counts).values
    tok_idx_hi = grouped["tokenized_price_index"].quantile(0.975).reindex(trader_counts).values
    trad_idx_lo = grouped["traditional_price_index"].quantile(0.025).reindex(trader_counts).values
    trad_idx_hi = grouped["traditional_price_index"].quantile(0.975).reindex(trader_counts).values

    ax2.plot(trader_counts, trad_idx_mean, marker="o", linewidth=2, label="Traditional mean endpoint index")
    ax2.fill_between(trader_counts, trad_idx_lo, trad_idx_hi, alpha=0.2, label="Traditional 95% band")
    ax2.plot(trader_counts, tok_idx_mean, marker="o", linewidth=2, label="Tokenized mean endpoint index")
    ax2.fill_between(trader_counts, tok_idx_lo, tok_idx_hi, alpha=0.2, label="Tokenized 95% band")
    ax2.axhline(100.0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Endpoint Price Index Proxy (Start=100)")
    ax2.set_xlabel("Tokenized trader bots")
    ax2.set_ylabel("Endpoint index")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_key_metrics(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby("trader_count", sort=True)
    trader_counts = np.array(sorted(df["trader_count"].unique().tolist()), dtype=int)

    spread_mean = grouped["spread_return_fwd"].mean().reindex(trader_counts).values
    spread_lo = grouped["spread_return_fwd"].quantile(0.025).reindex(trader_counts).values
    spread_hi = grouped["spread_return_fwd"].quantile(0.975).reindex(trader_counts).values
    p_outperform = grouped["spread_return_fwd"].apply(lambda s: 100.0 * float((s > 0).mean())).reindex(trader_counts).values

    micro_spread_mean = grouped["spread_micro_return_fwd"].mean().reindex(trader_counts).values
    stress_spread_mean = grouped["spread_stress_fwd"].mean().reindex(trader_counts).values
    sharpe_spread_mean = grouped["spread_sharpe_fwd"].mean().reindex(trader_counts).values
    sharpe_spread_lo = grouped["spread_sharpe_fwd"].quantile(0.025).reindex(trader_counts).values
    sharpe_spread_hi = grouped["spread_sharpe_fwd"].quantile(0.975).reindex(trader_counts).values
    p_sharpe_outperform = grouped["spread_sharpe_fwd"].apply(lambda s: 100.0 * float((s > 0).mean())).reindex(trader_counts).values

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    ax1.plot(trader_counts, spread_mean, marker="o", linewidth=2, color="#16a085")
    ax1.fill_between(trader_counts, spread_lo, spread_hi, alpha=0.25, color="#16a085")
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Return Spread: Tokenized - Traditional")
    ax1.set_xlabel("Tokenized trader bots")
    ax1.set_ylabel("Spread (%)")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(trader_counts, p_outperform, marker="o", linewidth=2, color="#2980b9")
    ax2.axhline(50.0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.set_title("Probability Tokenized Outperforms")
    ax2.set_xlabel("Tokenized trader bots")
    ax2.set_ylabel("P(spread > 0) %")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(trader_counts, micro_spread_mean, marker="o", linewidth=2, color="#8e44ad")
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax3.set_title("Micro Return Spread Mean")
    ax3.set_xlabel("Tokenized trader bots")
    ax3.set_ylabel("Micro spread (%)")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(trader_counts, stress_spread_mean, marker="o", linewidth=2, color="#c0392b")
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Stress-Time Spread Mean")
    ax4.set_xlabel("Tokenized trader bots")
    ax4.set_ylabel("Stress spread (pp)")
    ax4.grid(True, alpha=0.3)

    ax5 = axes[2, 0]
    ax5.plot(trader_counts, sharpe_spread_mean, marker="o", linewidth=2, color="#d35400")
    ax5.fill_between(trader_counts, sharpe_spread_lo, sharpe_spread_hi, alpha=0.25, color="#d35400")
    ax5.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax5.set_title("Sharpe Spread: Tokenized - Traditional")
    ax5.set_xlabel("Tokenized trader bots")
    ax5.set_ylabel("Sharpe spread")
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    ax6.plot(trader_counts, p_sharpe_outperform, marker="o", linewidth=2, color="#34495e")
    ax6.axhline(50.0, color="black", linestyle="--", linewidth=1)
    ax6.set_ylim(0, 100)
    ax6.set_title("Probability Tokenized Sharpe Outperforms")
    ax6.set_xlabel("Tokenized trader bots")
    ax6.set_ylabel("P(Sharpe spread > 0) %")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_annual_volatility(df: pd.DataFrame, output_path: Path) -> None:
    trader_counts = sorted(df["trader_count"].unique().tolist())
    grouped = df.groupby("trader_count", sort=True)

    trad_mean = grouped["traditional_volatility_fwd"].mean().reindex(trader_counts).values
    tok_mean = grouped["tokenized_volatility_fwd"].mean().reindex(trader_counts).values
    spread_mean = grouped["spread_volatility_fwd"].mean().reindex(trader_counts).values

    trad_lo = grouped["traditional_volatility_fwd"].quantile(0.025).reindex(trader_counts).values
    trad_hi = grouped["traditional_volatility_fwd"].quantile(0.975).reindex(trader_counts).values
    tok_lo = grouped["tokenized_volatility_fwd"].quantile(0.025).reindex(trader_counts).values
    tok_hi = grouped["tokenized_volatility_fwd"].quantile(0.975).reindex(trader_counts).values
    spread_lo = grouped["spread_volatility_fwd"].quantile(0.025).reindex(trader_counts).values
    spread_hi = grouped["spread_volatility_fwd"].quantile(0.975).reindex(trader_counts).values

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(trader_counts, trad_mean, marker="o", linewidth=2, label="Traditional mean annual vol")
    ax1.fill_between(trader_counts, trad_lo, trad_hi, alpha=0.2, label="Traditional 95% band")
    ax1.plot(trader_counts, tok_mean, marker="o", linewidth=2, label="Tokenized mean annual vol")
    ax1.fill_between(trader_counts, tok_lo, tok_hi, alpha=0.2, label="Tokenized 95% band")
    ax1.set_title("Forward Annualized Volatility vs Trader Count (Multi-Seed)")
    ax1.set_xlabel("Tokenized trader bots")
    ax1.set_ylabel("Annualized volatility (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(trader_counts, spread_mean, marker="o", linewidth=2, color="#8e44ad", label="Vol spread mean")
    ax2.fill_between(trader_counts, spread_lo, spread_hi, alpha=0.25, color="#8e44ad", label="Vol spread 95% band")
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Volatility Spread: Tokenized - Traditional")
    ax2.set_xlabel("Tokenized trader bots")
    ax2.set_ylabel("Volatility spread (pp)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PNG charts for multi-seed results.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="multi_seed_liquidity_sweep_results.csv",
        help="Input CSV filename under outputs/ (or absolute path).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="multi_seed",
        help="Prefix for generated PNG files.",
    )
    parser.add_argument(
        "--trader-counts",
        type=str,
        default="",
        help="Comma-separated tokenized trader counts to include (default: all rows in CSV).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir.parent / "outputs"

    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = outputs_dir / input_csv

    df = _load_data(input_csv)

    if args.trader_counts.strip():
        tc = [int(x.strip()) for x in args.trader_counts.split(",") if x.strip()]
        before = df
        df = df[df["trader_count"].isin(tc)].copy()
        if df.empty:
            have = sorted(before["trader_count"].unique().tolist())
            raise ValueError(f"No rows match --trader-counts {tc}. CSV has trader_count: {have}")
        missing = sorted(set(tc) - set(df["trader_count"].unique().tolist()))
        if missing:
            raise ValueError(f"--trader-counts includes counts with no data: {missing}")

    price_proxy_png = outputs_dir / f"{args.output_prefix}_price_evolution_proxy.png"
    key_metrics_png = outputs_dir / f"{args.output_prefix}_key_metrics.png"
    annual_volatility_png = outputs_dir / f"{args.output_prefix}_annual_volatility.png"

    _plot_price_proxy(df, price_proxy_png)
    _plot_key_metrics(df, key_metrics_png)
    _plot_annual_volatility(df, annual_volatility_png)

    print(f"Saved: {price_proxy_png}")
    print(f"Saved: {key_metrics_png}")
    print(f"Saved: {annual_volatility_png}")


if __name__ == "__main__":
    main()
