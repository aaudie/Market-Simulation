"""
Multi-seed robustness sweep for housing liquidity comparison.

Runs the existing traditional-vs-tokenized sim_projection workflow across
multiple random seeds and trader counts, then reports aggregate statistics.

Usage:
  python3 scripts/multi_seed_liquidity_sweep.py
  python3 scripts/multi_seed_liquidity_sweep.py --num-seeds 100 --seed-start 100
  python3 scripts/multi_seed_liquidity_sweep.py --trader-counts 1000,2000,5000

Unless --replace-csv is set, existing output CSV rows are preserved; new rows are
merged by (seed, trader_count); duplicates are skipped without re-running.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from housing_liquidity_comparison import (
    P_TRADITIONAL,
    P_TOKENIZED,
    calculate_metrics,
    run_simulation,
)


def parse_trader_counts(value: str) -> list[int]:
    counts = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        if n <= 0:
            raise ValueError("All trader counts must be positive integers.")
        counts.append(n)
    if not counts:
        raise ValueError("Provide at least one trader count.")
    return counts


def ci_95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _safe_sharpe(return_pct: float, volatility_pct: float, risk_free_pct: float) -> float:
    if abs(volatility_pct) < 1e-12:
        return 0.0
    return (return_pct - risk_free_pct) / volatility_pct


def run_one_seed(
    csv_path: str,
    months_ahead: int,
    ticks_per_candle: int,
    trader_count: int,
    seed: int,
    use_micro_feedback: bool,
    regime_micro_weight: float,
    fundamental_micro_feedback: float,
    risk_free_rate_pct: float,
) -> dict[str, Any]:
    # Traditional simulation stays fixed at 1000 trader bots.
    _, trad_prices, trad_micro_prices, trad_regimes, history_len = run_simulation(
        csv_path=csv_path,
        months_ahead=months_ahead,
        ticks_per_candle=ticks_per_candle,
        transition_matrix=P_TRADITIONAL,
        scenario_name="Traditional Housing (Illiquid, Simulated Projection)",
        seed=seed,
        verbose=False,
        use_micro_feedback=use_micro_feedback,
        regime_micro_weight=regime_micro_weight,
        fundamental_micro_feedback=fundamental_micro_feedback,
        trader_count=1000,
    )

    # Tokenized simulation uses swept trader count.
    _, tok_prices, tok_micro_prices, tok_regimes, _ = run_simulation(
        csv_path=csv_path,
        months_ahead=months_ahead,
        ticks_per_candle=ticks_per_candle,
        transition_matrix=P_TRADITIONAL,
        scenario_name="Tokenized Housing (REIT-like)",
        adoption_interpolated_markov=True,
        tokenized_endpoint_matrix=P_TOKENIZED,
        seed=seed,
        verbose=False,
        use_micro_feedback=use_micro_feedback,
        regime_micro_weight=regime_micro_weight,
        fundamental_micro_feedback=fundamental_micro_feedback,
        trader_count=trader_count,
    )

    trad_fwd = calculate_metrics(trad_prices[history_len:], trad_regimes[history_len:])
    tok_fwd = calculate_metrics(tok_prices[history_len:], tok_regimes[history_len:])
    trad_micro_fwd = calculate_metrics(trad_micro_prices[history_len:], trad_regimes[history_len:])
    tok_micro_fwd = calculate_metrics(tok_micro_prices[history_len:], tok_regimes[history_len:])
    trad_sharpe = _safe_sharpe(trad_fwd["total_return"], trad_fwd["volatility"], risk_free_rate_pct)
    tok_sharpe = _safe_sharpe(tok_fwd["total_return"], tok_fwd["volatility"], risk_free_rate_pct)

    return {
        "seed": seed,
        "trader_count": trader_count,
        "traditional_return_fwd": trad_fwd["total_return"],
        "tokenized_return_fwd": tok_fwd["total_return"],
        "spread_return_fwd": tok_fwd["total_return"] - trad_fwd["total_return"],
        "traditional_volatility_fwd": trad_fwd["volatility"],
        "tokenized_volatility_fwd": tok_fwd["volatility"],
        "spread_volatility_fwd": tok_fwd["volatility"] - trad_fwd["volatility"],
        "traditional_sharpe_fwd": trad_sharpe,
        "tokenized_sharpe_fwd": tok_sharpe,
        "spread_sharpe_fwd": tok_sharpe - trad_sharpe,
        "traditional_micro_return_fwd": trad_micro_fwd["total_return"],
        "tokenized_micro_return_fwd": tok_micro_fwd["total_return"],
        "spread_micro_return_fwd": tok_micro_fwd["total_return"] - trad_micro_fwd["total_return"],
        "traditional_stress_fwd": trad_fwd["stress_time_pct"],
        "tokenized_stress_fwd": tok_fwd["stress_time_pct"],
        "spread_stress_fwd": tok_fwd["stress_time_pct"] - trad_fwd["stress_time_pct"],
    }


RESULT_FIELDNAMES = [
    "seed",
    "trader_count",
    "traditional_return_fwd",
    "tokenized_return_fwd",
    "spread_return_fwd",
    "traditional_volatility_fwd",
    "tokenized_volatility_fwd",
    "spread_volatility_fwd",
    "traditional_sharpe_fwd",
    "tokenized_sharpe_fwd",
    "spread_sharpe_fwd",
    "traditional_micro_return_fwd",
    "tokenized_micro_return_fwd",
    "spread_micro_return_fwd",
    "traditional_stress_fwd",
    "tokenized_stress_fwd",
    "spread_stress_fwd",
]


def _coerce_saved_value(col: str, val: str) -> Any:
    if col in ("seed", "trader_count"):
        return int(float(val))
    return float(val)


def load_existing_results_csv(
    out_path: Path, fieldnames: list[str]
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    """Load CSV; dedupe (seed, trader_count), keep first occurrence per key."""
    if not out_path.exists():
        return [], set()

    rows_out: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    dup_lines = 0

    with out_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], set()
        missing = set(fieldnames) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Existing {out_path.name} missing columns {sorted(missing)}; "
                "fix file or run with --replace-csv."
            )

        for raw in reader:
            if not raw or raw.get("seed") in ("", None):
                continue
            key = (
                int(float(raw["seed"])),
                int(float(raw["trader_count"])),
            )
            if key in seen:
                dup_lines += 1
                continue
            seen.add(key)
            row = {col: _coerce_saved_value(col, raw[col]) for col in fieldnames}
            rows_out.append(row)

    if dup_lines:
        print(f"  Note: skipped {dup_lines} duplicate CSV line(s); kept first of each key.")
    return rows_out, seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed trader-count sweep.")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of seeds to run per trader count.")
    parser.add_argument("--seed-start", type=int, default=42, help="Starting seed value (inclusive).")
    parser.add_argument(
        "--trader-counts",
        type=str,
        default="1000,1500,2000,2500,5000,10000",
        help="Comma-separated tokenized trader counts.",
    )
    parser.add_argument("--projection-months", type=int, default=120, help="Forward projection months.")
    parser.add_argument("--ticks-per-candle", type=int, default=50, help="Micro ticks per month.")
    parser.add_argument(
        "--risk-free-rate-pct",
        type=float,
        default=0.0,
        help="Risk-free rate (%) used for Sharpe ratio calculations.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="multi_seed_liquidity_sweep_results.csv",
        help="Output CSV filename written under outputs/.",
    )
    parser.add_argument(
        "--replace-csv",
        action="store_true",
        help="Ignore existing CSV and overwrite with results from this run only.",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="Extend runs in batches until CI width change falls below threshold.",
    )
    parser.add_argument(
        "--auto-stop-threshold",
        type=float,
        default=0.05,
        help="Relative CI-width change threshold (default 0.05 == 5%%).",
    )
    parser.add_argument(
        "--auto-stop-batch-size",
        type=int,
        default=10,
        help="Additional seeds to add per convergence check.",
    )
    parser.add_argument(
        "--auto-stop-metric",
        type=str,
        default="spread_return_fwd",
        help="Metric column used for CI-width convergence checks.",
    )
    parser.add_argument(
        "--auto-stop-metrics",
        type=str,
        default="",
        help=(
            "Comma-separated metric columns for convergence checks. "
            "If provided, overrides --auto-stop-metric."
        ),
    )
    parser.add_argument(
        "--auto-stop-min-seeds",
        type=int,
        default=20,
        help="Minimum seeds before applying auto-stop checks.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=1000,
        help="Safety cap on total seeds per trader count in auto-stop mode.",
    )
    args = parser.parse_args()

    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.projection_months <= 0:
        raise ValueError("--projection-months must be positive.")
    if args.ticks_per_candle <= 0:
        raise ValueError("--ticks-per-candle must be positive.")
    if args.auto_stop_threshold < 0.0:
        raise ValueError("--auto-stop-threshold must be non-negative.")
    if args.auto_stop_batch_size <= 0:
        raise ValueError("--auto-stop-batch-size must be positive.")
    if args.auto_stop_min_seeds <= 1:
        raise ValueError("--auto-stop-min-seeds must be > 1.")
    if args.max_seeds <= 1:
        raise ValueError("--max-seeds must be > 1.")
    if args.auto_stop_min_seeds > args.max_seeds:
        raise ValueError("--auto-stop-min-seeds cannot exceed --max-seeds.")
    valid_metrics = {
        "traditional_return_fwd",
        "tokenized_return_fwd",
        "spread_return_fwd",
        "traditional_volatility_fwd",
        "tokenized_volatility_fwd",
        "spread_volatility_fwd",
        "traditional_sharpe_fwd",
        "tokenized_sharpe_fwd",
        "spread_sharpe_fwd",
        "traditional_micro_return_fwd",
        "tokenized_micro_return_fwd",
        "spread_micro_return_fwd",
        "traditional_stress_fwd",
        "tokenized_stress_fwd",
        "spread_stress_fwd",
    }
    if args.auto_stop_metrics.strip():
        metric_list = [m.strip() for m in args.auto_stop_metrics.split(",") if m.strip()]
    else:
        metric_list = [args.auto_stop_metric]

    invalid_metrics = [m for m in metric_list if m not in valid_metrics]
    if invalid_metrics:
        allowed = ", ".join(sorted(valid_metrics))
        invalid = ", ".join(invalid_metrics)
        raise ValueError(f"Invalid auto-stop metrics: {invalid}. Allowed: {allowed}")

    trader_counts = parse_trader_counts(args.trader_counts)

    script_dir = Path(__file__).resolve().parent
    csv_path = str(script_dir.parent / "data" / "cre_monthly.csv")
    outputs_dir = script_dir.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / args.output_csv

    fieldnames = RESULT_FIELDNAMES

    if args.replace_csv:
        all_rows: list[dict[str, Any]] = []
        merged_seen: set[tuple[int, int]] = set()
        print("  --replace-csv: starting from empty merged results.")
    else:
        all_rows, merged_seen = load_existing_results_csv(out_path, fieldnames)
        print(f"  Loaded {len(all_rows)} existing row(s) from {out_path.name}.")

    use_micro_feedback = True
    regime_micro_weight = 0.25
    fundamental_micro_feedback = 0.10

    print("=" * 72)
    print("MULTI-SEED HOUSING LIQUIDITY SWEEP")
    print(f"Seed start: {args.seed_start}")
    if args.auto_stop:
        print(
            "Auto-stop: ON "
            f"(metrics={metric_list}, batch={args.auto_stop_batch_size}, "
            f"threshold={args.auto_stop_threshold * 100:.2f}%)"
        )
        print(f"Initial seeds requested: {args.num_seeds}, max seeds: {args.max_seeds}")
    else:
        print(f"Seeds requested: {args.num_seeds}")
    print(f"Trader counts: {trader_counts}")
    print("=" * 72)

    for tc in trader_counts:
        print(f"\nRunning trader_count={tc} ...")
        prev_widths: dict[str, float] = {}
        seeds_run = 0  # new simulations executed this session (this trader count only)
        next_seed = args.seed_start

        while True:
            if args.auto_stop and seeds_run >= args.max_seeds:
                print(f"  reached max seeds ({args.max_seeds}); stopping this trader count.")
                break

            batch_target = args.num_seeds if seeds_run == 0 else args.auto_stop_batch_size
            if args.auto_stop:
                batch_target = min(batch_target, args.max_seeds - seeds_run)
            batch_target = max(0, batch_target)
            if batch_target == 0:
                break

            added_this_batch = 0
            skipped_overlap = 0
            for _ in range(batch_target):
                seed = next_seed
                next_seed += 1
                key = (seed, tc)
                if key in merged_seen:
                    skipped_overlap += 1
                    continue
                row = run_one_seed(
                    csv_path=csv_path,
                    months_ahead=args.projection_months,
                    ticks_per_candle=args.ticks_per_candle,
                    trader_count=tc,
                    seed=seed,
                    use_micro_feedback=use_micro_feedback,
                    regime_micro_weight=regime_micro_weight,
                    fundamental_micro_feedback=fundamental_micro_feedback,
                    risk_free_rate_pct=args.risk_free_rate_pct,
                )
                all_rows.append(row)
                merged_seen.add(key)
                seeds_run += 1
                added_this_batch += 1

            skip_msg = f", skipped_overlap={skipped_overlap}" if skipped_overlap else ""
            print(
                f"  new simulations this session: {seeds_run}; "
                f"batch added {added_this_batch}{skip_msg}"
            )

            if not args.auto_stop:
                break

            if args.auto_stop and added_this_batch == 0:
                print(
                    "  auto-stop outer loop: batch added no new rows "
                    "(seeds overlap existing CSV); stopping this trader count."
                )
                break

            subset = [r for r in all_rows if r["trader_count"] == tc]
            rel_changes: dict[str, float] = {}
            for metric in metric_list:
                metric_values = [float(r[metric]) for r in subset]
                lo, hi = ci_95(metric_values)
                width = hi - lo
                prev = prev_widths.get(metric)
                if prev is None:
                    prev_widths[metric] = width
                    continue
                denom = abs(prev) if abs(prev) > 1e-12 else 1.0
                rel_change = abs(width - prev) / denom
                rel_changes[metric] = rel_change
                print(
                    f"    CI width({metric}): prev={prev:.6f}, "
                    f"new={width:.6f}, change={rel_change * 100:.2f}%"
                )
                prev_widths[metric] = width

            if seeds_run >= args.auto_stop_min_seeds and rel_changes:
                stabilized = [
                    m for m, rc in rel_changes.items() if rc < args.auto_stop_threshold
                ]
                if stabilized:
                    print(
                        f"    auto-stop triggered: {stabilized} stabilized (< "
                        f"{args.auto_stop_threshold * 100:.2f}%) after +{args.auto_stop_batch_size} seeds."
                    )
                    break

    all_rows.sort(key=lambda r: (int(r["trader_count"]), int(r["seed"])))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved merged seed-level results ({len(all_rows)} row(s)) to: {out_path}")

    # Print concise aggregated summary.
    print("\n" + "=" * 72)
    print("AGGREGATED SUMMARY (Projection Window)")
    print("=" * 72)
    for tc in trader_counts:
        subset = [r for r in all_rows if r["trader_count"] == tc]
        if not subset:
            print(f"\nTrader count {tc}: (no rows in merged CSV)")
            continue
        spread = [float(r["spread_return_fwd"]) for r in subset]
        tok = [float(r["tokenized_return_fwd"]) for r in subset]
        trad = [float(r["traditional_return_fwd"]) for r in subset]
        vol_spread = [float(r["spread_volatility_fwd"]) for r in subset]
        tok_vol = [float(r["tokenized_volatility_fwd"]) for r in subset]
        trad_vol = [float(r["traditional_volatility_fwd"]) for r in subset]
        sharpe_spread = [float(r["spread_sharpe_fwd"]) for r in subset]
        tok_sharpe = [float(r["tokenized_sharpe_fwd"]) for r in subset]
        trad_sharpe = [float(r["traditional_sharpe_fwd"]) for r in subset]

        spread_stats = summarize(spread)
        p_outperform = 100.0 * sum(x > 0.0 for x in spread) / len(spread)
        lo, hi = ci_95(spread)
        vol_spread_stats = summarize(vol_spread)
        vol_lo, vol_hi = ci_95(vol_spread)
        sharpe_spread_stats = summarize(sharpe_spread)
        sharpe_lo, sharpe_hi = ci_95(sharpe_spread)
        p_sharpe_outperform = 100.0 * sum(x > 0.0 for x in sharpe_spread) / len(sharpe_spread)

        print(f"\nTrader count {tc}:")
        print(f"  Mean traditional return: {np.mean(trad):+.2f}%")
        print(f"  Mean tokenized return:   {np.mean(tok):+.2f}%")
        print(f"  Mean spread (tok-trad):  {spread_stats['mean']:+.2f}%")
        print(f"  95% CI spread:           [{lo:+.2f}%, {hi:+.2f}%]")
        print(f"  P(tok > trad):           {p_outperform:.1f}%")
        print(f"  Spread p10/p50/p90:      {spread_stats['p10']:+.2f}% / {spread_stats['median']:+.2f}% / {spread_stats['p90']:+.2f}%")
        print(f"  Mean traditional vol:    {np.mean(trad_vol):.2f}%")
        print(f"  Mean tokenized vol:      {np.mean(tok_vol):.2f}%")
        print(f"  Mean vol spread:         {vol_spread_stats['mean']:+.2f}%")
        print(f"  95% CI vol spread:       [{vol_lo:+.2f}%, {vol_hi:+.2f}%]")
        print(f"  Vol spread p10/p50/p90:  {vol_spread_stats['p10']:+.2f}% / {vol_spread_stats['median']:+.2f}% / {vol_spread_stats['p90']:+.2f}%")
        print(f"  Mean traditional Sharpe: {np.mean(trad_sharpe):+.3f}")
        print(f"  Mean tokenized Sharpe:   {np.mean(tok_sharpe):+.3f}")
        print(f"  Mean Sharpe spread:      {sharpe_spread_stats['mean']:+.3f}")
        print(f"  95% CI Sharpe spread:    [{sharpe_lo:+.3f}, {sharpe_hi:+.3f}]")
        print(f"  P(tok Sharpe > trad):    {p_sharpe_outperform:.1f}%")
        print(f"  Sharpe p10/p50/p90:      {sharpe_spread_stats['p10']:+.3f} / {sharpe_spread_stats['median']:+.3f} / {sharpe_spread_stats['p90']:+.3f}")


if __name__ == "__main__":
    main()
