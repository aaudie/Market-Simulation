"""
Complete Housing Liquidity Analysis Pipeline

End-to-end analysis workflow:

  EMPIRICAL FOUNDATIONS
    Step 1 — Traditional CRE baseline: 72+ years of historical data
    Step 2 — Bayesian CRE transition matrix: pooled Dirichlet-Multinomial
               estimate from pooled basket (O / NNN / WPC / ADC / VNQ);
               saves bayesian_cre_transition.npz used by Layer 2

  THESIS ANALYSIS (three-layer quantitative framework)
    Step 3 — Layer 1: Analytical Markov chain results
               (stationary distributions, mean first passage times,
                sojourn times, bootstrap CIs; P_TOKENIZED = Bayesian P_mean)
    Step 4 — Layer 2: Monte Carlo analysis
               (5 000 paths × 20 years, distributional results, t-tests;
                P_TOKENIZED drawn from Bayesian posterior in Step 3)
    Step 5 — Layer 3: Adoption sensitivity sweep
               (RWA empirical adoption curve + α-interpolation sensitivity)

  SIMULATION BENCHMARK
    Step 6 — CSV benchmark (tokenized vs observed traditional data)
    Step 7 — Simulated projection benchmark (traditional=1k bots baseline)
    Step 8 — Trader-bot scale sweep on tokenized path (1.5k / 2k / 2.5k / 5k bots)

Usage:
    python3 run_complete_analysis.py
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def print_header(text: str) -> None:
    """
    Print formatted section header.
    
    Args:
        text: Header text to display
    """
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def run_script(script_name: str, description: str, extra_args: list[str] | None = None) -> bool:
    """
    Execute a Python script and report results.
    
    Args:
        script_name: Name of script file to execute
        description: Human-readable description of the script
        extra_args: Optional CLI arguments passed to the script
        
    Returns:
        True if script executed successfully, False otherwise
    """
    print(f"\n🔄 {description}...")
    print(f"   Running: {script_name}")
    
    try:
        command = [sys.executable, script_name, *(extra_args or [])]
        subprocess.run(
            command,
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✅ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR")
        print(f"   Error: {e}")
        return False


def main() -> int:
    """
    Execute complete analysis pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("HOUSING MARKET LIQUIDITY ANALYSIS - COMPLETE PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get paths
    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir.parent / "outputs"
    
    results = {}

    # ==========================================================================
    # Step 1: Traditional CRE Baseline
    # ==========================================================================
    print_header("STEP 1: TRADITIONAL CRE BASELINE ANALYSIS")
    print("Analyzing 72+ years of commercial real estate data...")
    results['cre_analysis'] = run_script(
        str(script_dir / 'analyze_cre_regimes.py'),
        'Traditional CRE Regime Analysis'
    )

    # ==========================================================================
    # Step 2: Bayesian CRE Transition Matrix
    # ==========================================================================
    print_header("STEP 2: BAYESIAN CRE TRANSITION MATRIX (O / NNN / WPC / ADC / VNQ)")
    print("Pooling transition counts across net-lease REITs; fitting Dirichlet posterior...")
    print("Saves bayesian_cre_transition.npz — used as P_TOKENIZED in Step 5.")
    results['bayesian_cre'] = run_script(
        str(script_dir / 'bayesian_cre_transition.py'),
        'Bayesian CRE Transition Matrix'
    )

    # ==========================================================================
    # Step 3: Layer 1 — Analytical Markov Chain Results
    # ==========================================================================
    print_header("STEP 3: LAYER 1 — ANALYTICAL MARKOV CHAIN ANALYSIS")
    print("Computing stationary distributions, passage times, and bootstrap CIs...")
    results['layer1'] = run_script(
        str(script_dir / 'analytical_markov.py'),
        'Analytical Markov Chain Analysis'
    )

    # ==========================================================================
    # Step 4: Layer 2 — Monte Carlo Analysis
    # ==========================================================================
    print_header("STEP 4: LAYER 2 — MONTE CARLO ANALYSIS")
    print("Running 5,000 paths × 240 months. Reporting distributions + t-tests...")
    print("P_TOKENIZED loaded from Bayesian posterior (Step 3).")
    results['layer2'] = run_script(
        str(script_dir / 'monte_carlo_analysis.py'),
        'Monte Carlo Analysis'
    )

    # ==========================================================================
    # Step 5: Layer 3 — Adoption Sensitivity (RWA data + α-sweep)
    # ==========================================================================
    print_header("STEP 5: LAYER 3 — ADOPTION SENSITIVITY ANALYSIS")
    print("Fitting sigmoid to RWA TVL data; sweeping α from 0 → 1...")
    results['layer3'] = run_script(
        str(script_dir / 'adoption_sensitivity.py'),
        'Adoption Sensitivity Analysis'
    )

    # ==========================================================================
    # Step 6: Housing Market Simulation (CSV benchmark)
    # ==========================================================================
    print_header("STEP 6: HOUSING MARKET SIMULATION (CSV BENCHMARK)")
    print("Running tokenized simulation against observed traditional CSV path...")
    results['housing_sim_csv'] = run_script(
        str(script_dir / 'housing_liquidity_comparison.py'),
        'Housing Market Comparison (CSV benchmark)',
        extra_args=["--comparison-mode", "csv_benchmark"],
    )

    # ==========================================================================
    # Step 7: Housing Market Simulation (simulated projection benchmark)
    # ==========================================================================
    print_header("STEP 7: HOUSING MARKET SIMULATION (SIMULATED PROJECTION BENCHMARK)")
    print("Running projected traditional=1,000 bots baseline vs tokenized=1,000 bots...")
    results['housing_sim_projection'] = run_script(
        str(script_dir / 'housing_liquidity_comparison.py'),
        'Housing Market Comparison (Sim projection baseline)',
        extra_args=["--comparison-mode", "sim_projection", "--trader-count", "1000"],
    )

    # ==========================================================================
    # Step 8: Housing Market Simulation (trader-bot scale sweep)
    # ==========================================================================
    print_header("STEP 8: HOUSING MARKET SIMULATION (TRADER-BOT SCALE SWEEP)")
    print("Running tokenized sensitivity sweep with traditional fixed at 1,000 bots...")
    for trader_count in (1500, 2000, 2500, 5000):
        result_key = f"housing_sim_{trader_count}"
        results[result_key] = run_script(
            str(script_dir / 'housing_liquidity_comparison.py'),
            f'Housing Market Comparison (sim projection, tokenized {trader_count} bots)',
            extra_args=[
                "--comparison-mode",
                "sim_projection",
                "--trader-count",
                str(trader_count),
            ],
        )

    # ==========================================================================
    # Summary Report
    # ==========================================================================
    print_header("ANALYSIS COMPLETE")

    step_labels = {
        'cre_analysis': 'Step 1 — CRE Baseline Analysis',
        'bayesian_cre':  'Step 2 — Bayesian CRE Transition Matrix',
        'layer1':        'Step 3 — Layer 1 Analytical Markov',
        'layer2':        'Step 4 — Layer 2 Monte Carlo',
        'layer3':        'Step 5 — Layer 3 Adoption Sensitivity',
        'housing_sim_csv': 'Step 6 — Housing Simulation (CSV benchmark)',
        'housing_sim_projection': 'Step 7 — Housing Simulation (Sim projection baseline)',
        'housing_sim_1500':  'Step 8 — Housing Simulation (tokenized 1,500 trader bots)',
        'housing_sim_2000':  'Step 8 — Housing Simulation (tokenized 2,000 trader bots)',
        'housing_sim_2500':  'Step 8 — Housing Simulation (tokenized 2,500 trader bots)',
        'housing_sim_5000':  'Step 8 — Housing Simulation (tokenized 5,000 trader bots)',
    }
    print("\n  Results Summary:")
    for key, label in step_labels.items():
        status = "SUCCESS" if results.get(key) else "FAILED"
        mark   = "+" if results.get(key) else "x"
        print(f"   [{mark}] {label}: {status}")

    if all(results.values()):
        print("\n  All analyses completed successfully!")
        print(f"\n  Generated output files (in {outputs_dir}):")
        files = [
            ("CRE_regime_analysis.png",          "Step 1 — CRE regime chart"),
            ("bayesian_cre_transition.png",      "Step 2 — Bayesian posterior heatmap + diagnostics"),
            ("bayesian_cre_transition.npz",      "Step 2 — Posterior samples (P_mean, CI, counts)"),
            ("layer1_analytical_markov.png",     "Step 3 — Stationary distributions + passage times"),
            ("layer2_monte_carlo.png",           "Step 4 — Monte Carlo distributions"),
            ("layer2_regime_occupancy.png",      "Step 4 — Regime occupancy over time"),
            ("layer3_adoption_curve.png",        "Step 5 — Empirical RWA adoption sigmoid"),
            ("layer3_sensitivity_sweep.png",     "Step 5 — Sensitivity curves vs alpha"),
            ("housing_liquidity_comparison.png", "Step 6 — CSV benchmark (tokenized=1,000 bots)"),
            ("housing_liquidity_comparison_sim_projection.png", "Step 7 — Sim projection baseline (traditional=1,000; tokenized=1,000)"),
            ("housing_liquidity_comparison_sim_projection_1500_traders.png", "Step 8 — Sim projection sweep (tokenized 1,500 bots)"),
            ("housing_liquidity_comparison_sim_projection_2000_traders.png", "Step 8 — Sim projection sweep (tokenized 2,000 bots)"),
            ("housing_liquidity_comparison_sim_projection_2500_traders.png", "Step 8 — Sim projection sweep (tokenized 2,500 bots)"),
            ("housing_liquidity_comparison_sim_projection_5000_traders.png", "Step 8 — Sim projection sweep (tokenized 5,000 bots)"),
        ]
        for fname, desc in files:
            exists = (outputs_dir / fname).exists()
            mark   = "+" if exists else "x"
            print(f"   [{mark}] {fname}  ({desc})")

        print("\n  Thesis-ready quantitative findings:")
        print("   • Layer 1: Exact stationary distributions with 95% bootstrap CIs")
        print("   • Layer 2: Distributional results across 5,000 paths with p-values")
        print("   • Layer 3: Empirically-grounded adoption trajectory from RWA data")
        print("   • Layer 3: Sensitivity curve showing outcomes for every alpha level")

    else:
        print("\n  Some steps failed. Please review error messages above.")
        return 1
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
