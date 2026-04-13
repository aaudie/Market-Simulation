"""
Complete Housing Liquidity Analysis Pipeline

End-to-end analysis workflow:

  EMPIRICAL FOUNDATIONS
    Step 1 — Traditional CRE baseline: 72+ years of historical data
    Step 2 — REIT regime analysis: empirical VNQ transition matrix
    Step 3 — Bayesian CRE transition matrix: pooled Dirichlet-Multinomial
               estimate from net-lease REIT basket (O / NNN / WPC / ADC);
               saves bayesian_cre_transition.npz used by Layer 2

  THESIS ANALYSIS (three-layer quantitative framework)
    Step 4 — Layer 1: Analytical Markov chain results
               (stationary distributions, mean first passage times,
                sojourn times, bootstrap confidence intervals)
    Step 5 — Layer 2: Monte Carlo analysis
               (5 000 paths × 20 years, distributional results, t-tests;
                P_TOKENIZED drawn from Bayesian posterior in Step 3)
    Step 6 — Layer 3: Adoption sensitivity sweep
               (RWA empirical adoption curve + α-interpolation sensitivity)

  SIMULATION BENCHMARK
    Step 7 — Housing market comparison (single-run visualisation)

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


def run_script(script_name: str, description: str) -> bool:
    """
    Execute a Python script and report results.
    
    Args:
        script_name: Name of script file to execute
        description: Human-readable description of the script
        
    Returns:
        True if script executed successfully, False otherwise
    """
    print(f"\n🔄 {description}...")
    print(f"   Running: {script_name}")
    
    try:
        subprocess.run(
            [sys.executable, script_name],
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
    # Step 2: REIT Regime Analysis
    # ==========================================================================
    print_header("STEP 2: EMPIRICAL REIT ANALYSIS (VNQ 2005-2026)")
    print("Fetching VNQ data and extracting empirical transition matrix...")
    results['reit_analysis'] = run_script(
        str(script_dir / 'analyze_reit_regimes.py'),
        'REIT Regime Analysis'
    )

    # ==========================================================================
    # Step 3: Bayesian CRE Transition Matrix
    # ==========================================================================
    print_header("STEP 3: BAYESIAN CRE TRANSITION MATRIX (O / NNN / WPC / ADC)")
    print("Pooling transition counts across net-lease REITs; fitting Dirichlet posterior...")
    print("Saves bayesian_cre_transition.npz — used as P_TOKENIZED in Step 5.")
    results['bayesian_cre'] = run_script(
        str(script_dir / 'bayesian_cre_transition.py'),
        'Bayesian CRE Transition Matrix'
    )

    # ==========================================================================
    # Step 4: Layer 1 — Analytical Markov Chain Results
    # ==========================================================================
    print_header("STEP 4: LAYER 1 — ANALYTICAL MARKOV CHAIN ANALYSIS")
    print("Computing stationary distributions, passage times, and bootstrap CIs...")
    results['layer1'] = run_script(
        str(script_dir / 'analytical_markov.py'),
        'Analytical Markov Chain Analysis'
    )

    # ==========================================================================
    # Step 5: Layer 2 — Monte Carlo Analysis
    # ==========================================================================
    print_header("STEP 5: LAYER 2 — MONTE CARLO ANALYSIS")
    print("Running 5,000 paths × 240 months. Reporting distributions + t-tests...")
    print("P_TOKENIZED loaded from Bayesian posterior (Step 3).")
    results['layer2'] = run_script(
        str(script_dir / 'monte_carlo_analysis.py'),
        'Monte Carlo Analysis'
    )

    # ==========================================================================
    # Step 6: Layer 3 — Adoption Sensitivity (RWA data + α-sweep)
    # ==========================================================================
    print_header("STEP 6: LAYER 3 — ADOPTION SENSITIVITY ANALYSIS")
    print("Fitting sigmoid to RWA TVL data; sweeping α from 0 → 1...")
    results['layer3'] = run_script(
        str(script_dir / 'adoption_sensitivity.py'),
        'Adoption Sensitivity Analysis'
    )

    # ==========================================================================
    # Step 7: Housing Market Simulation (single-run benchmark)
    # ==========================================================================
    print_header("STEP 7: HOUSING MARKET SIMULATION (BENCHMARK)")
    print("Running single-path traditional vs tokenized simulation...")
    print("Tokenized path uses adoption-interpolated Markov matrix toward Bayesian endpoint.")
    results['housing_sim'] = run_script(
        str(script_dir / 'housing_liquidity_comparison.py'),
        'Housing Market Comparison'
    )

    # ==========================================================================
    # Summary Report
    # ==========================================================================
    print_header("ANALYSIS COMPLETE")

    step_labels = {
        'cre_analysis': 'Step 1 — CRE Baseline Analysis',
        'reit_analysis': 'Step 2 — REIT Regime Analysis',
        'bayesian_cre':  'Step 3 — Bayesian CRE Transition Matrix',
        'layer1':        'Step 4 — Layer 1 Analytical Markov',
        'layer2':        'Step 5 — Layer 2 Monte Carlo',
        'layer3':        'Step 6 — Layer 3 Adoption Sensitivity',
        'housing_sim':   'Step 7 — Housing Market Simulation',
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
            ("VNQ_regime_analysis.png",          "Step 2 — REIT regime chart"),
            ("bayesian_cre_transition.png",      "Step 3 — Bayesian posterior heatmap + diagnostics"),
            ("bayesian_cre_transition.npz",      "Step 3 — Posterior samples (P_mean, CI, counts)"),
            ("layer1_analytical_markov.png",     "Step 4 — Stationary distributions + passage times"),
            ("layer2_monte_carlo.png",           "Step 5 — Monte Carlo distributions"),
            ("layer2_regime_occupancy.png",      "Step 5 — Regime occupancy over time"),
            ("layer3_adoption_curve.png",        "Step 6 — Empirical RWA adoption sigmoid"),
            ("layer3_sensitivity_sweep.png",     "Step 6 — Sensitivity curves vs alpha"),
            ("housing_liquidity_comparison.png", "Step 7 — Single-run comparison"),
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
