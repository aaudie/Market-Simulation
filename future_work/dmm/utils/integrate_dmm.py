"""
Integrate Deep Markov Model with Market Simulator

Demonstrates three approaches to regime dynamics:
1. Fixed Matrix: Traditional static transition probabilities
2. Hybrid Model: Interpolation between empirical matrices (no training required)
3. Deep Markov Model: Trained neural network with context-aware transitions

The Deep Markov Model loads a pre-trained checkpoint from train_dmm_with_qfclient.py
and uses it for sophisticated regime predictions.

Usage:
    python3 dmm/integrate_dmm.py
"""

import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Set matplotlib to non-interactive backend to avoid blocking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from typing import List, Optional

from sim.market_simulator import MarketSimulator
from sim.data_loader import load_cre_csv
from sim.calibration import calibrate_from_history
from sim.types import ScenarioParams, HistoricalPoint
from dmm.utils.use_empirical_matrices import HybridMarkovModel
from dmm import DeepMarkovModel, TORCH_AVAILABLE

class HybridMarkovSimulator(MarketSimulator):
    """
    Enhanced MarketSimulator that uses Hybrid Markov Model for regime dynamics.
    
    Instead of fixed transition matrices, this uses context-dependent interpolation between
    traditional and tokenized matrices to:
    1. Predict regime transitions based on market context (tokenization level)
    2. Generate volatility parameters conditioned on regime
    3. Adapt behavior smoothly as market structure changes
    
    This approach provides flexibility without requiring extensive training data.
    """
    
    def __init__(self, hybrid_model: HybridMarkovModel):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.use_hybrid_regimes = False
        
    def enable_hybrid_regimes(self, start_regime: str = "neutral") -> None:
        """
        Enable Hybrid Markov Model regime dynamics.
        
        Args:
            start_regime: Initial regime state
        """
        self.use_hybrid_regimes = True
        self.regime = start_regime
        self._markov_regime = start_regime
        print(f"✓ Hybrid regime dynamics enabled (starting in '{start_regime}')")
    
    def _step_fundamental_month(self) -> None:
        """
        Override fundamental step to use Hybrid Model for regime transitions.
        """
        if self._fundamental_price is None:
            self._fundamental_price = float(self.order_book.last_price)
        
        if self.month_idx == 0 and not self._micro_scaled_once:
            self._scale_micro_to_fundamental_once()
        
        # Replay history or project with GBM
        if self._history_index < len(self._history_prices):
            new_price = self._history_prices[self._history_index]
            self._history_index += 1
        else:
            if self._scenario is None:
                new_price = self._fundamental_price
            else:
                mu = self._scenario.mu_monthly
                sigma = self.current_sigma_monthly
                dt = 1.0
                z = self._sample_standard_normal()
                log_ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
                new_price = self._fundamental_price * math.exp(log_ret)
        
        self._fundamental_price = float(new_price)
        
        # Use Hybrid Model for regime transition instead of fixed Markov chain
        if self.use_hybrid_regimes:
            # Build context for Hybrid Model
            # Context includes tokenization/adoption level
            context = {
                'is_tokenized': self.adoption,  # Use adoption rate directly for interpolation
                'adoption_rate': self.adoption
            }
            
            # Predict next regime using Hybrid Model
            next_regime, probs = self.hybrid_model.predict_next_regime(self.regime, context)
            
            # Sample from predicted distribution (adds stochasticity)
            regime_idx = np.random.choice(len(self.hybrid_model.regime_names), p=probs)
            self._markov_regime = self.hybrid_model.regime_names[regime_idx]
        
        # Update volatility regime from realized vol (same as before)
        self._update_vol_regime(self._fundamental_price)
        
        # Update fundamental anchor
        self.target_price = int(round(self._fundamental_price))
        
        if (not self._micro_initialized) and (self.month_idx == 0) and (self.target_price is not None):
            p0 = int(self.target_price)
            self.order_book.reset_with_price(p0)
            self.live_candlestick.reset(int(self.order_book.last_price))
            self._micro_initialized = True


class DeepMarkovSimulator(MarketSimulator):
    """
    Enhanced MarketSimulator that uses trained Deep Markov Model for regime dynamics.
    
    Uses a neural network trained on historical data to:
    1. Learn complex regime transition patterns from data
    2. Adapt to market context (tokenization, adoption, time)
    3. Capture non-linear relationships between market conditions and regimes
    
    This approach requires pre-training but can capture more sophisticated patterns.
    """
    
    def __init__(self, deep_model: DeepMarkovModel):
        super().__init__()
        self.deep_model = deep_model
        self.use_deep_regimes = False
        
    def enable_deep_regimes(self, start_regime: str = "neutral") -> None:
        """
        Enable Deep Markov Model regime dynamics.
        
        Args:
            start_regime: Initial regime state
        """
        self.use_deep_regimes = True
        self.regime = start_regime
        self._markov_regime = start_regime
        print(f"✓ Deep Markov regime dynamics enabled (starting in '{start_regime}')")
    
    def _step_fundamental_month(self) -> None:
        """
        Override fundamental step to use Deep Markov Model for regime transitions.
        """
        if self._fundamental_price is None:
            self._fundamental_price = float(self.order_book.last_price)
        
        if self.month_idx == 0 and not self._micro_scaled_once:
            self._scale_micro_to_fundamental_once()
        
        # Replay history or project with GBM
        if self._history_index < len(self._history_prices):
            new_price = self._history_prices[self._history_index]
            self._history_index += 1
        else:
            if self._scenario is None:
                new_price = self._fundamental_price
            else:
                mu = self._scenario.mu_monthly
                sigma = self.current_sigma_monthly
                dt = 1.0
                z = self._sample_standard_normal()
                log_ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
                new_price = self._fundamental_price * math.exp(log_ret)
        
        self._fundamental_price = float(new_price)
        
        # Use Deep Markov Model for regime transition
        if self.use_deep_regimes:
            # Build context for Deep Model
            # Normalize time to [0, 1] based on month index
            time_normalized = min(self.month_idx / 1000.0, 1.0)  # Assume max 1000 months
            
            context = {
                'is_tokenized': self.adoption,
                'time_normalized': time_normalized,
                'adoption_rate': self.adoption
            }
            
            # Predict next regime using Deep Model
            next_regime, probs = self.deep_model.predict_next_regime(self.regime, context)
            
            # Sample from predicted distribution (adds stochasticity)
            regime_idx = np.random.choice(len(self.deep_model.regime_names), p=probs)
            self._markov_regime = self.deep_model.regime_names[regime_idx]
        
        # Update volatility regime from realized vol
        self._update_vol_regime(self._fundamental_price)
        
        # Update fundamental anchor
        self.target_price = int(round(self._fundamental_price))
        
        if (not self._micro_initialized) and (self.month_idx == 0) and (self.target_price is not None):
            p0 = int(self.target_price)
            self.order_book.reset_with_price(p0)
            self.live_candlestick.reset(int(self.order_book.last_price))
            self._micro_initialized = True


def run_comparison_simulation(
    hybrid_model: HybridMarkovModel,
    deep_model: Optional[DeepMarkovModel],
    path_to_csv: str,
    months_ahead: int = 60,
    ticks_per_candle: int = 50,
    n_runs: int = 10
) -> dict:
    """
    Compare three approaches: Fixed Matrix vs Hybrid Model vs Deep Markov Model.
    
    Args:
        hybrid_model: Hybrid Markov Model (interpolates between empirical matrices)
        deep_model: Trained Deep Markov Model (optional, trained neural network)
        path_to_csv: Path to historical CRE data
        months_ahead: Months to simulate into future
        ticks_per_candle: Micro ticks per month
        n_runs: Number of Monte Carlo runs
    
    Returns:
        Dictionary with simulation results
    """
    print("="*70)
    print("RUNNING THREE-WAY COMPARISON SIMULATIONS")
    print("="*70)
    
    # Load historical data
    history = load_cre_csv(path_to_csv)
    calib = calibrate_from_history(history)
    
    # Create scenario
    scenario = ScenarioParams(
        name="tokenized_comparison",
        mu_monthly=calib.mu_monthly * 1.10,
        sigma_monthly=calib.sigma_monthly * 0.90,
    )
    
    # Results storage
    results = {
        'hybrid': {'prices': [], 'regimes': [], 'volatilities': []},
        'fixed': {'prices': [], 'regimes': [], 'volatilities': []}
    }
    
    # Add deep model results if available
    if deep_model is not None:
        results['deep'] = {'prices': [], 'regimes': [], 'volatilities': []}
        print(f"\n✓ Deep Markov Model available - running three-way comparison")
    else:
        print(f"\n⚠ Deep Markov Model not available - running two-way comparison")
    
    print(f"\nRunning {n_runs} Monte Carlo simulations...")
    
    for run in range(n_runs):
        # --- Hybrid Model-based simulator ---
        sim_hybrid = HybridMarkovSimulator(hybrid_model)
        sim_hybrid.attach_history_and_scenario(history, scenario)
        sim_hybrid.enable_hybrid_regimes(start_regime="neutral")
        
        hybrid_prices = []
        hybrid_regimes = []
        hybrid_vols = []
        
        total_months = len(history) + months_ahead
        for m in range(total_months):
            sim_hybrid.run_micro_ticks(ticks_per_candle)
            hybrid_prices.append(sim_hybrid.order_book.last_price)
            hybrid_regimes.append(sim_hybrid.regime)
            hybrid_vols.append(sim_hybrid.current_sigma_monthly)
            sim_hybrid.roll_candle()
        
        results['hybrid']['prices'].append(hybrid_prices)
        results['hybrid']['regimes'].append(hybrid_regimes)
        results['hybrid']['volatilities'].append(hybrid_vols)
        
        # --- Deep Markov Model simulator (if available) ---
        if deep_model is not None:
            sim_deep = DeepMarkovSimulator(deep_model)
            sim_deep.attach_history_and_scenario(history, scenario)
            sim_deep.enable_deep_regimes(start_regime="neutral")
            
            deep_prices = []
            deep_regimes = []
            deep_vols = []
            
            for m in range(total_months):
                sim_deep.run_micro_ticks(ticks_per_candle)
                deep_prices.append(sim_deep.order_book.last_price)
                deep_regimes.append(sim_deep.regime)
                deep_vols.append(sim_deep.current_sigma_monthly)
                sim_deep.roll_candle()
            
            results['deep']['prices'].append(deep_prices)
            results['deep']['regimes'].append(deep_regimes)
            results['deep']['volatilities'].append(deep_vols)
        
        # --- Fixed matrix simulator ---
        sim_fixed = MarketSimulator()
        sim_fixed.attach_history_and_scenario(history, scenario)
        
        # Use empirical tokenized matrix from main.py
        P_TOKENIZED = [
            [0.8174, 0.1739, 0.0087, 0.0000],
            [0.1887, 0.7736, 0.0283, 0.0094],
            [0.0500, 0.2000, 0.7500, 0.0000],
            [0.0000, 0.0000, 0.1000, 0.9000],
        ]
        sim_fixed.enable_markov_regimes(P_TOKENIZED, start_state="neutral")
        
        fixed_prices = []
        fixed_regimes = []
        fixed_vols = []
        
        for m in range(total_months):
            sim_fixed.run_micro_ticks(ticks_per_candle)
            fixed_prices.append(sim_fixed.order_book.last_price)
            fixed_regimes.append(sim_fixed.regime)
            fixed_vols.append(sim_fixed.current_sigma_monthly)
            sim_fixed.roll_candle()
        
        results['fixed']['prices'].append(fixed_prices)
        results['fixed']['regimes'].append(fixed_regimes)
        results['fixed']['volatilities'].append(fixed_vols)
        
        if (run + 1) % 2 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")
    
    print("✓ Simulations complete\n")
    
    return results


def analyze_and_visualize_results(results: dict) -> None:
    """
    Analyze and visualize comparison results.
    """
    print("="*70)
    print("ANALYSIS OF RESULTS")
    print("="*70)
    
    # Convert to arrays
    hybrid_prices = np.array(results['hybrid']['prices'])
    fixed_prices = np.array(results['fixed']['prices'])
    has_deep = 'deep' in results
    
    if has_deep:
        deep_prices = np.array(results['deep']['prices'])
    
    # Calculate statistics
    hybrid_mean = hybrid_prices.mean(axis=0)
    hybrid_std = hybrid_prices.std(axis=0)
    fixed_mean = fixed_prices.mean(axis=0)
    fixed_std = fixed_prices.std(axis=0)
    
    if has_deep:
        deep_mean = deep_prices.mean(axis=0)
        deep_std = deep_prices.std(axis=0)
    
    # Final price statistics
    print("\n1. Final Price Distribution:")
    print(f"   Hybrid: Mean={hybrid_prices[:, -1].mean():.2f}, Std={hybrid_prices[:, -1].std():.2f}")
    if has_deep:
        print(f"   Deep:   Mean={deep_prices[:, -1].mean():.2f}, Std={deep_prices[:, -1].std():.2f}")
    print(f"   Fixed:  Mean={fixed_prices[:, -1].mean():.2f}, Std={fixed_prices[:, -1].std():.2f}")
    
    # Regime statistics
    print("\n2. Regime Distribution (Forecast Period):")
    regime_names = ['calm', 'neutral', 'volatile', 'panic']
    
    print("\n   Hybrid Model:")
    for regime in regime_names:
        count = sum(sum(1 for r in run if r == regime) 
                   for run in results['hybrid']['regimes'])
        total = sum(len(run) for run in results['hybrid']['regimes'])
        pct = 100 * count / total
        print(f"   - {regime:10s}: {pct:5.1f}%")
    
    if has_deep:
        print("\n   Deep Markov Model:")
        for regime in regime_names:
            count = sum(sum(1 for r in run if r == regime) 
                       for run in results['deep']['regimes'])
            total = sum(len(run) for run in results['deep']['regimes'])
            pct = 100 * count / total
            print(f"   - {regime:10s}: {pct:5.1f}%")
    
    print("\n   Fixed Matrix:")
    for regime in regime_names:
        count = sum(sum(1 for r in run if r == regime) 
                   for run in results['fixed']['regimes'])
        total = sum(len(run) for run in results['fixed']['regimes'])
        pct = 100 * count / total
        print(f"   - {regime:10s}: {pct:5.1f}%")
    
    # Volatility statistics
    hybrid_vols = np.array(results['hybrid']['volatilities'])
    fixed_vols = np.array(results['fixed']['volatilities'])
    
    print("\n3. Volatility Statistics (Monthly):")
    print(f"   Hybrid: Mean={hybrid_vols.mean()*100:.3f}%, Std={hybrid_vols.std()*100:.3f}%")
    if has_deep:
        deep_vols = np.array(results['deep']['volatilities'])
        print(f"   Deep:   Mean={deep_vols.mean()*100:.3f}%, Std={deep_vols.std()*100:.3f}%")
    print(f"   Fixed:  Mean={fixed_vols.mean()*100:.3f}%, Std={fixed_vols.std()*100:.3f}%")
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Price trajectories
    ax1 = plt.subplot(2, 2, 1)
    months = np.arange(len(hybrid_mean))
    
    # Plot mean ± std for all models
    ax1.plot(months, hybrid_mean, label='Hybrid', color='#3498db', linewidth=2)
    ax1.fill_between(months, hybrid_mean - hybrid_std, hybrid_mean + hybrid_std, 
                     alpha=0.2, color='#3498db')
    
    if has_deep:
        ax1.plot(months, deep_mean, label='Deep Markov', color='#9b59b6', linewidth=2)
        ax1.fill_between(months, deep_mean - deep_std, deep_mean + deep_std, 
                         alpha=0.2, color='#9b59b6')
    
    ax1.plot(months, fixed_mean, label='Fixed', color='#e74c3c', linewidth=2)
    ax1.fill_between(months, fixed_mean - fixed_std, fixed_mean + fixed_std, 
                     alpha=0.2, color='#e74c3c')
    
    title = 'Price Trajectories: Fixed vs Hybrid vs Deep' if has_deep else 'Price Trajectories: Fixed vs Hybrid'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final price distributions
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(hybrid_prices[:, -1], bins=20, alpha=0.5, label='Hybrid', color='#3498db')
    if has_deep:
        ax2.hist(deep_prices[:, -1], bins=20, alpha=0.5, label='Deep', color='#9b59b6')
    ax2.hist(fixed_prices[:, -1], bins=20, alpha=0.5, label='Fixed', color='#e74c3c')
    ax2.set_title('Final Price Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Final Price')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Volatility over time
    ax3 = plt.subplot(2, 2, 3)
    hybrid_vol_mean = hybrid_vols.mean(axis=0) * 100
    fixed_vol_mean = fixed_vols.mean(axis=0) * 100
    
    ax3.plot(months, hybrid_vol_mean, label='Hybrid', color='#3498db', linewidth=2)
    if has_deep:
        deep_vol_mean = deep_vols.mean(axis=0) * 100
        ax3.plot(months, deep_vol_mean, label='Deep Markov', color='#9b59b6', linewidth=2)
    ax3.plot(months, fixed_vol_mean, label='Fixed', color='#e74c3c', linewidth=2)
    ax3.set_title('Average Volatility Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Monthly Volatility (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Regime distribution comparison
    ax4 = plt.subplot(2, 2, 4)
    
    hybrid_regime_counts = []
    fixed_regime_counts = []
    if has_deep:
        deep_regime_counts = []
    
    for regime in regime_names:
        hybrid_count = sum(sum(1 for r in run if r == regime) 
                       for run in results['hybrid']['regimes'])
        hybrid_total = sum(len(run) for run in results['hybrid']['regimes'])
        hybrid_regime_counts.append(100 * hybrid_count / hybrid_total)
        
        if has_deep:
            deep_count = sum(sum(1 for r in run if r == regime) 
                           for run in results['deep']['regimes'])
            deep_total = sum(len(run) for run in results['deep']['regimes'])
            deep_regime_counts.append(100 * deep_count / deep_total)
        
        fixed_count = sum(sum(1 for r in run if r == regime) 
                         for run in results['fixed']['regimes'])
        fixed_total = sum(len(run) for run in results['fixed']['regimes'])
        fixed_regime_counts.append(100 * fixed_count / fixed_total)
    
    x = np.arange(len(regime_names))
    
    if has_deep:
        width = 0.25
        ax4.bar(x - width, hybrid_regime_counts, width, label='Hybrid', color='#3498db', alpha=0.8)
        ax4.bar(x, deep_regime_counts, width, label='Deep', color='#9b59b6', alpha=0.8)
        ax4.bar(x + width, fixed_regime_counts, width, label='Fixed', color='#e74c3c', alpha=0.8)
    else:
        width = 0.35
        ax4.bar(x - width/2, hybrid_regime_counts, width, label='Hybrid', color='#3498db', alpha=0.8)
        ax4.bar(x + width/2, fixed_regime_counts, width, label='Fixed', color='#e74c3c', alpha=0.8)
    
    ax4.set_title('Regime Distribution Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([r.capitalize() for r in regime_names])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    filename = "three_way_comparison.png" if has_deep else "hybrid_vs_fixed_comparison.png"
    output_path = parent_dir / "outputs" / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    # plt.show()  # Commented out to avoid blocking in headless environments


def main():
    """Main execution."""
    print("="*70)
    print(" DEEP MARKOV MODEL INTEGRATION WITH MARKET SIMULATOR")
    print("="*70)
    
    # Initialize Hybrid Markov Model (no training required!)
    print(f"\n1. Initializing Hybrid Markov Model...")
    print(f"   - Uses empirical transition matrices")
    print(f"   - Interpolates between traditional and tokenized markets")
    print(f"   - No training required (works immediately)")
    
    hybrid_model = HybridMarkovModel()
    print(f"   ✓ Hybrid model ready")
    
    # Try to load trained Deep Markov Model
    print(f"\n2. Loading Deep Markov Model...")
    deep_model = None
    model_path = parent_dir / "outputs" / "deep_markov_model_qfclient.pt"
    
    if TORCH_AVAILABLE and model_path.exists():
        try:
            deep_model = DeepMarkovModel(
                regime_names=['calm', 'neutral', 'volatile', 'panic'],
                context_dim=3,
                hidden_dim=128
            )
            deep_model.load(str(model_path))
            print(f"   ✓ Deep Markov Model loaded from {model_path}")
            print(f"   - Trained neural network with context-aware transitions")
        except Exception as e:
            print(f"   ⚠ Failed to load Deep Markov Model: {e}")
            print(f"   - Continuing with Hybrid model only")
    else:
        if not TORCH_AVAILABLE:
            print(f"   ⚠ PyTorch not available")
        elif not model_path.exists():
            print(f"   ⚠ Trained model not found at {model_path}")
        print(f"   - To use Deep Markov Model, run: python3 dmm/train_dmm_with_qfclient.py")
        print(f"   - Continuing with Hybrid model only")
    
    # Load CRE data
    print(f"\n3. Loading historical data...")
    csv_path = parent_dir / "data" / "cre_monthly.csv"
    if not csv_path.exists():
        print(f"\nERROR: CRE data not found at {csv_path}")
        print("Please run: python3 scripts/run_complete_analysis.py")
        return
    print(f"   ✓ Data loaded from {csv_path}")
    
    # Run comparison
    print(f"\n4. Running simulations...")
    results = run_comparison_simulation(
        hybrid_model=hybrid_model,
        deep_model=deep_model,
        path_to_csv=str(csv_path),
        months_ahead=60,
        ticks_per_candle=50,
        n_runs=10
    )
    
    # Analyze and visualize
    analyze_and_visualize_results(results)
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\nKey insights:")
    print("- Hybrid model: Context-dependent interpolation (no training)")
    if deep_model is not None:
        print("- Deep Markov: Neural network learns complex patterns from data")
    print("- Fixed matrix: Traditional static probabilities (baseline)")
    print("- Comparison shows evolution from simple to sophisticated models")
    print("\nUse these simulators in your own code:")
    print("  - HybridMarkovSimulator(hybrid_model)")
    if deep_model is not None:
        print("  - DeepMarkovSimulator(deep_model)")


if __name__ == "__main__":
    import math  # Need this for DMM simulator
    main()
