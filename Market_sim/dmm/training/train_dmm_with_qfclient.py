"""
Train Deep Markov Model with Real-Time Data from qfclient

This script demonstrates training the DMM using live data from
financial APIs instead of CSV files.

Advantages:
- Always up-to-date data
- Access to multiple data sources with automatic failover
- Can fetch data for any REIT or real estate security
- No manual CSV management
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent  # .../dmm/training
dmm_dir = script_dir.parent                   # .../dmm
parent_dir = dmm_dir.parent                   # .../Market_sim (contains dmm module)
sys.path.insert(0, str(parent_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from dmm import DeepMarkovModel, TORCH_AVAILABLE
from dmm.utils.qfclient_data_loader import (
    load_reit_data,
    load_multi_reit_data,
    combine_multi_reit_data,
    prepare_dmm_training_data,
    load_economic_indicators,
    QFCLIENT_AVAILABLE
)

# For traditional CRE data, still use CSV
from sim.data_loader import load_cre_csv
from sim.types import HistoricalPoint


# =============================================================================
# Data Loading with qfclient
# =============================================================================

def load_training_data_from_qfclient() -> Dict[str, np.ndarray]:
    """
    Load training data using qfclient instead of CSV files.
    
    Fetches:
    - Traditional CRE data from CSV (historical baseline)
    - Multiple REITs via qfclient (tokenized proxy)
    - Economic indicators for context (optional)
    
    Returns:
        Dictionary with prepared training data
    """
    print("="*70)
    print("LOADING DATA VIA QFCLIENT")
    print("="*70)
    
    # 1. Load traditional CRE from CSV (still needed for long history)
    print("\n1. Loading traditional CRE data from CSV...")
    csv_path = parent_dir / "data" / "cre_monthly.csv"
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Using synthetic data.")
        from dmm.utils.qfclient_data_loader import generate_synthetic_reit_data
        trad_prices = generate_synthetic_reit_data(864)  # 72 years
    else:
        history = load_cre_csv(str(csv_path))
        trad_prices = np.array([p.price for p in history])
        print(f"✓ Loaded {len(trad_prices):,} monthly CRE data points")
    
    # 2. Load multiple REITs via qfclient
    print("\n2. Loading REIT data via qfclient...")
    
    if not QFCLIENT_AVAILABLE:
        print("⚠ qfclient not available. Using synthetic tokenized data.")
        from dmm.utils.qfclient_data_loader import generate_synthetic_reit_data
        token_prices = generate_synthetic_reit_data(252)  # ~21 years
    else:
        # Import REIT symbols
        try:
            from dmm.utils.reit_symbols import TOP_20_LIQUID, ALL_REITS, get_recommended_reits
            
            # OPTION 1: Use all 100+ REITs (comprehensive but slower)
            reit_symbols = ALL_REITS
            
            # OPTION 2: Use top 50 diversified REITs (good balance)
            # reit_symbols = get_recommended_reits(n=50, diversified=True)
            
            # OPTION 3: Use top 20 liquid REITs (fast, reliable)
            # reit_symbols = TOP_20_LIQUID
            
            print(f"  Loading {len(reit_symbols)} REITs...")
            
        except ImportError:
            # Fallback to original 3 REITs
            reit_symbols = ["VNQ", "IYR", "SCHH"]
            print(f"  Using default {len(reit_symbols)} REITs")
        
        # Load all REITs with progress tracking
        from dmm.utils.qfclient_data_loader import load_multi_reit_data, combine_multi_reit_data
        
        reit_data = load_multi_reit_data(
            symbols=reit_symbols,
            years=20,  # Request 20 years (API will return what it has)
            interval="monthly",
            max_failures=20,  # Stop after 20 failures to save time
            verbose=True
        )
        
        # Print actual data received for debugging
        if reit_data:
            lengths = [len(prices) for prices in reit_data.values()]
            print(f"  Data length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f} months")
        
        if not reit_data:
            print("  ⚠ No REITs loaded. Using synthetic data.")
            from dmm.utils.qfclient_data_loader import generate_synthetic_reit_data
            token_prices = generate_synthetic_reit_data(252)
        else:
            # Combine REITs using averaging (creates smoother, more representative data)
            # Alternative: use "concatenate" to create more training sequences
            token_prices = combine_multi_reit_data(
                reit_data, 
                method="average",  # Options: "average", "median", "longest", "concatenate"
                min_length=36  # Only require 3 years minimum (enough for 1 window)
            )
            
            print(f"  ✓ Combined {len(reit_data)} REITs into {len(token_prices)} data points")
            print(f"  ✓ Successfully loaded: {', '.join(list(reit_data.keys())[:10])}...")
            if len(reit_data) > 10:
                print(f"     ... and {len(reit_data) - 10} more")
    
    # 3. Optional: Load economic indicators for context
    print("\n3. Loading economic indicators (optional)...")
    economic_data = {}
    
    if QFCLIENT_AVAILABLE:
        try:
            economic_data = load_economic_indicators(years=10)
            print(f"  ✓ Loaded {len(economic_data)} economic series")
        except Exception as e:
            print(f"  ✗ Failed to load economic data: {e}")
    else:
        print("  ⚠ Skipping economic indicators (qfclient not available)")
    
    # 4. Prepare training data
    # Use shorter windows and smaller stride to maximize training sequences
    # With limited API data, we need to extract as many examples as possible
    print("\n4. Preparing training data...")
    training_data = prepare_dmm_training_data(
        traditional_data=trad_prices,
        tokenized_data=token_prices,
        window_size=24,  # 2 years - shorter to get more sequences from limited data
        stride=3         # 3 months - aggressive overlap to maximize examples
    )
    
    # Add economic data if available
    if economic_data:
        training_data['economic_indicators'] = economic_data
    
    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)
    print(f"Total sequences: {len(training_data['prices'])}")
    print(f"Window size: {training_data['window_size']} months")
    print(f"Traditional sequences: {int((training_data['is_tokenized'] == 0).sum())}")
    print(f"Tokenized sequences: {int((training_data['is_tokenized'] == 1).sum())}")
    
    return training_data


# =============================================================================
# Training
# =============================================================================

def train_model_with_qfclient_data(
    epochs: int = 200,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    save_path: str = None
) -> DeepMarkovModel:
    """
    Train DMM using data fetched via qfclient.
    
    Uses two-phase training:
      Phase 1 (40% of epochs): Supervised on heuristic regime labels
      Phase 2 (60% of epochs): VAE fine-tuning with ELBO
    
    Args:
        epochs: Total number of training epochs
        hidden_dim: Hidden dimension for neural networks
        learning_rate: Learning rate
        save_path: Path to save trained model
        
    Returns:
        Trained DeepMarkovModel
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")
    
    print("\n" + "="*70)
    print("TRAINING DEEP MARKOV MODEL (TWO-PHASE)")
    print("="*70)
    
    # Load data
    data = load_training_data_from_qfclient()
    
    # Initialize model
    print("\nTwo-Phase Training Strategy:")
    print("  Phase 1: Supervised on volatility-based regime labels")
    print("           (teaches model what regimes look like)")
    print("  Phase 2: VAE fine-tuning with ELBO objective")
    print("           (refines boundaries using variational inference)")
    
    model = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate
    )
    
    print(f"\nModel Configuration:")
    print(f"  - Device: {model.device}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Regimes: {model.regime_names}")
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    print("="*70)
    
    history = model.train(
        prices=data['prices'],
        is_tokenized=data['is_tokenized'],
        adoption_rate=data['adoption_rates'],
        epochs=epochs,
        batch_size=16,
        supervised_fraction=0.4,
        beta_schedule='slow_linear',
        verbose=True
    )
    
    print("="*70)
    print("Training complete!\n")
    
    # Save model
    if save_path is None:
        save_path = str(parent_dir / "outputs" / "deep_markov_model_qfclient.pt")
    
    model.save(save_path)
    print(f"Model saved: {save_path}")
    
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_and_visualize(model: DeepMarkovModel, data: Dict[str, np.ndarray]):
    """
    Evaluate trained model and create visualizations.
    Matches the output format of train_dmm.py
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Extract data
    n_trad = int((data['is_tokenized'] == 0).sum())
    trad_prices = data['prices'][:n_trad]
    token_prices = data['prices'][n_trad:]
    
    # 1. Regime inference on samples
    print("\n1. Traditional CRE Sample Regime Inference:")
    trad_sample = data['prices'][0]
    trad_regimes, trad_probs = model.infer_regimes(
        prices=trad_sample,
        is_tokenized=0.0,
        adoption_rate=np.zeros(len(trad_sample))
    )
    
    print(f"   Regime distribution:")
    for regime in model.regime_names:
        count = (trad_regimes == regime).sum()
        pct = 100 * count / len(trad_regimes)
        print(f"   - {regime:10s}: {pct:5.1f}%")
    
    print("\n2. Tokenized REIT Sample Regime Inference:")
    if len(token_prices) > 0:
        token_sample = data['prices'][n_trad]
        token_regimes, token_probs = model.infer_regimes(
            prices=token_sample,
            is_tokenized=1.0,
            adoption_rate=data['adoption_rates'][n_trad]
        )
        
        print(f"   Regime distribution:")
        for regime in model.regime_names:
            count = (token_regimes == regime).sum()
            pct = 100 * count / len(token_regimes)
            print(f"   - {regime:10s}: {pct:5.1f}%")
    
    # 3. Compute learned transition matrices
    print("\n3. Learned Transition Matrices:")
    
    print("\n   Traditional CRE:")
    print("   From \\ To    Calm      Neutral   Volatile  Panic")
    print("   " + "-"*60)
    
    trad_context = {'is_tokenized': 0.0, 'time_normalized': 0.5, 'adoption_rate': 0.0}
    trad_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, trad_context)
        trad_matrix[i] = probs
        row_str = f"   {regime:10s}"
        for p in probs:
            row_str += f"  {p:.4f}"
        print(row_str)
    
    print("\n   Tokenized REIT:")
    print("   From \\ To    Calm      Neutral   Volatile  Panic")
    print("   " + "-"*60)
    
    token_context = {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    token_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, token_context)
        token_matrix[i] = probs
        row_str = f"   {regime:10s}"
        for p in probs:
            row_str += f"  {p:.4f}"
        print(row_str)
    
    # 4. Compare with empirical matrices
    print("\n4. Comparison with Empirical Matrices:")
    
    P_TRADITIONAL_EMPIRICAL = [
        [0.8591, 0.1389, 0.0020, 0.0000],
        [0.2339, 0.7186, 0.0475, 0.0000],
        [0.0339, 0.2203, 0.6949, 0.0508],
        [0.0000, 0.0000, 0.7500, 0.2500],
    ]
    
    P_TOKENIZED_EMPIRICAL = [
        [0.8174, 0.1739, 0.0087, 0.0000],
        [0.1887, 0.7736, 0.0283, 0.0094],
        [0.0500, 0.2000, 0.7500, 0.0000],
        [0.0000, 0.0000, 0.1000, 0.9000],
    ]
    
    print("\n   Traditional - Empirical vs Learned:")
    for i, regime in enumerate(model.regime_names):
        _, learned = model.predict_next_regime(regime, trad_context)
        empirical = P_TRADITIONAL_EMPIRICAL[i]
        diff = np.abs(learned - np.array(empirical))
        print(f"   {regime:10s} - MAE: {diff.mean():.4f}, Max diff: {diff.max():.4f}")
    
    print("\n   Tokenized - Empirical vs Learned:")
    for i, regime in enumerate(model.regime_names):
        _, learned = model.predict_next_regime(regime, token_context)
        empirical = P_TOKENIZED_EMPIRICAL[i]
        diff = np.abs(learned - np.array(empirical))
        print(f"   {regime:10s} - MAE: {diff.mean():.4f}, Max diff: {diff.max():.4f}")
    
    # 5. Create comprehensive visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Training Loss (with phase annotation)
    ax1 = plt.subplot(3, 2, 1)
    phases = model.training_history.get('phase', [])
    loss_hist = model.training_history['loss']
    ax1.plot(loss_hist, label='Total Loss', linewidth=2)
    
    # Mark phase boundary
    if phases:
        phase1_end = sum(1 for p in phases if p == 1)
        if 0 < phase1_end < len(loss_hist):
            ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1/2 boundary')
            ax1.text(phase1_end * 0.3, max(loss_hist) * 0.9, 'Supervised', fontsize=9, ha='center', color='gray')
            ax1.text(phase1_end + (len(loss_hist) - phase1_end) * 0.5, max(loss_hist) * 0.9, 'VAE', fontsize=9, ha='center', color='gray')
    
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(model.training_history['reconstruction_loss'], 
             label='Reconstruction', linewidth=2, color='#2ecc71')
    ax2.plot(model.training_history['kl_loss'], 
             label='KL Divergence', linewidth=2, color='#e74c3c')
    if 'supervised_loss' in model.training_history:
        ax2.plot(model.training_history['supervised_loss'], 
                 label='Supervised', linewidth=2, color='#9b59b6', linestyle='--')
    if phases:
        phase1_end = sum(1 for p in phases if p == 1)
        if 0 < phase1_end < len(loss_hist):
            ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Traditional CRE Predicted Regimes
    ax3 = plt.subplot(3, 2, 3)
    regime_colors = {'calm': 0, 'neutral': 1, 'volatile': 2, 'panic': 3}
    regime_numeric = [regime_colors[r] for r in trad_regimes]
    colors = ['#2ecc71' if r == 'calm' else '#3498db' if r == 'neutral' 
              else '#f39c12' if r == 'volatile' else '#e74c3c' for r in trad_regimes]
    
    ax3.scatter(range(len(trad_regimes)), regime_numeric, c=colors, alpha=0.6, s=30)
    ax3.set_title('Traditional CRE - Predicted Regimes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Regime')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tokenized REIT Predicted Regimes
    ax4 = plt.subplot(3, 2, 4)
    if len(token_prices) > 0:
        regime_numeric = [regime_colors[r] for r in token_regimes]
        colors = ['#2ecc71' if r == 'calm' else '#3498db' if r == 'neutral' 
                  else '#f39c12' if r == 'volatile' else '#e74c3c' for r in token_regimes]
        
        ax4.scatter(range(len(token_regimes)), regime_numeric, c=colors, alpha=0.6, s=30)
        ax4.set_title('Tokenized REIT - Predicted Regimes', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Regime')
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Traditional Transition Matrix Heatmap
    ax5 = plt.subplot(3, 2, 5)
    im1 = ax5.imshow(trad_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax5.set_title('Learned Transition Matrix - Traditional', fontsize=14, fontweight='bold')
    ax5.set_xticks([0, 1, 2, 3])
    ax5.set_yticks([0, 1, 2, 3])
    ax5.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax5.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax5.set_xlabel('To State')
    ax5.set_ylabel('From State')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            ax5.text(j, i, f'{trad_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax5)
    
    # Plot 6: Tokenized Transition Matrix Heatmap
    ax6 = plt.subplot(3, 2, 6)
    im2 = ax6.imshow(token_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Learned Transition Matrix - Tokenized', fontsize=14, fontweight='bold')
    ax6.set_xticks([0, 1, 2, 3])
    ax6.set_yticks([0, 1, 2, 3])
    ax6.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax6.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax6.set_xlabel('To State')
    ax6.set_ylabel('From State')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            ax6.text(j, i, f'{token_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax6)
    
    plt.tight_layout()
    
    output_path = parent_dir / "outputs" / "dmm_qfclient_training.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main training pipeline with qfclient data."""
    
    # Check prerequisites
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        return
    
    if not QFCLIENT_AVAILABLE:
        print("WARNING: qfclient is not available.")
        print("The script will use synthetic data for tokenized markets.")
        print("\nTo use real data:")
        print("1. Install qfclient dependencies:")
        print("   cd qfclient-main && pip install -e .")
        print("2. Create .env file with API keys (see qfclient-main/.env.example)")
        print("3. Free options: Yahoo Finance (no key), Finnhub, FMP, FRED")
        print("\nContinuing with synthetic data...\n")
    
    # Train model (two-phase: supervised + VAE)
    model = train_model_with_qfclient_data(
        epochs=200,
        hidden_dim=64,
        learning_rate=1e-3
    )
    
    # Load data again for evaluation
    data = load_training_data_from_qfclient()
    
    # Evaluate
    evaluate_and_visualize(model, data)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Use trained model in simulations (see integrate_dmm.py)")
    print("2. Compare DMM vs HMM performance")
    print("3. Fine-tune hyperparameters for better fit")
    print("\nModel saved at: outputs/deep_markov_model_qfclient.pt")
    print("="*70)


if __name__ == "__main__":
    main()
