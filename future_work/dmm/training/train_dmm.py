"""
Train Deep Markov Model on Traditional and Tokenized Market Data

This script demonstrates:
1. Loading historical CRE and REIT data
2. Training the Deep Markov Model
3. Comparing learned transition matrices with empirical ones
4. Visualizing regime predictions
5. Saving trained model for use in simulations
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from dmm import DeepMarkovModel, TORCH_AVAILABLE
from sim.data_loader import load_cre_csv, load_twelvedata_api
from sim.types import HistoricalPoint


# =============================================================================
# Data Loading
# =============================================================================

def load_traditional_cre_data() -> np.ndarray:
    """
    Load traditional CRE historical data.
    
    Returns:
        Array of prices [seq_len]
    """
    print("Loading traditional CRE data...")
    csv_path = parent_dir / "data" / "cre_monthly.csv"
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Generating synthetic data.")
        # Generate synthetic data for demonstration
        n_months = 864  # 72 years
        prices = [100.0]
        for _ in range(n_months - 1):
            ret = np.random.normal(0.003, 0.018)  # Low volatility
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)
    
    history = load_cre_csv(str(csv_path))
    prices = np.array([p.price for p in history])
    
    print(f"✓ Loaded {len(prices):,} monthly data points (Traditional CRE)")
    return prices


def load_tokenized_reit_data() -> np.ndarray:
    """
    Load tokenized/REIT historical data (VNQ).
    
    Returns:
        Array of prices [seq_len]
    """
    print("Loading tokenized REIT data (VNQ)...")
    
    try:
        history = load_twelvedata_api(
            symbol="VNQ",
            interval="1month",
            start_date="2005-01-01 00:00:00",
            end_date="2026-01-20 00:00:00"
        )
        
        prices = np.array([p.price for p in history])
        print(f"✓ Loaded {len(prices):,} monthly data points (REIT/Tokenized)")
        return prices
    
    except Exception as e:
        print(f"Warning: Could not fetch VNQ data: {e}")
        print("Generating synthetic tokenized data...")
        
        # Generate synthetic high-volatility data
        n_months = 252  # ~21 years
        prices = [100.0]
        for _ in range(n_months - 1):
            ret = np.random.normal(0.008, 0.064)  # High volatility
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)


def prepare_training_data() -> Dict[str, np.ndarray]:
    """
    Prepare training data by combining traditional and tokenized sequences.
    
    Returns:
        Dictionary with prices, labels, and metadata
    """
    trad_prices = load_traditional_cre_data()
    token_prices = load_tokenized_reit_data()
    
    # Create sliding windows for better training
    # Traditional: use 72-month windows
    window_size = 72
    stride = 12
    
    trad_windows = []
    for i in range(0, len(trad_prices) - window_size, stride):
        trad_windows.append(trad_prices[i:i + window_size])
    
    token_windows = []
    for i in range(0, len(token_prices) - window_size, stride):
        token_windows.append(token_prices[i:i + window_size])
    
    # Combine into training dataset
    all_windows = trad_windows + token_windows
    is_tokenized = np.concatenate([
        np.zeros(len(trad_windows)),
        np.ones(len(token_windows))
    ])
    
    prices_array = np.array(all_windows)
    
    # Generate adoption rates (only relevant for tokenized data)
    adoption_rates = np.zeros((len(all_windows), window_size))
    for i in range(len(trad_windows), len(all_windows)):
        # Sigmoid adoption curve
        t = np.linspace(0, 1, window_size)
        adoption_rates[i] = 1 / (1 + np.exp(-10 * (t - 0.5)))
    
    print(f"\n✓ Prepared training data:")
    print(f"  - {len(trad_windows)} traditional CRE windows")
    print(f"  - {len(token_windows)} tokenized REIT windows")
    print(f"  - Window size: {window_size} months")
    print(f"  - Total sequences: {len(all_windows)}")
    
    return {
        'prices': prices_array,
        'is_tokenized': is_tokenized,
        'adoption_rates': adoption_rates,
        'window_size': window_size
    }


# =============================================================================
# Training
# =============================================================================

def train_model(
    data: Dict[str, np.ndarray],
    epochs: int = 200,
    hidden_dim: int = 128,
    learning_rate: float = 5e-4,
    save_path: str = None
) -> DeepMarkovModel:
    """
    Train Deep Markov Model on prepared data.
    
    Args:
        data: Dictionary from prepare_training_data()
        epochs: Number of training epochs
        hidden_dim: Hidden dimension for neural networks
        learning_rate: Learning rate
        save_path: Optional path to save trained model
    
    Returns:
        Trained DeepMarkovModel
    """
    print("\n" + "="*70)
    print("TRAINING DEEP MARKOV MODEL")
    print("="*70)
    
    # Initialize model
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
    history = model.train(
        prices=data['prices'],
        is_tokenized=data['is_tokenized'],
        adoption_rate=data['adoption_rates'],
        epochs=epochs,
        batch_size=16,
        beta_schedule='slow_linear',  # Slower annealing to prevent posterior collapse
        verbose=True
    )
    
    # Save model
    if save_path is None:
        save_path = str(parent_dir / "outputs" / "deep_markov_model.pt")
    
    model.save(save_path)
    
    return model


# =============================================================================
# Evaluation and Visualization
# =============================================================================

def evaluate_model(model: DeepMarkovModel, data: Dict[str, np.ndarray]) -> None:
    """
    Evaluate trained model and compare with empirical transition matrices.
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Extract traditional and tokenized sequences
    n_trad = int((data['is_tokenized'] == 0).sum())
    trad_prices = data['prices'][:n_trad]
    token_prices = data['prices'][n_trad:]
    
    # Infer regimes for a sample from each type
    print("\n1. Traditional CRE Sample Regime Inference:")
    trad_sample = trad_prices[0]
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
        token_sample = token_prices[0]
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
    
    # Compute learned transition matrices
    print("\n3. Learned Transition Matrices:")
    
    print("\n   Traditional CRE:")
    print("   From \\ To    Calm      Neutral   Volatile  Panic")
    print("   " + "-"*60)
    
    trad_context = {'is_tokenized': 0.0, 'time_normalized': 0.5, 'adoption_rate': 0.0}
    for regime in model.regime_names:
        _, probs = model.predict_next_regime(regime, trad_context)
        row_str = f"   {regime:10s}"
        for p in probs:
            row_str += f"  {p:.4f}"
        print(row_str)
    
    print("\n   Tokenized REIT:")
    print("   From \\ To    Calm      Neutral   Volatile  Panic")
    print("   " + "-"*60)
    
    token_context = {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    for regime in model.regime_names:
        _, probs = model.predict_next_regime(regime, token_context)
        row_str = f"   {regime:10s}"
        for p in probs:
            row_str += f"  {p:.4f}"
        print(row_str)
    
    # Compare with empirical matrices from main.py
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


def visualize_results(model: DeepMarkovModel, data: Dict[str, np.ndarray]) -> None:
    """
    Create visualizations of training results and regime predictions.
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training curves
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(model.training_history['loss'], label='Total Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(model.training_history['reconstruction_loss'], 
             label='Reconstruction', linewidth=2, color='#2ecc71')
    ax2.plot(model.training_history['kl_loss'], 
             label='KL Divergence', linewidth=2, color='#e74c3c')
    if 'entropy' in model.training_history and len(model.training_history['entropy']) > 0:
        ax2.plot(model.training_history['entropy'], 
                 label='Entropy', linewidth=2, color='#9b59b6', linestyle='--')
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. Regime predictions on sample sequences
    n_trad = int((data['is_tokenized'] == 0).sum())
    
    # Traditional sample
    ax3 = plt.subplot(3, 2, 3)
    trad_sample = data['prices'][0]
    trad_regimes, _ = model.infer_regimes(
        prices=trad_sample,
        is_tokenized=0.0,
        adoption_rate=np.zeros(len(trad_sample))
    )
    
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
    
    # Tokenized sample
    ax4 = plt.subplot(3, 2, 4)
    if len(data['prices']) > n_trad:
        token_sample = data['prices'][n_trad]
        token_regimes, _ = model.infer_regimes(
            prices=token_sample,
            is_tokenized=1.0,
            adoption_rate=data['adoption_rates'][n_trad]
        )
        
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
    
    # 3. Transition matrix heatmaps
    ax5 = plt.subplot(3, 2, 5)
    trad_context = {'is_tokenized': 0.0, 'time_normalized': 0.5, 'adoption_rate': 0.0}
    trad_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, trad_context)
        trad_matrix[i] = probs
    
    im1 = ax5.imshow(trad_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax5.set_title('Learned Transition Matrix - Traditional', fontsize=14, fontweight='bold')
    ax5.set_xticks([0, 1, 2, 3])
    ax5.set_yticks([0, 1, 2, 3])
    ax5.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax5.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax5.set_xlabel('To State')
    ax5.set_ylabel('From State')
    
    for i in range(4):
        for j in range(4):
            ax5.text(j, i, f'{trad_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax5)
    
    ax6 = plt.subplot(3, 2, 6)
    token_context = {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    token_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, token_context)
        token_matrix[i] = probs
    
    im2 = ax6.imshow(token_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax6.set_title('Learned Transition Matrix - Tokenized', fontsize=14, fontweight='bold')
    ax6.set_xticks([0, 1, 2, 3])
    ax6.set_yticks([0, 1, 2, 3])
    ax6.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax6.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax6.set_xlabel('To State')
    ax6.set_ylabel('From State')
    
    for i in range(4):
        for j in range(4):
            ax6.text(j, i, f'{token_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax6)
    
    plt.tight_layout()
    
    # Save figure
    output_path = parent_dir / "outputs" / "deep_markov_model_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main training pipeline."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        return
    
    print("="*70)
    print(" DEEP MARKOV MODEL TRAINING")
    print("="*70)
    
    # 1. Prepare data
    data = prepare_training_data()
    
    # 2. Train model
    model = train_model(
        data=data,
        epochs=200,
        hidden_dim=128,
        learning_rate=5e-4
    )
    
    # 3. Evaluate
    evaluate_model(model, data)
    
    # 4. Visualize
    visualize_results(model, data)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Use trained model in simulations (see integrate_dmm_simulator.py)")
    print("2. Compare DMM vs HMM performance")
    print("3. Fine-tune hyperparameters for better fit")
    print("\nModel saved at: outputs/deep_markov_model.pt")


if __name__ == "__main__":
    main()
