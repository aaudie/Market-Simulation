"""
Simplified DMM Training - Better for Small Datasets

Uses a simpler architecture that can learn from limited data:
- Smaller hidden dimensions
- More aggressive KL annealing
- Data augmentation
- Direct supervised learning approach
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dmm import DeepMarkovModel
from dmm.training.train_dmm import prepare_training_data


def augment_data(data: dict) -> dict:
    """
    Augment training data by adding noise and small perturbations.
    This helps the model generalize better from limited data.
    """
    print("\n" + "="*70)
    print("AUGMENTING TRAINING DATA")
    print("="*70)
    
    original_prices = data['prices']
    original_is_tokenized = data['is_tokenized']
    original_adoption = data['adoption_rates']
    
    augmented_prices = [original_prices]
    augmented_tokenized = [original_is_tokenized]
    augmented_adoption = [original_adoption]
    
    # Create 3 augmented versions
    np.random.seed(42)
    for aug_idx in range(3):
        # Add small noise to prices (±2%)
        noise = np.random.normal(1.0, 0.02, original_prices.shape)
        noisy_prices = original_prices * noise
        
        augmented_prices.append(noisy_prices)
        augmented_tokenized.append(original_is_tokenized)
        augmented_adoption.append(original_adoption)
    
    # Concatenate all augmented data
    data_augmented = {
        'prices': np.concatenate(augmented_prices, axis=0),
        'is_tokenized': np.concatenate(augmented_tokenized, axis=0),
        'adoption_rates': np.concatenate(augmented_adoption, axis=0),
        'window_size': data['window_size']
    }
    
    print(f"✓ Augmented from {len(original_prices)} to {len(data_augmented['prices'])} sequences")
    return data_augmented


def train_simple_model():
    """Train with simpler architecture and better hyperparameters."""
    print("="*70)
    print(" SIMPLIFIED DMM TRAINING (OPTIMIZED FOR SMALL DATA)")
    print("="*70)
    
    # 1. Load and augment data
    data = prepare_training_data()
    data_augmented = augment_data(data)
    
    # 2. Initialize with SMALLER hidden dimension
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    model = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=32,  # Much smaller - prevents overfitting
        learning_rate=1e-3  # Slightly higher learning rate
    )
    
    print(f"\nModel Configuration (Optimized for Small Data):")
    print(f"  - Device: {model.device}")
    print(f"  - Hidden dimension: 32 (reduced from 128)")
    print(f"  - Learning rate: 0.001")
    print(f"  - Training sequences: {len(data_augmented['prices'])}")
    print(f"  - Regimes: {model.regime_names}")
    
    # 3. Train with better settings
    print(f"\nTraining for 300 epochs with aggressive KL annealing...")
    history = model.train(
        prices=data_augmented['prices'],
        is_tokenized=data_augmented['is_tokenized'],
        adoption_rate=data_augmented['adoption_rates'],
        epochs=300,  # More epochs
        batch_size=32,  # Larger batch size
        beta_schedule='cosine',  # Smoother annealing
        verbose=True
    )
    
    # 4. Save model
    save_path = str(parent_dir / "outputs" / "deep_markov_model_simple.pt")
    model.save(save_path)
    print(f"\n✓ Model saved: {save_path}")
    
    # 5. Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Check learned matrices
    print("\nLearned Transition Matrix - Traditional:")
    trad_context = {'is_tokenized': 0.0, 'time_normalized': 0.5, 'adoption_rate': 0.0}
    for regime in model.regime_names:
        _, probs = model.predict_next_regime(regime, trad_context)
        print(f"  {regime:10s}: {' '.join(f'{p:.3f}' for p in probs)}")
    
    print("\nLearned Transition Matrix - Tokenized:")
    token_context = {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    for regime in model.regime_names:
        _, probs = model.predict_next_regime(regime, token_context)
        print(f"  {regime:10s}: {' '.join(f'{p:.3f}' for p in probs)}")
    
    # 6. Visualize
    create_visualization(model, history)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved at: {save_path}")
    print("\nNext steps:")
    print("  1. python3 dmm/examples.py  # Update to load 'deep_markov_model_simple.pt'")
    print("  2. python3 dmm/integrate_dmm.py  # Update to load simple model")
    
    return model


def create_visualization(model, history):
    """Create training visualization."""
    fig = plt.figure(figsize=(14, 8))
    
    # Training curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history['loss'], linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history['reconstruction_loss'], label='Reconstruction', linewidth=2)
    ax2.plot(history['kl_loss'], label='KL Divergence', linewidth=2)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Transition matrices
    ax3 = plt.subplot(2, 2, 3)
    trad_context = {'is_tokenized': 0.0, 'time_normalized': 0.5, 'adoption_rate': 0.0}
    trad_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, trad_context)
        trad_matrix[i] = probs
    
    im1 = ax3.imshow(trad_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('Traditional Transition Matrix', fontsize=14, fontweight='bold')
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'], rotation=45)
    ax3.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{trad_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax3)
    
    ax4 = plt.subplot(2, 2, 4)
    token_context = {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.7}
    token_matrix = np.zeros((4, 4))
    for i, regime in enumerate(model.regime_names):
        _, probs = model.predict_next_regime(regime, token_context)
        token_matrix[i] = probs
    
    im2 = ax4.imshow(token_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax4.set_title('Tokenized Transition Matrix', fontsize=14, fontweight='bold')
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_xticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'], rotation=45)
    ax4.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f'{token_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax4)
    
    plt.tight_layout()
    
    output_path = parent_dir / "outputs" / "dmm_simple_training.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")


if __name__ == "__main__":
    try:
        model = train_simple_model()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
