"""
Hybrid Approach: Use Empirical Transition Matrices with Context-Based Interpolation

Instead of training a complex neural network on limited data, this approach:
1. Uses the empirical transition matrices you already have
2. Interpolates between them based on context
3. Provides similar flexibility without the training complexity

This is more practical for small datasets and gives interpretable results.
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


class HybridMarkovModel:
    """
    Context-aware Markov model using empirical matrices with interpolation.
    
    Much simpler than DMM but provides similar functionality:
    - Context-dependent transitions
    - Smooth interpolation between traditional and tokenized
    - No training required
    - Fully interpretable
    """
    
    def __init__(self):
        self.regime_names = ['calm', 'neutral', 'volatile', 'panic']
        
        # Empirical matrices from your data (from main.py)
        self.P_traditional = np.array([
            [0.8591, 0.1389, 0.0020, 0.0000],
            [0.2339, 0.7186, 0.0475, 0.0000],
            [0.0339, 0.2203, 0.6949, 0.0508],
            [0.0000, 0.0000, 0.7500, 0.2500],
        ])
        
        self.P_tokenized = np.array([
            [0.8174, 0.1739, 0.0087, 0.0000],
            [0.1887, 0.7736, 0.0283, 0.0094],
            [0.0500, 0.2000, 0.7500, 0.0000],
            [0.0000, 0.0000, 0.1000, 0.9000],
        ])
        
        self.regime_to_idx = {name: i for i, name in enumerate(self.regime_names)}
        self.idx_to_regime = {i: name for i, name in enumerate(self.regime_names)}
    
    def predict_next_regime(self, current_regime: str, context: dict) -> tuple:
        """
        Predict next regime based on context.
        
        Args:
            current_regime: Current regime name
            context: Dictionary with keys:
                - is_tokenized: 0.0 to 1.0
                - adoption_rate: 0.0 to 1.0 (optional, uses is_tokenized if not provided)
        
        Returns:
            (predicted_regime, probabilities)
        """
        # Get interpolation weight
        if 'adoption_rate' in context:
            weight = context['adoption_rate']
        else:
            weight = context['is_tokenized']
        
        # Clamp weight to [0, 1]
        weight = max(0.0, min(1.0, weight))
        
        # Interpolate between matrices
        # P_interp = (1 - weight) * P_traditional + weight * P_tokenized
        P_interp = (1 - weight) * self.P_traditional + weight * self.P_tokenized
        
        # Get row for current regime
        regime_idx = self.regime_to_idx[current_regime]
        probs = P_interp[regime_idx]
        
        # Normalize to ensure probabilities sum to exactly 1.0
        # (floating-point arithmetic can cause small deviations)
        probs = probs / probs.sum()
        
        # Predict most likely next regime
        next_idx = np.argmax(probs)
        next_regime = self.idx_to_regime[next_idx]
        
        return next_regime, probs
    
    def infer_regimes(self, prices: np.ndarray, is_tokenized: float = 0.0, 
                     adoption_rate: np.ndarray = None) -> tuple:
        """
        Infer most likely regime sequence from price data.
        
        Uses volatility-based heuristic:
        - Calculate rolling volatility
        - Map to regime based on thresholds
        """
        if len(prices) < 2:
            return np.array([]), np.zeros((len(prices), len(self.regime_names)))
        
        # Calculate log returns
        returns = np.diff(np.log(prices))
        
        # Calculate realized volatility (absolute returns as proxy)
        vols = np.abs(returns)
        
        # Define volatility thresholds (calibrated to your data)
        if is_tokenized > 0.5:
            # Tokenized: higher volatility thresholds
            calm_thresh = 0.02
            neutral_thresh = 0.04
            volatile_thresh = 0.08
        else:
            # Traditional: lower volatility thresholds
            calm_thresh = 0.01
            neutral_thresh = 0.02
            volatile_thresh = 0.05
        
        # Map volatility to regimes
        regimes = []
        for vol in vols:
            if vol < calm_thresh:
                regimes.append('calm')
            elif vol < neutral_thresh:
                regimes.append('neutral')
            elif vol < volatile_thresh:
                regimes.append('volatile')
            else:
                regimes.append('panic')
        
        # Create probability matrix (one-hot encoding)
        probs = np.zeros((len(regimes), len(self.regime_names)))
        for i, regime in enumerate(regimes):
            regime_idx = self.regime_to_idx[regime]
            probs[i, regime_idx] = 1.0
        
        return np.array(regimes), probs
    
    def save(self, path: str):
        """Save model (just saves the matrices)."""
        np.savez(path, 
                 P_traditional=self.P_traditional,
                 P_tokenized=self.P_tokenized,
                 regime_names=self.regime_names)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model."""
        data = np.load(path, allow_pickle=True)
        self.P_traditional = data['P_traditional']
        self.P_tokenized = data['P_tokenized']
        self.regime_names = list(data['regime_names'])
        print(f"Model loaded from {path}")


def demonstrate_hybrid_model():
    """Demonstrate the hybrid model."""
    print("="*70)
    print(" HYBRID MARKOV MODEL (EMPIRICAL MATRICES + INTERPOLATION)")
    print("="*70)
    
    model = HybridMarkovModel()
    
    print("\n1. Traditional Market Transitions:")
    print("   " + "-"*60)
    context_trad = {'is_tokenized': 0.0}
    for regime in model.regime_names:
        next_regime, probs = model.predict_next_regime(regime, context_trad)
        print(f"   {regime:10s} → {next_regime:10s}  ", end="")
        print("  ".join(f"{p:.3f}" for p in probs))
    
    print("\n2. Tokenized Market Transitions:")
    print("   " + "-"*60)
    context_token = {'is_tokenized': 1.0}
    for regime in model.regime_names:
        next_regime, probs = model.predict_next_regime(regime, context_token)
        print(f"   {regime:10s} → {next_regime:10s}  ", end="")
        print("  ".join(f"{p:.3f}" for p in probs))
    
    print("\n3. Hybrid Market (50% adoption):")
    print("   " + "-"*60)
    context_hybrid = {'is_tokenized': 0.5}
    for regime in model.regime_names:
        next_regime, probs = model.predict_next_regime(regime, context_hybrid)
        print(f"   {regime:10s} → {next_regime:10s}  ", end="")
        print("  ".join(f"{p:.3f}" for p in probs))
    
    # Visualize interpolation
    visualize_interpolation(model)
    
    # Save model
    save_path = parent_dir / "outputs" / "hybrid_markov_model.npz"
    model.save(str(save_path))
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nModel saved: {save_path}")
    print("\nAdvantages of this approach:")
    print("  ✓ No training required")
    print("  ✓ Uses your proven empirical matrices")
    print("  ✓ Fully interpretable")
    print("  ✓ Smooth interpolation between contexts")
    print("  ✓ No risk of posterior collapse")
    print("\nYou can use this model exactly like the DMM!")


def visualize_interpolation(model):
    """Visualize how transitions change with adoption rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    adoption_rates = np.linspace(0, 1, 11)
    
    for regime_idx, regime in enumerate(model.regime_names):
        ax = axes[regime_idx // 2, regime_idx % 2]
        
        # Calculate transition probabilities for each adoption rate
        probs_over_time = []
        for adoption in adoption_rates:
            _, probs = model.predict_next_regime(regime, {'adoption_rate': adoption})
            probs_over_time.append(probs)
        
        probs_over_time = np.array(probs_over_time)
        
        # Plot
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        for i, target_regime in enumerate(model.regime_names):
            ax.plot(adoption_rates * 100, probs_over_time[:, i] * 100,
                   label=target_regime.capitalize(), linewidth=2.5,
                   marker='o', color=colors[i])
        
        ax.set_title(f'From: {regime.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tokenization/Adoption Rate (%)', fontsize=11)
        ax.set_ylabel('Transition Probability (%)', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    output_path = parent_dir / "outputs" / "hybrid_model_interpolation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")


if __name__ == "__main__":
    demonstrate_hybrid_model()
