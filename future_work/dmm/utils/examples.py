"""
Minimal Example: Using Deep Markov Model

This is the simplest possible example of using the DMM.
Run this after training the model with train_dmm.py
"""

import sys
from pathlib import Path

# Add parent to path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

import numpy as np
from dmm import DeepMarkovModel

def example_1_load_and_predict():
    """Load trained model and make predictions."""
    print("="*60)
    print("EXAMPLE 1: Load Model and Predict Next Regime")
    print("="*60)
    
    # Load trained model
    model_path = parent / "outputs" / "deep_markov_model.pt"
    
    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Please run: python3 scripts/train_deep_markov_model.py")
        return
    
    dmm = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=128  # Must match training configuration
    )
    dmm.load(str(model_path))
    print(f"✓ Loaded model from {model_path}\n")
    
    # Predict next regime for traditional market
    print("Traditional Market (calm regime):")
    next_regime, probs = dmm.predict_next_regime(
        current_regime='calm',
        context={
            'is_tokenized': 0.0,
            'time_normalized': 0.5,
            'adoption_rate': 0.0
        }
    )
    
    print(f"  Current: calm")
    print(f"  Predicted next: {next_regime}")
    print(f"  Probabilities:")
    for regime, prob in zip(dmm.regime_names, probs):
        print(f"    - {regime:10s}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Predict next regime for tokenized market
    print("\nTokenized Market (volatile regime):")
    next_regime, probs = dmm.predict_next_regime(
        current_regime='volatile',
        context={
            'is_tokenized': 1.0,
            'time_normalized': 0.7,
            'adoption_rate': 0.8
        }
    )
    
    print(f"  Current: volatile")
    print(f"  Predicted next: {next_regime}")
    print(f"  Probabilities:")
    for regime, prob in zip(dmm.regime_names, probs):
        print(f"    - {regime:10s}: {prob:.4f} ({prob*100:.1f}%)")


def example_2_infer_regimes():
    """Infer regimes from synthetic price data."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Infer Regimes from Price Series")
    print("="*60)
    
    # Load model
    model_path = parent / "outputs" / "deep_markov_model.pt"
    if not model_path.exists():
        print("ERROR: Model not found. Please train first.")
        return
    
    dmm = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=128  # Must match training configuration
    )
    dmm.load(str(model_path))
    
    # Generate synthetic price series
    print("\nGenerating synthetic price series...")
    
    # Scenario 1: Low volatility (traditional)
    np.random.seed(42)
    prices_calm = [100.0]
    for _ in range(23):
        ret = np.random.normal(0.003, 0.015)  # Low vol
        prices_calm.append(prices_calm[-1] * np.exp(ret))
    
    # Scenario 2: High volatility (tokenized panic)
    prices_volatile = [100.0]
    for _ in range(23):
        ret = np.random.normal(0.005, 0.08)  # High vol
        prices_volatile.append(prices_volatile[-1] * np.exp(ret))
    
    # Infer regimes - Low volatility
    print("\nLow Volatility Series (Traditional):")
    regimes_calm, _ = dmm.infer_regimes(
        prices=np.array(prices_calm),
        is_tokenized=0.0
    )
    
    print(f"  Price path: {prices_calm[0]:.1f} → {prices_calm[-1]:.1f}")
    print(f"  Inferred regimes: {list(regimes_calm)}")
    print(f"  Distribution:")
    for regime in dmm.regime_names:
        count = (regimes_calm == regime).sum()
        pct = 100 * count / len(regimes_calm)
        print(f"    - {regime:10s}: {pct:5.1f}%")
    
    # Infer regimes - High volatility
    print("\nHigh Volatility Series (Tokenized):")
    regimes_volatile, _ = dmm.infer_regimes(
        prices=np.array(prices_volatile),
        is_tokenized=1.0
    )
    
    print(f"  Price path: {prices_volatile[0]:.1f} → {prices_volatile[-1]:.1f}")
    print(f"  Inferred regimes: {list(regimes_volatile)}")
    print(f"  Distribution:")
    for regime in dmm.regime_names:
        count = (regimes_volatile == regime).sum()
        pct = 100 * count / len(regimes_volatile)
        print(f"    - {regime:10s}: {pct:5.1f}%")


def example_3_monte_carlo_simulation():
    """Run simple Monte Carlo simulation using DMM."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Monte Carlo Simulation with DMM")
    print("="*60)
    
    # Load model
    model_path = parent / "outputs" / "deep_markov_model.pt"
    if not model_path.exists():
        print("ERROR: Model not found. Please train first.")
        return
    
    dmm = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=128  # Must match training configuration
    )
    dmm.load(str(model_path))
    
    print("\nSimulating 12-month regime path (tokenized market)...")
    
    n_simulations = 100
    regime_counts = {regime: 0 for regime in dmm.regime_names}
    
    for sim in range(n_simulations):
        current_regime = 'neutral'  # Start neutral
        
        for month in range(12):
            # Context evolves over time
            context = {
                'is_tokenized': 1.0,
                'time_normalized': month / 12.0,
                'adoption_rate': month / 12.0  # Linear adoption
            }
            
            # Predict and sample next regime
            _, probs = dmm.predict_next_regime(current_regime, context)
            current_regime = np.random.choice(dmm.regime_names, p=probs)
            
            # Count final regime
            if month == 11:  # Last month
                regime_counts[current_regime] += 1
    
    print(f"\nFinal Regime Distribution (after 12 months, {n_simulations} simulations):")
    for regime, count in regime_counts.items():
        pct = 100 * count / n_simulations
        bar = '█' * int(pct / 2)
        print(f"  {regime:10s}: {pct:5.1f}% {bar}")
    
    print("\nInterpretation:")
    print("  - This shows where regimes tend to end after 12 months")
    print("  - Starting from 'neutral', markets evolve based on learned dynamics")
    print("  - Compare with fixed transition matrix predictions!")


def example_4_context_sensitivity():
    """Demonstrate how context affects predictions."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Context Sensitivity")
    print("="*60)
    
    # Load model
    model_path = parent / "outputs" / "deep_markov_model.pt"
    if not model_path.exists():
        print("ERROR: Model not found. Please train first.")
        return
    
    dmm = DeepMarkovModel(
        regime_names=['calm', 'neutral', 'volatile', 'panic'],
        context_dim=3,
        hidden_dim=128  # Must match training configuration
    )
    dmm.load(str(model_path))
    
    print("\nHow does context affect regime transitions?")
    print("Starting from 'calm' regime:\n")
    
    contexts = [
        ("Traditional, early", {'is_tokenized': 0.0, 'time_normalized': 0.1, 'adoption_rate': 0.0}),
        ("Traditional, late", {'is_tokenized': 0.0, 'time_normalized': 0.9, 'adoption_rate': 0.0}),
        ("Tokenized, low adoption", {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.2}),
        ("Tokenized, high adoption", {'is_tokenized': 1.0, 'time_normalized': 0.5, 'adoption_rate': 0.9}),
    ]
    
    for label, context in contexts:
        _, probs = dmm.predict_next_regime('calm', context)
        print(f"{label:25s}:")
        
        # Find most likely next regime
        max_idx = np.argmax(probs)
        most_likely = dmm.regime_names[max_idx]
        
        for regime, prob in zip(dmm.regime_names, probs):
            marker = " ←" if regime == most_likely else ""
            bar = '▓' * int(prob * 30)
            print(f"  → {regime:10s}: {bar:30s} {prob:.3f}{marker}")
        print()
    
    print("Insights:")
    print("  - Different contexts → different transition probabilities")
    print("  - This is what makes DMM more flexible than fixed matrices!")


def main():
    """Run all examples."""
    try:
        example_1_load_and_predict()
        example_2_infer_regimes()
        example_3_monte_carlo_simulation()
        example_4_context_sensitivity()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Integrate DMM into your market simulator")
        print("  2. Compare with fixed transition matrices")
        print("  3. Fine-tune hyperparameters for your use case")
        print("\nSee DMM_README.md for more details!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure you've trained the model first:")
        print("  python3 scripts/train_deep_markov_model.py")


if __name__ == "__main__":
    main()
