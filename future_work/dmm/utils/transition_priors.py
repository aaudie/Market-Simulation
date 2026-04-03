"""
Transition Matrix Priors for Deep Markov Model

Provides theoretically-grounded initial transition matrices for traditional
and tokenized CRE markets based on financial theory and empirical observations.

These can be used to:
1. Initialize the model when training data is limited
2. Regularize learned transitions toward realistic values
3. Compare learned vs theoretical behavior
"""

import numpy as np
from typing import Dict, Tuple


def get_traditional_cre_prior() -> np.ndarray:
    """
    Get prior transition matrix for traditional CRE markets.
    
    Based on empirical observations:
    - High persistence (diagonal dominance)
    - Slow regime changes
    - Crisis states are absorbing until external intervention
    
    Returns:
        4x4 transition matrix [calm, neutral, volatile, panic]
    """
    return np.array([
        # From calm: Mostly stays calm, occasionally becomes neutral
        [0.85, 0.12, 0.02, 0.01],  # To: calm, neutral, volatile, panic
        
        # From neutral: Can go either way
        [0.15, 0.65, 0.18, 0.02],
        
        # From volatile: Hard to escape, can crash
        [0.05, 0.10, 0.70, 0.15],
        
        # From panic: Very sticky, hard to recover
        [0.02, 0.03, 0.15, 0.80],
    ])


def get_tokenized_cre_prior() -> np.ndarray:
    """
    Get prior transition matrix for tokenized CRE markets (REITs).
    
    Tokenized markets have:
    - Higher liquidity → faster regime transitions
    - More responsive to information → less persistence
    - Better crisis recovery due to liquidity
    - Still affected by fundamentals but less sticky
    
    Returns:
        4x4 transition matrix [calm, neutral, volatile, panic]
    """
    return np.array([
        # From calm: Less persistent than traditional, more responsive
        [0.70, 0.20, 0.08, 0.02],  # To: calm, neutral, volatile, panic
        
        # From neutral: More fluid transitions
        [0.20, 0.50, 0.25, 0.05],
        
        # From volatile: Better recovery than traditional
        [0.10, 0.20, 0.60, 0.10],
        
        # From panic: Faster recovery due to liquidity
        [0.05, 0.15, 0.30, 0.50],  # Less sticky than traditional
    ])


def get_tokenization_effect_prior() -> Dict[str, np.ndarray]:
    """
    Get both traditional and tokenized priors for comparison.
    
    Key differences:
    - Tokenized: Lower diagonal (less persistence)
    - Tokenized: Higher off-diagonal in recovery direction
    - Tokenized: Panic state is less absorbing
    
    Returns:
        Dictionary with 'traditional' and 'tokenized' matrices
    """
    return {
        'traditional': get_traditional_cre_prior(),
        'tokenized': get_tokenized_cre_prior()
    }


def compute_steady_state(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute steady-state distribution from transition matrix.
    
    Args:
        transition_matrix: NxN row-stochastic matrix
        
    Returns:
        Steady-state probability distribution
    """
    # Find eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    
    # Find index of eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    
    # Get corresponding eigenvector and normalize
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / steady_state.sum()
    
    return np.abs(steady_state)


def get_regime_persistence_stats(transition_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate persistence statistics from transition matrix.
    
    Args:
        transition_matrix: NxN transition matrix
        
    Returns:
        Dictionary with persistence metrics
    """
    n = len(transition_matrix)
    
    # Expected duration in each regime (1 / (1 - self_transition_prob))
    expected_duration = 1 / (1 - np.diag(transition_matrix) + 1e-10)
    
    return {
        'mean_diagonal': np.mean(np.diag(transition_matrix)),
        'mean_duration': np.mean(expected_duration),
        'max_duration': np.max(expected_duration),
        'min_duration': np.min(expected_duration)
    }


def visualize_priors():
    """Print comparison of traditional vs tokenized priors."""
    regime_names = ['Calm', 'Neutral', 'Volatile', 'Panic']
    
    print("="*70)
    print("TRANSITION MATRIX PRIORS")
    print("="*70)
    
    priors = get_tokenization_effect_prior()
    
    for market_type, matrix in priors.items():
        print(f"\n{market_type.upper()} CRE:")
        print("\nFrom \\ To    " + "  ".join(f"{name:8s}" for name in regime_names))
        print("-" * 60)
        
        for i, from_regime in enumerate(regime_names):
            row_str = f"{from_regime:10s}  "
            row_str += "  ".join(f"{matrix[i, j]:8.4f}" for j in range(len(regime_names)))
            print(row_str)
        
        # Calculate statistics
        steady = compute_steady_state(matrix)
        stats = get_regime_persistence_stats(matrix)
        
        print(f"\nSteady-state distribution:")
        for name, prob in zip(regime_names, steady):
            print(f"  {name:10s}: {prob:6.1%}")
        
        print(f"\nPersistence statistics:")
        print(f"  Mean self-transition: {stats['mean_diagonal']:.3f}")
        print(f"  Mean duration: {stats['mean_duration']:.1f} periods")
        print(f"  Range: {stats['min_duration']:.1f} - {stats['max_duration']:.1f} periods")
    
    # Compare differences
    print("\n" + "="*70)
    print("KEY DIFFERENCES")
    print("="*70)
    
    diff = priors['tokenized'] - priors['traditional']
    
    print("\nTokenized - Traditional (difference matrix):")
    print("\nFrom \\ To    " + "  ".join(f"{name:8s}" for name in regime_names))
    print("-" * 60)
    
    for i, from_regime in enumerate(regime_names):
        row_str = f"{from_regime:10s}  "
        row_str += "  ".join(f"{diff[i, j]:+8.4f}" for j in range(len(regime_names)))
        print(row_str)
    
    print("\nInterpretation:")
    print("  Negative diagonal → Tokenized less persistent (faster transitions)")
    print("  Positive off-diagonal → Tokenized more fluid (easier regime changes)")
    print("  Panic row → Tokenized recovers faster (better liquidity)")


def apply_prior_regularization(
    learned_matrix: np.ndarray,
    prior_matrix: np.ndarray,
    strength: float = 0.1
) -> np.ndarray:
    """
    Regularize learned transition matrix toward prior.
    
    Useful when training data is limited or noisy.
    
    Args:
        learned_matrix: Matrix learned from data
        prior_matrix: Prior/theoretical matrix
        strength: Regularization strength [0, 1]
                 0 = use learned only
                 1 = use prior only
                 
    Returns:
        Regularized transition matrix
    """
    regularized = (1 - strength) * learned_matrix + strength * prior_matrix
    
    # Ensure row-stochastic (rows sum to 1)
    row_sums = regularized.sum(axis=1, keepdims=True)
    regularized = regularized / row_sums
    
    return regularized


def suggest_initialization(n_tokenized_samples: int, n_traditional_samples: int) -> Dict:
    """
    Suggest whether to use prior initialization based on data availability.
    
    Args:
        n_tokenized_samples: Number of tokenized training sequences
        n_traditional_samples: Number of traditional training sequences
        
    Returns:
        Dictionary with recommendations
    """
    # Rules of thumb
    MIN_SAMPLES = 50
    GOOD_SAMPLES = 200
    
    tokenized_ratio = n_tokenized_samples / max(n_traditional_samples, 1)
    
    recommendation = {
        'use_prior': False,
        'regularization_strength': 0.0,
        'reason': '',
        'warnings': []
    }
    
    if n_tokenized_samples < MIN_SAMPLES:
        recommendation['use_prior'] = True
        recommendation['regularization_strength'] = 0.3
        recommendation['reason'] = f"Very limited tokenized data ({n_tokenized_samples} samples)"
        recommendation['warnings'].append(
            f"Only {n_tokenized_samples} tokenized samples available. "
            f"Recommend using prior initialization with regularization."
        )
    
    elif n_tokenized_samples < GOOD_SAMPLES:
        recommendation['use_prior'] = True
        recommendation['regularization_strength'] = 0.1
        recommendation['reason'] = f"Limited tokenized data ({n_tokenized_samples} samples)"
        recommendation['warnings'].append(
            f"Only {n_tokenized_samples} tokenized samples. "
            f"Consider light regularization toward prior."
        )
    
    if tokenized_ratio < 0.2:
        recommendation['warnings'].append(
            f"Severe data imbalance: {n_tokenized_samples} tokenized vs "
            f"{n_traditional_samples} traditional. Model may overfit to traditional patterns."
        )
    
    return recommendation


if __name__ == "__main__":
    visualize_priors()
    
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    
    print("\nExample 1: Check if you need prior initialization")
    print("```python")
    print("from dmm.utils.transition_priors import suggest_initialization")
    print("")
    print("recommendation = suggest_initialization(")
    print("    n_tokenized_samples=23,  # Your current situation")
    print("    n_traditional_samples=283")
    print(")")
    print("print(recommendation)")
    print("```")
    
    # Actually run it
    rec = suggest_initialization(23, 283)
    print("\nYour situation:")
    print(f"  Use prior: {rec['use_prior']}")
    print(f"  Strength: {rec['regularization_strength']}")
    print(f"  Reason: {rec['reason']}")
    for warning in rec['warnings']:
        print(f"  ⚠ {warning}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION FOR YOUR CASE")
    print("="*70)
    print("\n✓ FIRST: Fix your data loading!")
    print("  Change combine_multi_reit_data(method='average')")
    print("  to combine_multi_reit_data(method='concatenate')")
    print("  This will give you ~88 separate REIT sequences instead of 1 averaged sequence")
    print("\n✓ THEN: Re-train with more data")
    print("  You should get hundreds of tokenized sequences instead of 23")
    print("\n✓ IF STILL NEEDED: Apply prior regularization")
    print("  Use the priors in this module as backup/regularization")
