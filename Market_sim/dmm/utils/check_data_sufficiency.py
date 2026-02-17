"""
Check if you have enough data to train the Deep Markov Model

Run this before training to see if your dataset is sufficient.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))


def estimate_model_parameters(hidden_dim: int = 128, n_regimes: int = 4, context_dim: int = 3) -> int:
    """
    Estimate number of parameters in the DMM.
    
    Args:
        hidden_dim: Hidden dimension size
        n_regimes: Number of regimes (4: calm, neutral, volatile, panic)
        context_dim: Context feature dimension (3: is_tokenized, time, adoption)
    
    Returns:
        Total number of trainable parameters
    """
    # Transition Network: (n_regimes + context_dim) → hidden → hidden → n_regimes
    trans_input_dim = n_regimes + context_dim
    trans_params = (
        trans_input_dim * hidden_dim + hidden_dim +  # Layer 1
        hidden_dim * hidden_dim + hidden_dim +       # Layer 2
        hidden_dim * n_regimes + n_regimes           # Layer 3
    )
    
    # Emission Network: (n_regimes + context_dim) → hidden → hidden → 4 outputs
    emis_input_dim = n_regimes + context_dim
    emis_params = (
        emis_input_dim * hidden_dim + hidden_dim +
        hidden_dim * hidden_dim + hidden_dim +
        hidden_dim * 4 + 4  # Outputs: mu_return, log_sigma_return, mu_vol, log_sigma_vol
    )
    
    # Inference Network: obs_dim → hidden → hidden → n_regimes
    obs_dim = 2  # (return, volatility)
    infer_params = (
        obs_dim * hidden_dim + hidden_dim +
        hidden_dim * hidden_dim + hidden_dim +
        hidden_dim * n_regimes + n_regimes
    )
    
    total = trans_params + emis_params + infer_params
    
    return total


def check_data_sufficiency(
    n_sequences: int,
    sequence_length: int = 72,
    hidden_dim: int = 128,
    verbose: bool = True
) -> dict:
    """
    Check if you have enough data for DMM training.
    
    Args:
        n_sequences: Number of training sequences
        sequence_length: Length of each sequence (months)
        hidden_dim: Hidden dimension of neural networks
        verbose: Print detailed output
    
    Returns:
        Dictionary with assessment results
    """
    total_params = estimate_model_parameters(hidden_dim)
    total_obs = n_sequences * sequence_length
    ratio = total_obs / total_params
    
    # Determine status
    if ratio < 0.5:
        status = "CRITICAL"
        risk = "Posterior collapse almost certain"
        recommendation = f"Need at least {int(total_params * 2 / sequence_length):,} sequences"
    elif ratio < 2:
        status = "WARNING"
        risk = "High risk of posterior collapse"
        recommendation = f"Need {int(total_params * 5 / sequence_length):,} sequences for stable training"
    elif ratio < 5:
        status = "ACCEPTABLE"
        risk = "May work but could be unstable"
        recommendation = f"Ideal: {int(total_params * 10 / sequence_length):,} sequences"
    elif ratio < 10:
        status = "GOOD"
        risk = "Should train successfully"
        recommendation = "Consider more data for production use"
    else:
        status = "EXCELLENT"
        risk = "Well-resourced training expected"
        recommendation = "Dataset is sufficient for robust training"
    
    results = {
        'status': status,
        'n_sequences': n_sequences,
        'sequence_length': sequence_length,
        'total_observations': total_obs,
        'model_parameters': total_params,
        'ratio': ratio,
        'risk': risk,
        'recommendation': recommendation
    }
    
    if verbose:
        print("="*70)
        print("DATA SUFFICIENCY CHECK")
        print("="*70)
        print(f"\nDataset:")
        print(f"  Sequences:         {n_sequences:,}")
        print(f"  Sequence length:   {sequence_length} months")
        print(f"  Total observations: {total_obs:,}")
        print(f"\nModel:")
        print(f"  Hidden dimension:  {hidden_dim}")
        print(f"  Parameters:        {total_params:,}")
        print(f"\nAssessment:")
        print(f"  Data:Parameter:    {ratio:.2f}x")
        print(f"  Status:            {status}")
        print(f"  Risk:              {risk}")
        print(f"\nRecommendation:")
        print(f"  {recommendation}")
        print()
        
        # Show progress bar
        bar_length = 50
        filled = int(bar_length * min(ratio / 10, 1.0))
        bar = '█' * filled + '░' * (bar_length - filled)
        
        status_emoji = {
            'CRITICAL': '❌',
            'WARNING': '⚠️ ',
            'ACCEPTABLE': '✅',
            'GOOD': '✅',
            'EXCELLENT': '🌟'
        }
        
        print(f"{status_emoji.get(status, '')} [{bar}] {ratio:.1f}x / 10x (target)")
        print()
    
    return results


def compare_configurations():
    """Compare different model configurations."""
    print("="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)
    print()
    
    configs = [
        ("Current Setup", 83, 72, 128),
        ("With 500 sequences", 500, 72, 128),
        ("With 1,000 sequences", 1000, 72, 128),
        ("With 5,000 sequences", 5000, 72, 128),
        ("Simplified (32-dim)", 1000, 72, 32),
        ("Ideal (10,000 seqs)", 10000, 72, 128),
    ]
    
    print(f"{'Configuration':<25} {'Sequences':<12} {'Observations':<15} {'Ratio':<10} {'Status':<12}")
    print("-" * 70)
    
    for name, n_seq, seq_len, hidden in configs:
        result = check_data_sufficiency(n_seq, seq_len, hidden, verbose=False)
        
        status_symbol = {
            'CRITICAL': '❌',
            'WARNING': '⚠️ ',
            'ACCEPTABLE': '✅',
            'GOOD': '✅',
            'EXCELLENT': '🌟'
        }
        
        print(f"{name:<25} {n_seq:>10,}  {result['total_observations']:>13,}  "
              f"{result['ratio']:>8.2f}x  {status_symbol[result['status']]} {result['status']}")
    
    print()
    print("Legend:")
    print("  ❌ CRITICAL:    <0.5x ratio - Will almost certainly fail")
    print("  ⚠️  WARNING:     0.5-2x ratio - High risk of failure")
    print("  ✅ ACCEPTABLE:  2-5x ratio - May work, but risky")
    print("  ✅ GOOD:        5-10x ratio - Should work well")
    print("  🌟 EXCELLENT:   >10x ratio - Robust training expected")
    print()


def main():
    """Main assessment."""
    print("\n" + "="*70)
    print(" DEEP MARKOV MODEL - DATA REQUIREMENTS ASSESSMENT")
    print("="*70)
    print()
    
    # Check current setup
    print("1. YOUR CURRENT DATASET:")
    print("-" * 70)
    
    try:
        from dmm.training.train_dmm import prepare_training_data
        data = prepare_training_data()
        n_sequences = len(data['prices'])
        sequence_length = data['window_size']
        
        check_data_sufficiency(n_sequences, sequence_length, hidden_dim=128)
    except Exception as e:
        print(f"Could not load current data: {e}")
        print("Using default values (83 sequences):\n")
        check_data_sufficiency(83, 72, hidden_dim=128)
    
    # Show comparisons
    print("\n2. CONFIGURATION SCENARIOS:")
    print("-" * 70)
    compare_configurations()
    
    # Recommendations
    print("\n3. RECOMMENDATIONS:")
    print("-" * 70)
    print("""
To achieve different data:parameter ratios:

Option A: Collect More Real Data
  → Download 200+ REIT stocks (yfinance - FREE)
  → Access FRED economic data (FREE)
  → Purchase NCREIF data ($2,500/year)
  → Result: 500-2,000 sequences (2-5x ratio)

Option B: Use Simplified Model
  → Reduce hidden_dim from 128 to 32
  → Use data augmentation (bootstrap)
  → Run: python3 dmm/train_dmm_simple.py
  → Result: Same data, better ratio

Option C: Use Hybrid Model (RECOMMENDED)
  → No neural network → No parameter issues
  → Uses empirical matrices + interpolation
  → Run: python3 dmm/use_empirical_matrices.py
  → Result: Works immediately with proven results

Most practical: Option C (Hybrid Model)
Most ambitious: Option A + B combined
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
