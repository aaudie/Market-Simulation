#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script to inspect data quality before training
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
dmm_dir = script_dir.parent
parent_dir = dmm_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt

# Import the data loading function
from dmm.utils.qfclient_data_loader import (
    load_reit_data,
    prepare_dmm_training_data,
    generate_synthetic_reit_data,
    QFCLIENT_AVAILABLE
)
from sim.data_loader import load_cre_csv


def inspect_data():
    """Inspect the data that will be fed to the DMM."""
    
    print("="*70)
    print("DATA QUALITY INSPECTION")
    print("="*70)
    
    # 1. Load traditional CRE data
    print("\n1. Loading Traditional CRE Data...")
    csv_path = parent_dir / "data" / "cre_monthly.csv"
    
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found. Using synthetic data.")
        trad_prices = generate_synthetic_reit_data(864)
    else:
        from sim.types import HistoricalPoint
        history = load_cre_csv(str(csv_path))
        trad_prices = np.array([p.price for p in history])
    
    print(f"   Shape: {trad_prices.shape}")
    print(f"   Range: [{trad_prices.min():.2f}, {trad_prices.max():.2f}]")
    print(f"   Mean: {trad_prices.mean():.2f}, Std: {trad_prices.std():.2f}")
    print(f"   NaN count: {np.isnan(trad_prices).sum()}")
    print(f"   Inf count: {np.isinf(trad_prices).sum()}")
    
    # Calculate returns
    trad_returns = np.diff(np.log(trad_prices))
    print(f"\n   Returns - Mean: {trad_returns.mean():.6f}, Std: {trad_returns.std():.6f}")
    print(f"   Returns - Min: {trad_returns.min():.6f}, Max: {trad_returns.max():.6f}")
    
    # 2. Load tokenized data
    print("\n2. Loading Tokenized (REIT) Data...")
    
    if not QFCLIENT_AVAILABLE:
        print("WARNING: qfclient not available. Using synthetic data.")
        token_prices = generate_synthetic_reit_data(252)
    else:
        try:
            token_prices = load_reit_data("VNQ", years=20, interval="monthly")
        except Exception as e:
            print(f"WARNING: Failed to load REIT: {e}")
            token_prices = generate_synthetic_reit_data(252)
    
    print(f"   Shape: {token_prices.shape}")
    print(f"   Range: [{token_prices.min():.2f}, {token_prices.max():.2f}]")
    print(f"   Mean: {token_prices.mean():.2f}, Std: {token_prices.std():.2f}")
    print(f"   NaN count: {np.isnan(token_prices).sum()}")
    print(f"   Inf count: {np.isinf(token_prices).sum()}")
    
    # Calculate returns
    token_returns = np.diff(np.log(token_prices))
    print(f"\n   Returns - Mean: {token_returns.mean():.6f}, Std: {token_returns.std():.6f}")
    print(f"   Returns - Min: {token_returns.min():.6f}, Max: {token_returns.max():.6f}")
    
    # 3. Prepare training data
    print("\n3. Preparing Training Data...")
    training_data = prepare_dmm_training_data(
        traditional_data=trad_prices,
        tokenized_data=token_prices,
        window_size=72,
        stride=12
    )
    
    prices = training_data['prices']
    is_tokenized = training_data['is_tokenized']
    adoption_rates = training_data['adoption_rates']
    
    print(f"\n   Total sequences: {len(prices)}")
    print(f"   Traditional: {(is_tokenized == 0).sum()}")
    print(f"   Tokenized: {(is_tokenized == 1).sum()}")
    print(f"   Window size: {prices.shape[1]}")
    
    # 4. Check for data issues
    print("\n4. Checking for Data Issues...")
    
    # Check for NaN/Inf
    nan_count = np.isnan(prices).sum()
    inf_count = np.isinf(prices).sum()
    print(f"   NaN values in windows: {nan_count}")
    print(f"   Inf values in windows: {inf_count}")
    
    # Check for constant sequences
    constant_sequences = 0
    for i, seq in enumerate(prices):
        if np.std(seq) < 1e-6:
            constant_sequences += 1
            print(f"   WARNING: Sequence {i} is nearly constant: std={np.std(seq):.8f}")
    
    print(f"   Total constant sequences: {constant_sequences}")
    
    # Calculate log returns for all sequences
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=1)
    
    print(f"\n   Returns statistics across all sequences:")
    print(f"   Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    print(f"   Min: {returns.min():.6f}, Max: {returns.max():.6f}")
    print(f"   NaN count: {np.isnan(returns).sum()}")
    print(f"   Inf count: {np.isinf(returns).sum()}")
    
    # Calculate volatility
    window = 6
    volatility = np.zeros_like(returns)
    for i in range(returns.shape[1]):
        start = max(0, i - window + 1)
        window_returns = returns[:, start:i+1]
        volatility[:, i] = np.std(window_returns, axis=1)
    
    print(f"\n   Volatility statistics:")
    print(f"   Mean: {volatility.mean():.6f}, Std: {volatility.std():.6f}")
    print(f"   Min: {volatility.min():.6f}, Max: {volatility.max():.6f}")
    print(f"   NaN count: {np.isnan(volatility).sum()}")
    print(f"   Inf count: {np.isinf(volatility).sum()}")
    
    # Check if volatility is extremely high (might trigger panic regime)
    high_vol_pct = (volatility > 0.1).sum() / volatility.size * 100
    extreme_vol_pct = (volatility > 0.5).sum() / volatility.size * 100
    print(f"\n   High volatility (>0.1): {high_vol_pct:.1f}%")
    print(f"   Extreme volatility (>0.5): {extreme_vol_pct:.1f}%")
    
    # 5. Visualize data
    print("\n5. Creating Visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Price series
    axes[0, 0].plot(trad_prices, label='Traditional', alpha=0.7)
    axes[0, 0].set_title('Traditional CRE Prices')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(token_prices, label='Tokenized', alpha=0.7, color='orange')
    axes[0, 1].set_title('Tokenized (REIT) Prices')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 2: Returns distributions
    axes[0, 2].hist(trad_returns, bins=50, alpha=0.7, label='Traditional')
    axes[0, 2].hist(token_returns, bins=50, alpha=0.7, label='Tokenized')
    axes[0, 2].set_title('Returns Distribution')
    axes[0, 2].set_xlabel('Log Return')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 3: Sample windows (traditional)
    n_trad = (is_tokenized == 0).sum()
    for i in range(min(5, n_trad)):
        axes[1, 0].plot(prices[i], alpha=0.6)
    axes[1, 0].set_title('Sample Traditional Windows')
    axes[1, 0].set_xlabel('Time in Window')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Sample windows (tokenized)
    for i in range(n_trad, min(n_trad + 5, len(prices))):
        axes[1, 1].plot(prices[i], alpha=0.6)
    axes[1, 1].set_title('Sample Tokenized Windows')
    axes[1, 1].set_xlabel('Time in Window')
    axes[1, 1].set_ylabel('Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Volatility over time for all sequences
    axes[1, 2].hist(volatility.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Volatility Distribution')
    axes[1, 2].set_xlabel('Volatility')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(0.05, color='orange', linestyle='--', label='Moderate')
    axes[1, 2].axvline(0.1, color='red', linestyle='--', label='High')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = parent_dir / "outputs" / "data_quality_inspection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    plt.show()
    
    # 6. Final assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    issues = []
    
    if nan_count > 0 or inf_count > 0:
        issues.append("WARNING: Data contains NaN or Inf values!")
    
    if len(prices) < 20:
        issues.append(f"WARNING: Only {len(prices)} sequences - need at least 20-30 for training!")
    
    if constant_sequences > 0:
        issues.append(f"WARNING: {constant_sequences} sequences are nearly constant!")
    
    if extreme_vol_pct > 50:
        issues.append(f"WARNING: {extreme_vol_pct:.1f}% of data has extreme volatility (>0.5)!")
        issues.append("   This might cause the model to always predict 'panic' regime.")
    
    if returns.std() > 0.2:
        issues.append(f"WARNING: Returns std ({returns.std():.4f}) is very high!")
        issues.append("   Consider normalizing the data or using shorter windows.")
    
    if len(issues) == 0:
        print("OK: No major data quality issues detected!")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    print("="*70)
    
    return training_data


if __name__ == "__main__":
    data = inspect_data()
