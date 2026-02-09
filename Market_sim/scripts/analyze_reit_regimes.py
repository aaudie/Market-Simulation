"""
REIT Regime Analysis - Empirical Transition Matrix Estimation

Analyzes real REIT data to extract empirical regime transition probabilities
for use in tokenized housing market simulations.

Data sources:
    - VNQ (Vanguard Real Estate ETF): Broad REIT exposure
    - IYR (iShares US Real Estate ETF): Alternative broad REIT exposure
    - RMZ (MSCI US REIT Index): Benchmark index
"""

import warnings
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim.data_loader import load_twelvedata_api
from sim.types import HistoricalPoint

warnings.filterwarnings('ignore')


# =============================================================================
# Regime Classification
# =============================================================================

def classify_regime_from_volatility(realized_vol: float, base_vol: float) -> str:
    """
    Classify market regime based on realized volatility.
    
    Uses same thresholds as the market simulator for consistency.
    
    Args:
        realized_vol: Observed volatility
        base_vol: Baseline/calibrated volatility
        
    Returns:
        Regime label: 'calm', 'neutral', 'volatile', or 'panic'
    """
    if realized_vol < 0.7 * base_vol:
        return "calm"
    elif realized_vol < 1.2 * base_vol:
        return "neutral"
    elif realized_vol < 2.0 * base_vol:
        return "volatile"
    else:
        return "panic"


def calculate_rolling_volatility(returns: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Calculate rolling window volatility.
    
    Computes sample standard deviation over a rolling window of returns.
    Early observations with insufficient history use available data.
    
    Args:
        returns: Array of log returns
        window: Rolling window size (number of periods)
        
    Returns:
        Array of rolling volatility estimates
    """
    n = len(returns)
    vols = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window + 1)
        window_returns = returns[start:i+1]
        
        if len(window_returns) >= 2:
            vols[i] = np.std(window_returns, ddof=1)
        else:
            vols[i] = 0.0
    
    return vols


# =============================================================================
# Transition Matrix Estimation
# =============================================================================

def estimate_transition_matrix(regimes: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Estimate Markov transition matrix from observed regime sequence.
    
    Counts transitions between regimes and normalizes to get probabilities.
    Also computes regime statistics (percentages, durations).
    
    Args:
        regimes: Sequence of regime labels (chronologically ordered)
        
    Returns:
        tuple: (transition_matrix, statistics_dict)
            - transition_matrix: 4x4 array of transition probabilities
            - statistics: Dict with regime counts, percentages, durations
    """
    states = ["calm", "neutral", "volatile", "panic"]
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    # Count transitions between consecutive regimes
    transition_counts = np.zeros((n_states, n_states))
    
    for i in range(len(regimes) - 1):
        current_state = regimes[i]
        next_state = regimes[i + 1]
        
        if current_state in state_to_idx and next_state in state_to_idx:
            from_idx = state_to_idx[current_state]
            to_idx = state_to_idx[next_state]
            transition_counts[from_idx, to_idx] += 1
    
    # Normalize to get transition probabilities
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = transition_counts[i, :].sum()
        if row_sum > 0:
            transition_matrix[i, :] = transition_counts[i, :] / row_sum
        else:
            # No observations: assume persistence in same state
            transition_matrix[i, i] = 1.0
    
    # Calculate regime statistics
    regime_counts = {s: regimes.count(s) for s in states}
    total = len(regimes)
    regime_percentages = {
        s: 100.0 * count / total 
        for s, count in regime_counts.items()
    }
    
    # Calculate average durations (run lengths)
    durations = {s: [] for s in states}
    current_regime = regimes[0]
    duration = 1
    
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            duration += 1
        else:
            durations[current_regime].append(duration)
            current_regime = regimes[i]
            duration = 1
    durations[current_regime].append(duration)
    
    avg_durations = {s: np.mean(d) if d else 0.0 for s, d in durations.items()}
    
    statistics = {
        "regime_counts": regime_counts,
        "regime_percentages": regime_percentages,
        "avg_durations": avg_durations,
        "transition_counts": transition_counts,
    }
    
    return transition_matrix, statistics


# =============================================================================
# Data Fetching and Processing
# =============================================================================

def fetch_reit_data(
    symbol: str = "VNQ",
    start_date: str = "2010-01-01 00:00:00",
    end_date: str = "2026-01-01 00:00:00",
) -> pd.DataFrame:
    """
    Fetch REIT historical data from Twelve Data API.
    
    Args:
        symbol: REIT ticker symbol (e.g., 'VNQ', 'IYR')
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        DataFrame with date and price columns, or None if fetch fails
    """
    print(f"\nFetching {symbol} data from Twelve Data API...")
    
    try:
        history = load_twelvedata_api(
            symbol=symbol,
            interval="1month",
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✓ Loaded {len(history):,} monthly data points")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {"date": p.date, "price": p.price}
            for p in history
        ])
        
        return df
    
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None


def analyze_reit_regimes(
    df: pd.DataFrame, 
    window: int = 6
) -> Tuple[List[str], float, pd.DataFrame]:
    """
    Analyze REIT data to identify regime sequence.
    
    Calculates rolling volatility and classifies each period into a regime
    based on volatility thresholds.
    
    Args:
        df: DataFrame with 'date' and 'price' columns
        window: Rolling window size for volatility calculation (months)
        
    Returns:
        tuple: (regime_list, baseline_volatility, processed_dataframe)
    """
    # Calculate log returns
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    returns = df['log_return'].values
    
    # Calculate baseline volatility (full sample)
    base_vol = np.std(returns, ddof=1)
    
    # Calculate rolling volatility
    rolling_vols = calculate_rolling_volatility(returns, window=window)
    
    # Classify each period into a regime
    regimes = [
        classify_regime_from_volatility(vol, base_vol) 
        for vol in rolling_vols
    ]
    
    # Add features to dataframe
    df['rolling_vol'] = rolling_vols
    df['regime'] = regimes
    
    return regimes, base_vol, df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_reit_analysis(df: pd.DataFrame, symbol: str, statistics: Dict):
    """
    Create comprehensive visualization of REIT regime analysis.
    """
    fig = plt.figure(figsize=(16, 10))
    
    regime_colors = {
        "calm": "#2ecc71",
        "neutral": "#3498db",
        "volatile": "#f39c12",
        "panic": "#e74c3c"
    }
    
    # Plot 1: Price history
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df['date'], df['price'], linewidth=2, color='#2c3e50')
    ax1.set_title(f'{symbol} Price History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling volatility
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df['date'], df['rolling_vol'] * 100, linewidth=2, color='#e74c3c')
    ax2.set_title('Rolling Volatility (Monthly)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime timeline
    ax3 = plt.subplot(3, 2, 3)
    regime_numeric = [["calm", "neutral", "volatile", "panic"].index(r) for r in df['regime']]
    colors = [regime_colors[r] for r in df['regime']]
    ax3.scatter(df['date'], regime_numeric, c=colors, alpha=0.6, s=20)
    ax3.set_title('Regime Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Regime')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Calm', 'Neutral', 'Volatile', 'Panic'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regime distribution
    ax4 = plt.subplot(3, 2, 4)
    regimes = ["calm", "neutral", "volatile", "panic"]
    percentages = [statistics['regime_percentages'][r] for r in regimes]
    colors_bar = [regime_colors[r] for r in regimes]
    
    bars = ax4.bar(range(len(regimes)), percentages, color=colors_bar, alpha=0.8)
    ax4.set_title('Time in Each Regime', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xticks(range(len(regimes)))
    ax4.set_xticklabels([r.capitalize() for r in regimes])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 5: Transition matrix heatmap
    ax5 = plt.subplot(3, 2, 5)
    transition_matrix = statistics['transition_counts'] / statistics['transition_counts'].sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix)
    
    im = ax5.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax5.set_title('Empirical Transition Matrix', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    ax5.set_xticklabels([r.capitalize() for r in regimes])
    ax5.set_yticklabels([r.capitalize() for r in regimes])
    ax5.set_xlabel('To State')
    ax5.set_ylabel('From State')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax5.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Average durations
    ax6 = plt.subplot(3, 2, 6)
    durations = [statistics['avg_durations'][r] for r in regimes]
    bars = ax6.bar(range(len(regimes)), durations, color=colors_bar, alpha=0.8)
    ax6.set_title('Average Regime Duration (Months)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Duration (Months)')
    ax6.set_xticks(range(len(regimes)))
    ax6.set_xticklabels([r.capitalize() for r in regimes])
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, dur in zip(bars, durations):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{dur:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_regime_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {symbol}_regime_analysis.png")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main analysis function."""
    print("="*70)
    print(" REIT REGIME ANALYSIS - Empirical Transition Matrix")
    print("="*70)
    
    # Symbols to analyze (can add more: "IYR", "XLRE", etc.)
    symbols = ["VNQ"]
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f" Analyzing {symbol}")
        print(f"{'='*70}")
        
        # =================================================================
        # Step 1: Fetch Data
        # =================================================================
        df = fetch_reit_data(
            symbol=symbol,
            start_date="2005-01-01 00:00:00",
            end_date="2026-01-20 00:00:00"
        )
        
        if df is None or len(df) < 12:
            print(f"✗ Insufficient data for {symbol}")
            continue
        
        # =================================================================
        # Step 2: Analyze Regimes
        # =================================================================
        print(f"\nAnalyzing regime transitions...")
        regimes, base_vol, df_analyzed = analyze_reit_regimes(df, window=6)
        
        print(f"✓ Identified {len(regimes):,} regime observations")
        print(f"✓ Baseline volatility: {base_vol*100:.2f}% (monthly)")
        print(f"✓ Annualized volatility: {base_vol*np.sqrt(12)*100:.2f}%")
        
        # =================================================================
        # Step 3: Estimate Transition Matrix
        # =================================================================
        print(f"\nEstimating transition matrix...")
        transition_matrix, statistics = estimate_transition_matrix(regimes)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"EMPIRICAL TRANSITION MATRIX - {symbol}")
        print(f"{'='*70}")
        print("\nTransition Probabilities:")
        print("From \\ To    Calm      Neutral   Volatile  Panic")
        print("-" * 70)
        states = ["calm", "neutral", "volatile", "panic"]
        for i, from_state in enumerate(states):
            row_str = f"{from_state:10s}"
            for j in range(4):
                row_str += f"  {transition_matrix[i, j]:.4f}"
            print(row_str)
        
        print(f"\n{'='*70}")
        print("REGIME STATISTICS")
        print(f"{'='*70}")
        for state in states:
            pct = statistics['regime_percentages'][state]
            dur = statistics['avg_durations'][state]
            print(f"{state.capitalize():10s}: {pct:5.1f}% of time, avg duration {dur:.1f} months")
        
        # Format for Python code
        print(f"\n{'='*70}")
        print("PYTHON CODE FORMAT (for housing_liquidity_comparison.py)")
        print(f"{'='*70}")
        print(f"\nP_REIT_{symbol} = [")
        for i in range(4):
            row = [transition_matrix[i, j] for j in range(4)]
            print(f"    {row},  # {states[i]}")
        print("]")
        
        # Store results
        all_results[symbol] = {
            "transition_matrix": transition_matrix,
            "statistics": statistics,
            "df": df_analyzed,
            "base_vol": base_vol
        }
        
        # Visualize
        plot_reit_analysis(df_analyzed, symbol, statistics)
    
    # Compare with traditional housing
    print(f"\n{'='*70}")
    print("COMPARISON: REIT vs Traditional Housing Assumptions")
    print(f"{'='*70}")
    
    print("\nTraditional Housing (Assumed):")
    P_trad = [
        [0.85, 0.14, 0.01, 0.00],
        [0.10, 0.75, 0.14, 0.01],
        [0.02, 0.18, 0.70, 0.10],
        [0.01, 0.09, 0.30, 0.60],
    ]
    states = ["calm", "neutral", "volatile", "panic"]
    print("From \\ To    Calm      Neutral   Volatile  Panic")
    print("-" * 70)
    for i, state in enumerate(states):
        row_str = f"{state:10s}"
        for j in range(4):
            row_str += f"  {P_trad[i][j]:.4f}"
        print(row_str)
    
    if "VNQ" in all_results:
        print("\nREIT (Empirical from VNQ):")
        P_reit = all_results["VNQ"]["transition_matrix"]
        print("From \\ To    Calm      Neutral   Volatile  Panic")
        print("-" * 70)
        for i, state in enumerate(states):
            row_str = f"{state:10s}"
            for j in range(4):
                row_str += f"  {P_reit[i, j]:.4f}"
            print(row_str)
        
        print("\nKey Differences:")
        print(f"  • Panic persistence: Traditional {P_trad[3][3]:.2f} vs REIT {P_reit[3, 3]:.2f}")
        print(f"  • Panic to Neutral: Traditional {P_trad[3][1]:.2f} vs REIT {P_reit[3, 1]:.2f}")
        print(f"  • Volatile persistence: Traditional {P_trad[2][2]:.2f} vs REIT {P_reit[2, 2]:.2f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
