"""
qfclient Data Loader for Deep Markov Model

This module provides functions to load real-time and historical data
from various financial APIs using qfclient for DMM training and inference.

Usage:
    from dmm.utils.qfclient_data_loader import load_reit_data, load_cre_alternatives
    
    # Load REIT data (tokenized proxy)
    reit_prices = load_reit_data(symbol="VNQ", years=5)
    
    # Load multiple REITs
    multi_reit = load_multi_reit_data(["VNQ", "IYR", "SCHH"], years=3)
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# Add qfclient to path
qfclient_path = Path(__file__).resolve().parent.parent.parent / "qfclient-main" / "src"
if str(qfclient_path) not in sys.path:
    sys.path.insert(0, str(qfclient_path))

try:
    from qfclient import MarketClient, Interval
    QFCLIENT_AVAILABLE = True
except ImportError:
    QFCLIENT_AVAILABLE = False
    print("Warning: qfclient not available. Install with: pip install -e qfclient-main/")


def load_reit_data(
    symbol: str = "VNQ",
    years: int = 5,
    interval: str = "monthly"
) -> np.ndarray:
    """
    Load REIT data as a proxy for tokenized real estate.
    
    REITs (Real Estate Investment Trusts) are liquid, publicly traded
    securities that represent a good proxy for what tokenized real estate
    markets might look like.
    
    Args:
        symbol: REIT ticker symbol (default: VNQ = Vanguard Real Estate ETF)
        years: Number of years of historical data
        interval: Data interval - "daily" or "monthly"
        
    Returns:
        Array of closing prices
        
    Recommended REITs for analysis:
        - VNQ: Vanguard Real Estate ETF (broad market)
        - IYR: iShares U.S. Real Estate ETF (liquid, large cap)
        - SCHH: Schwab U.S. REIT ETF (low cost)
        - RWR: SPDR Dow Jones REIT ETF
        - USRT: iShares Core U.S. REIT ETF
    """
    if not QFCLIENT_AVAILABLE:
        raise RuntimeError("qfclient is not available. Please install it first.")
    
    # Suppress verbose output for individual loads
    # print(f"Loading {symbol} data via qfclient...")
    
    client = MarketClient()
    
    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)
    
    # Set interval
    interval_map = {
        "daily": Interval.DAY_1,
        "monthly": Interval.MONTH_1
    }
    
    if interval not in interval_map:
        raise ValueError(f"Interval must be 'daily' or 'monthly', got: {interval}")
    
    try:
        # Fetch OHLCV data with appropriate limit
        # For monthly data: 20 years = 240 months, use limit=500 to be safe
        # For daily data: 5 years = 1260 days, use limit=2000
        limit = 500 if interval == "monthly" else min(years * 365, 2000)
        
        candles = client.get_ohlcv(
            symbol=symbol,
            interval=interval_map[interval],
            start=start_date,
            end=end_date,
            limit=limit  # Reasonable limit to avoid provider errors
        )
        
        # Convert to DataFrame
        df = candles.to_df()
        
        # Extract closing prices
        prices = df['close'].values
        
        # Suppressed output - only show in verbose mode
        # print(f"✓ Loaded {len(prices)} {interval} data points for {symbol}")
        # print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        # print(f"✓ Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        
        return prices
        
    except Exception as e:
        # Re-raise the exception so calling function can handle it
        raise RuntimeError(f"Failed to load {symbol}: {str(e)}")


def load_multi_reit_data(
    symbols: List[str],
    years: int = 5,
    interval: str = "monthly",
    max_failures: int = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load multiple REIT datasets for ensemble training.
    
    Using multiple REITs can help the DMM learn more robust patterns
    of tokenized real estate behavior.
    
    Args:
        symbols: List of REIT ticker symbols
        years: Number of years of historical data
        interval: Data interval - "daily" or "monthly"
        max_failures: Maximum number of failures before stopping (None = no limit)
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping symbol to price array
        
    Example:
        data = load_multi_reit_data(["VNQ", "IYR", "SCHH"], years=3)
        for symbol, prices in data.items():
            print(f"{symbol}: {len(prices)} data points")
    """
    results = {}
    failures = 0
    
    if verbose:
        print(f"Loading {len(symbols)} REITs...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            prices = load_reit_data(symbol, years, interval)
            results[symbol] = prices
            
            if verbose and i % 10 == 0:
                print(f"  Progress: {i}/{len(symbols)} REITs processed, {len(results)} successful")
                
        except Exception as e:
            failures += 1
            if verbose:
                print(f"  ✗ Failed to load {symbol}: {str(e)[:50]}")
            
            # Stop if too many failures
            if max_failures and failures >= max_failures:
                if verbose:
                    print(f"  ⚠ Reached max failures ({max_failures}), stopping early")
                break
            continue
    
    if verbose:
        success_rate = 100 * len(results) / len(symbols) if symbols else 0
        print(f"\n✓ Successfully loaded {len(results)}/{len(symbols)} REITs ({success_rate:.1f}%)")
    
    return results


def load_cre_alternatives(
    years: int = 3
) -> Dict[str, np.ndarray]:
    """
    Load alternative CRE-related securities as additional training data.
    
    Includes:
    - Commercial REITs
    - Residential REITs
    - Real estate developers
    - Mortgage REITs
    
    Args:
        years: Number of years of historical data
        
    Returns:
        Dictionary mapping category to price arrays
    """
    symbols = {
        "broad_market": ["VNQ", "IYR"],
        "commercial": ["SPG", "PLD", "PSA"],  # Simon Property, Prologis, Public Storage
        "residential": ["EQR", "AVB", "MAA"],  # Equity Residential, AvalonBay, Mid-America
        "office": ["BXP", "VNO"],  # Boston Properties, Vornado
        "industrial": ["PLD", "DRE"],  # Prologis, Duke Realty
        "retail": ["SPG", "REG"],  # Simon Property, Regency Centers
    }
    
    results = {}
    
    for category, tickers in symbols.items():
        print(f"\nLoading {category} REITs...")
        category_data = []
        
        for symbol in tickers:
            try:
                prices = load_reit_data(symbol, years, "monthly")
                category_data.append(prices)
            except Exception as e:
                print(f"  Skipped {symbol}: {e}")
                continue
        
        if category_data:
            # Stack all prices for this category
            results[category] = np.array(category_data)
            print(f"✓ Loaded {len(category_data)} series for {category}")
    
    return results


def load_economic_indicators(
    years: int = 10
) -> Dict[str, np.ndarray]:
    """
    Load economic indicators relevant to real estate markets.
    
    These can be used as context features for the DMM to condition
    regime transitions on macroeconomic conditions.
    
    Args:
        years: Number of years of historical data
        
    Returns:
        Dictionary mapping indicator name to values
    """
    if not QFCLIENT_AVAILABLE:
        raise RuntimeError("qfclient is not available.")
    
    client = MarketClient()
    
    indicators = {
        "fed_funds_rate": "FEDFUNDS",      # Federal Funds Rate
        "mortgage_30y": "MORTGAGE30US",    # 30-Year Fixed Rate Mortgage
        "housing_starts": "HOUST",         # Housing Starts
        "unemployment": "UNRATE",          # Unemployment Rate
        "cpi": "CPIAUCSL",                 # Consumer Price Index
        "gdp": "GDP",                      # Gross Domestic Product
    }
    
    results = {}
    
    # Calculate date range
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"\nLoading economic indicators...")
    
    for name, series_id in indicators.items():
        try:
            data = client.get_economic_indicator(
                series_id=series_id,
                start=start_date,
                end=end_date
            )
            
            df = data.to_df()
            values = df['value'].values
            
            results[name] = values
            print(f"✓ {name}: {len(values)} observations")
            
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue
    
    return results


def generate_synthetic_reit_data(n_points: int) -> np.ndarray:
    """
    Generate synthetic REIT-like data for testing/fallback.
    
    Simulates higher volatility and more regime switching than
    traditional CRE.
    
    Args:
        n_points: Number of data points to generate
        
    Returns:
        Array of synthetic prices
    """
    print("Generating synthetic REIT-like data...")
    
    np.random.seed(42)
    
    # Higher volatility parameters for REITs
    prices = [100.0]
    
    for _ in range(n_points - 1):
        # Mean reversion with occasional jumps
        ret = np.random.normal(0.008, 0.064)  # ~10% annual return, 22% vol
        
        # Add occasional regime switches (crisis events)
        if np.random.random() < 0.02:  # 2% chance per period
            ret += np.random.normal(-0.15, 0.05)  # Sudden drop
        
        prices.append(prices[-1] * np.exp(ret))
    
    return np.array(prices)


def combine_multi_reit_data(
    reit_data: Dict[str, np.ndarray],
    method: str = "average",
    min_length: int = None
) -> np.ndarray:
    """
    Combine multiple REIT price series into a single series.
    
    Args:
        reit_data: Dictionary mapping symbol to price array
        method: Combination method:
                - "average": Equal-weighted average of all REITs
                - "longest": Use the longest available series
                - "median": Median price across all REITs
                - "concatenate": Stack all series (creates more training data)
        min_length: Minimum length required (shorter series are excluded)
        
    Returns:
        Combined price array or list of arrays (for concatenate method)
    """
    if not reit_data:
        raise ValueError("No REIT data provided")
    
    # Filter by minimum length if specified
    if min_length:
        reit_data = {
            symbol: prices 
            for symbol, prices in reit_data.items() 
            if len(prices) >= min_length
        }
        
        if not reit_data:
            raise ValueError(f"No REITs meet minimum length requirement of {min_length}")
    
    if method == "longest":
        # Return the longest series
        return max(reit_data.values(), key=len)
    
    elif method == "concatenate":
        # Stack all series (useful for creating more training sequences)
        return list(reit_data.values())
    
    elif method in ["average", "median"]:
        # Find common length (use minimum to avoid NaNs)
        min_len = min(len(prices) for prices in reit_data.values())
        
        # Truncate all series to common length
        aligned_data = np.array([prices[:min_len] for prices in reit_data.values()])
        
        if method == "average":
            return np.mean(aligned_data, axis=0)
        else:  # median
            return np.median(aligned_data, axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average', 'longest', 'median', or 'concatenate'")


def load_reit_portfolio(
    symbols: List[str] = None,
    years: int = 5,
    interval: str = "monthly",
    combination_method: str = "average",
    max_failures: int = 10,
    verbose: bool = True
) -> np.ndarray:
    """
    Load and combine multiple REITs into a portfolio for training.
    
    This is a convenience function that combines load_multi_reit_data
    and combine_multi_reit_data.
    
    Args:
        symbols: List of REIT symbols (if None, uses recommended defaults)
        years: Number of years of historical data
        interval: "daily" or "monthly"
        combination_method: How to combine multiple REITs
        max_failures: Maximum failures before stopping
        verbose: Print progress
        
    Returns:
        Combined price array
    """
    # Use default symbols if none provided
    if symbols is None:
        symbols = ["VNQ", "IYR", "SCHH", "USRT", "RWR"]  # Top 5 ETFs
        if verbose:
            print("Using default REIT ETF portfolio")
    
    # Load all REITs
    reit_data = load_multi_reit_data(
        symbols=symbols,
        years=years,
        interval=interval,
        max_failures=max_failures,
        verbose=verbose
    )
    
    if not reit_data:
        raise ValueError("Failed to load any REIT data")
    
    # Combine them
    combined = combine_multi_reit_data(reit_data, method=combination_method)
    
    if verbose:
        if isinstance(combined, list):
            print(f"\n✓ Created {len(combined)} separate REIT series")
        else:
            print(f"\n✓ Combined into single series: {len(combined)} data points")
    
    return combined


def prepare_dmm_training_data(
    traditional_data: np.ndarray,
    tokenized_data,  # Can be np.ndarray or list of np.ndarray
    window_size: int = 72,
    stride: int = 12
) -> Dict[str, np.ndarray]:
    """
    Prepare data in the format expected by DMM training.
    
    Creates sliding windows and labels for traditional vs tokenized data.
    
    Args:
        traditional_data: Array of traditional CRE prices
        tokenized_data: Array of tokenized/REIT prices OR list of arrays (from concatenate)
        window_size: Size of sliding window (months)
        stride: Stride between windows
        
    Returns:
        Dictionary with training data arrays
    """
    print("\nPreparing DMM training data...")
    
    # Create windows for traditional data
    trad_windows = []
    for i in range(0, len(traditional_data) - window_size, stride):
        trad_windows.append(traditional_data[i:i + window_size])
    
    # Create windows for tokenized data
    # Handle both single array and list of arrays (from concatenate method)
    token_windows = []
    
    if isinstance(tokenized_data, list):
        # Concatenate method: create windows from each REIT separately
        print(f"  Processing {len(tokenized_data)} separate REIT series...")
        for reit_prices in tokenized_data:
            for i in range(0, len(reit_prices) - window_size, stride):
                token_windows.append(reit_prices[i:i + window_size])
    else:
        # Average/median/longest method: single array
        for i in range(0, len(tokenized_data) - window_size, stride):
            token_windows.append(tokenized_data[i:i + window_size])
    
    # Combine and label
    all_windows = trad_windows + token_windows
    is_tokenized = np.concatenate([
        np.zeros(len(trad_windows)),
        np.ones(len(token_windows))
    ])
    
    # Generate adoption rates (linear ramp for tokenized data)
    adoption_rates = np.zeros((len(all_windows), window_size))
    for i in range(len(trad_windows), len(all_windows)):
        t = np.linspace(0, 1, window_size)
        adoption_rates[i] = 1 / (1 + np.exp(-10 * (t - 0.5)))  # Sigmoid
    
    prices_array = np.array(all_windows)
    
    print(f"✓ Created {len(trad_windows)} traditional windows")
    print(f"✓ Created {len(token_windows)} tokenized windows")
    print(f"✓ Window size: {window_size} months, Stride: {stride} months")
    
    return {
        'prices': prices_array,
        'is_tokenized': is_tokenized,
        'adoption_rates': adoption_rates,
        'window_size': window_size
    }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("qfclient Data Loader for DMM")
    print("="*70)
    
    # Example 1: Load single REIT
    print("\nExample 1: Loading VNQ (Vanguard Real Estate ETF)")
    vnq_prices = load_reit_data("VNQ", years=5, interval="monthly")
    print(f"Loaded {len(vnq_prices)} monthly prices")
    print(f"Mean: ${vnq_prices.mean():.2f}, Std: ${vnq_prices.std():.2f}")
    
    # Example 2: Load multiple REITs
    print("\nExample 2: Loading multiple REITs")
    multi_data = load_multi_reit_data(["VNQ", "IYR", "SCHH"], years=3, interval="monthly")
    for symbol, prices in multi_data.items():
        print(f"{symbol}: {len(prices)} points, ${prices[-1]:.2f} current")
    
    # Example 3: Load economic indicators
    try:
        print("\nExample 3: Loading economic indicators")
        indicators = load_economic_indicators(years=5)
        for name, values in indicators.items():
            print(f"{name}: {len(values)} observations, latest={values[-1]:.2f}")
    except Exception as e:
        print(f"Could not load economic indicators: {e}")
    
    print("\n" + "="*70)
    print("Setup instructions:")
    print("1. Create a .env file in Market_Sim/ directory")
    print("2. Add your API keys (see qfclient-main/.env.example)")
    print("3. Free options: Yahoo Finance (no key), Finnhub, FMP, FRED")
    print("4. Run this script to test your setup")
    print("="*70)
