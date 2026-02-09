"""
Technical indicator calculations for trading environment.

Loads and preprocesses price data, adding relative technical features
suitable for reinforcement learning.
"""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator


def load_and_preprocess_data(csv_path: str):
    """
    Load price data and add relative technical indicators.
    
    Calculates scale-invariant features (RSI, ATR, MA slopes, MA divergences)
    that work across different price levels. Avoids raw price features.
    
    Args:
        csv_path: Path to OHLCV CSV file
        
    Returns:
        tuple: (processed DataFrame with OHLCV + indicators, list of feature column names)
        
    Expected CSV columns:
        - Time/datetime (any time column name)
        - Open, High, Low, Close, Volume (case-insensitive)
    """
    # Read CSV and normalize column names
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Map various column name formats to standard names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['time (eet)', 'gmt time', 'datetime', 'date', 'time']:
            column_mapping[col] = 'Time (EET)'
        elif col_lower == 'open':
            column_mapping[col] = 'Open'
        elif col_lower == 'high':
            column_mapping[col] = 'High'
        elif col_lower == 'low':
            column_mapping[col] = 'Low'
        elif col_lower == 'close':
            column_mapping[col] = 'Close'
        elif col_lower == 'volume':
            column_mapping[col] = 'Volume'
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Parse datetime and set as index
    df["Time (EET)"] = pd.to_datetime(df["Time (EET)"])
    df = df.set_index("Time (EET)")
    df.sort_index(inplace=True)

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =============================================================================
    # Technical Indicators (Scale-Invariant Features)
    # =============================================================================
    
    # RSI: bounded [0, 100], naturally scale-invariant
    df["rsi_14"] = RSIIndicator(
        close=df["Close"], 
        window=14
    ).rsi()
    
    # ATR: measures absolute volatility range
    df["atr_14"] = AverageTrueRange(
        high=df["High"], 
        low=df["Low"], 
        close=df["Close"], 
        window=14
    ).average_true_range()

    # Moving averages (not directly observed by agent)
    df["ma_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["ma_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()

    # MA slopes (rate of change)
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Price distance from MAs (relative positioning)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA spread (trend strength indicator)
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # Drop rows with NaN values from indicator calculation
    df.dropna(inplace=True)

    # Feature columns for agent (excludes raw prices and raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
    ]

    return df, feature_cols
