"""
Data loading utilities for market simulation.

Provides functions to load historical price data from CSV files and APIs.
"""

import csv
import requests
from datetime import datetime
from typing import List

from sim.types import HistoricalPoint


def load_cre_csv(path: str) -> List[HistoricalPoint]:
    """
    Load CRE (Commercial Real Estate) historical data from CSV file.
    
    Expected CSV format: date,price
    Date can be YYYY-MM or YYYY-MM-DD format.
    
    Args:
        path: Path to CSV file
        
    Returns:
        List of HistoricalPoint objects sorted by date
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    points: List[HistoricalPoint] = []
    
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row

        for row in reader:
            if len(row) < 2:
                continue

            date_str = row[0].strip()
            price_str = row[1].strip()

            try:
                price = float(price_str)
            except ValueError:
                continue

            # Try parsing different date formats
            dt = None
            for fmt in ("%Y-%m-%d", "%Y-%m"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

            if dt is None:
                continue

            points.append(HistoricalPoint(dt, price))

    points.sort(key=lambda p: p.date)
    return points


def load_twelvedata_api(
    symbol: str = "AAPL",
    interval: str = "1month",
    start_date: str = "2004-01-01 00:00:00",
    end_date: str = "2026-01-01 00:00:00",
    api_key: str = "d0769a0d6646410880b5a6fd5d19328d"
) -> List[HistoricalPoint]:
    """
    Fetch historical stock data from Twelve Data API.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "MSFT", "GOOGL", "VNQ")
        interval: Time interval ("1month", "1week", "1day", etc.)
        start_date: Start date in format "YYYY-MM-DD HH:MM:SS"
        end_date: End date in format "YYYY-MM-DD HH:MM:SS"
        api_key: Twelve Data API key
    
    Returns:
        List of HistoricalPoint objects sorted by date (oldest first)
        
    Raises:
        requests.HTTPError: If API request fails
    """
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"apikey={api_key}&symbol={symbol}&interval={interval}&format=CSV"
        f"&start_date={start_date}&end_date={end_date}"
    )
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Parse CSV response (semicolon-delimited)
    # Format: datetime;open;high;low;close;volume
    points: List[HistoricalPoint] = []
    lines = response.text.strip().split('\n')
    
    for i, line in enumerate(lines):
        if i == 0:  # Skip header row
            continue
        
        parts = line.split(';')
        if len(parts) < 5:
            continue
        
        try:
            date_str = parts[0].strip()
            close_price = float(parts[4].strip())
            
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            points.append(HistoricalPoint(dt, close_price))
        except (ValueError, IndexError):
            continue
    
    # Sort by date (API may return newest first, we want oldest first)
    points.sort(key=lambda p: p.date)
    return points
