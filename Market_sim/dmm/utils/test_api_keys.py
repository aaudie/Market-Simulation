"""
Test API Keys and Data Sources

This script tests all configured API keys and data sources
to ensure they're working properly for DMM training.

Usage:
    cd Market_sim/dmm/utils
    python3 test_api_keys.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
# Navigate from utils/ -> dmm/ -> Market_sim/ -> Market_Sim/ -> .env
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Add qfclient to path
# Navigate from utils/ -> dmm/ -> Market_sim/ -> Market_Sim/ -> qfclient-main/
qfclient_path = Path(__file__).resolve().parent.parent.parent.parent / "qfclient-main" / "src"
sys.path.insert(0, str(qfclient_path))

try:
    from qfclient import MarketClient, Interval
    from datetime import date, timedelta
    QFCLIENT_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import qfclient: {e}")
    print("Install with: cd qfclient-main && pip install -e .")
    sys.exit(1)


def test_yahoo_finance():
    """Test Yahoo Finance (no API key needed)."""
    print("\n" + "="*70)
    print("YAHOO FINANCE (FREE - No API Key Required)")
    print("="*70)
    
    client = MarketClient()
    
    # Test 1: Stock quote
    print("\n1. Testing stock quote...")
    try:
        quote = client.get_quote("AAPL")
        print(f"   ✓ AAPL quote: ${quote.price:.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: REIT data (critical for training)
    print("\n2. Testing REIT data (VNQ, IYR, SCHH)...")
    reits = ["VNQ", "IYR", "SCHH"]
    success_count = 0
    
    for symbol in reits:
        try:
            candles = client.get_ohlcv(
                symbol, 
                interval=Interval.MONTH_1, 
                limit=12
            )
            df = candles.to_df()
            print(f"   ✓ {symbol}: {len(df)} months, latest price: ${df['close'].iloc[-1]:.2f}")
            success_count += 1
        except Exception as e:
            print(f"   ✗ {symbol} failed: {e}")
    
    print(f"\n   Result: {success_count}/{len(reits)} REITs loaded successfully")
    return success_count > 0


def test_fred():
    """Test FRED economic indicators."""
    print("\n" + "="*70)
    print("FRED - Federal Reserve Economic Data")
    print("="*70)
    
    # Check if key is configured
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        print("   ⚠ FRED_API_KEY not found in .env file")
        print("   Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False
    
    print(f"   API Key configured: {fred_key[:10]}...")
    
    client = MarketClient()
    
    # Test economic indicators
    indicators = {
        "FEDFUNDS": "Federal Funds Rate",
        "MORTGAGE30US": "30-Year Mortgage Rate",
        "UNRATE": "Unemployment Rate"
    }
    
    print("\n   Testing economic indicators:")
    success_count = 0
    
    for series_id, name in indicators.items():
        try:
            data = client.get_economic_indicator(series_id, limit=1)
            print(f"   ✓ {name}: {data[-1].value}")
            success_count += 1
        except Exception as e:
            print(f"   ✗ {name} failed: {e}")
    
    print(f"\n   Result: {success_count}/{len(indicators)} indicators loaded")
    return success_count > 0


def test_sec():
    """Test SEC Edgar (no API key needed)."""
    print("\n" + "="*70)
    print("SEC EDGAR (FREE - No API Key Required)")
    print("="*70)
    
    client = MarketClient()
    
    # Test insider trading data
    print("\n   Testing SEC Form 4 filings...")
    try:
        filings = client.get_sec_filings("AAPL", limit=5)
        print(f"   ✓ Retrieved {len(filings)} recent Form 4 filings")
        if filings:
            print(f"   Latest: {filings[0].owner_name}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def test_other_keys():
    """Test other configured API keys."""
    print("\n" + "="*70)
    print("OTHER CONFIGURED KEYS")
    print("="*70)
    
    optional_keys = {
        "FINNHUB_API_KEY": "Finnhub",
        "FMP_API_KEY": "Financial Modeling Prep",
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage",
        "TWELVE_DATA_API_KEY": "Twelve Data",
        "ALPACA_API_KEY_ID": "Alpaca Markets"
    }
    
    found_keys = []
    missing_keys = []
    
    for env_var, name in optional_keys.items():
        key = os.getenv(env_var)
        if key:
            found_keys.append(f"   ✓ {name}")
        else:
            missing_keys.append(f"   ○ {name}")
    
    if found_keys:
        print("\n   Configured:")
        for line in found_keys:
            print(line)
    
    if missing_keys:
        print("\n   Not configured (optional):")
        for line in missing_keys:
            print(line)


def main():
    """Run all API tests."""
    print("="*70)
    print("API KEY AND DATA SOURCE TEST")
    print("="*70)
    print(f"\nChecking .env file at: {env_path}")
    print(f"Exists: {'✓' if env_path.exists() else '✗'}")
    
    if not env_path.exists():
        print("\nERROR: .env file not found!")
        print("Create one based on qfclient-main/.env.example")
        return
    
    # Run tests
    results = {
        "Yahoo Finance": test_yahoo_finance(),
        "FRED": test_fred(),
        "SEC Edgar": test_sec(),
    }
    
    test_other_keys()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nCritical for DMM Training:")
    for service, success in results.items():
        status = "✓ WORKING" if success else "✗ NOT WORKING"
        print(f"  {status}: {service}")
    
    all_critical_working = all(results.values())
    
    print("\n" + "="*70)
    if all_critical_working:
        print("✓ ALL SYSTEMS GO!")
        print("="*70)
        print("\nYou can train your DMM with real data:")
        print("  cd Market_sim/dmm/training")
        print("  python3 train_dmm_with_qfclient.py")
    else:
        print("⚠ SOME ISSUES DETECTED")
        print("="*70)
        print("\nYahoo Finance is the most important (it's free!).")
        print("FRED and SEC are optional but recommended.")
        print("\nIf Yahoo Finance is working, you're good to go!")
    print("="*70)


if __name__ == "__main__":
    main()
