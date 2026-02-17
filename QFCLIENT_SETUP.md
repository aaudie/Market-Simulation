# qfclient Setup Guide for DMM

This guide shows you how to configure qfclient to fetch real-time financial data for training your Deep Markov Model.

## Quick Start (No API Keys Required)

qfclient works out of the box with **Yahoo Finance** - no API keys needed:

```python
from dmm.qfclient_data_loader import load_reit_data

# This will work immediately without any setup
vnq_prices = load_reit_data("VNQ", years=5, interval="monthly")
```

## Step 1: Create .env File

Create a file named `.env` in your Market_Sim directory:

```bash
cd /Users/axelaudie/Desktop/Market_Sim\(wAI\)/Market_Sim
touch .env
```

## Step 2: Add API Keys (Optional but Recommended)

Edit the `.env` file and add your API keys. Here are the **best free options** for real estate data:

```bash
# ==============================================
# RECOMMENDED FREE API KEYS FOR DMM
# ==============================================

# 1. FRED (Federal Reserve) - HIGHLY RECOMMENDED
# Best for: Economic indicators (mortgage rates, housing starts, GDP)
# Rate limit: 120 req/min
# Signup: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your-fred-key-here

# 2. Finnhub - RECOMMENDED
# Best for: Company profiles, earnings, news
# Rate limit: 60 req/min
# Signup: https://finnhub.io/register
FINNHUB_API_KEY=your-finnhub-key-here

# 3. Financial Modeling Prep (FMP) - RECOMMENDED
# Best for: Historical OHLCV data, dividends
# Rate limit: 250 req/day
# Signup: https://site.financialmodelingprep.com/developer
FMP_API_KEY=your-fmp-key-here

# 4. Alpaca - OPTIONAL
# Best for: Real-time quotes
# Rate limit: 200 req/min
# Signup: https://alpaca.markets (free brokerage account)
ALPACA_API_KEY_ID=your-alpaca-key-here
ALPACA_API_SECRET_KEY=your-alpaca-secret-here

# Yahoo Finance works automatically - NO KEY NEEDED
```

### Getting Free API Keys (5 minutes total)

#### 1. FRED (1 minute)
- Go to: https://fred.stlouisfed.org/docs/api/api_key.html
- Click "Request API Key"
- Fill in your email
- Key is emailed instantly

#### 2. Finnhub (1 minute)
- Go to: https://finnhub.io/register
- Sign up with email
- Copy API key from dashboard

#### 3. FMP (2 minutes)
- Go to: https://site.financialmodelingprep.com/developer
- Sign up with email
- Verify email
- Copy API key from dashboard

## Step 3: Test Your Setup

Run the data loader test:

```bash
cd Market_sim/dmm
python qfclient_data_loader.py
```

You should see:

```
==================================================
qfclient Data Loader for DMM
==================================================

Example 1: Loading VNQ (Vanguard Real Estate ETF)
Loading VNQ data via qfclient...
✓ Loaded 60 monthly data points for VNQ
✓ Date range: 2019-02-11 to 2024-02-11
✓ Price range: $45.32 - $105.67
```

## Step 4: Train DMM with Live Data

Train your DMM using real-time data:

```bash
cd Market_sim/dmm
python train_dmm_with_qfclient.py
```

## What Data Can You Access?

### 1. REIT Data (Tokenized Real Estate Proxy)
```python
from dmm.qfclient_data_loader import load_reit_data

# Broad market REITs
vnq = load_reit_data("VNQ", years=20, interval="monthly")  # Vanguard Real Estate
iyr = load_reit_data("IYR", years=15, interval="monthly")  # iShares Real Estate

# Specific sectors
spg = load_reit_data("SPG", years=10, interval="monthly")  # Retail (Simon Property)
pld = load_reit_data("PLD", years=10, interval="monthly")  # Industrial (Prologis)
eqr = load_reit_data("EQR", years=10, interval="monthly")  # Residential (Equity Res)
```

### 2. Multiple REITs for Ensemble Training
```python
from dmm.qfclient_data_loader import load_multi_reit_data

data = load_multi_reit_data(["VNQ", "IYR", "SCHH"], years=5)
```

### 3. Economic Indicators
```python
from dmm.qfclient_data_loader import load_economic_indicators

indicators = load_economic_indicators(years=10)
# Returns: fed_funds_rate, mortgage_30y, housing_starts, unemployment, cpi, gdp
```

### 4. Alternative CRE Securities
```python
from dmm.qfclient_data_loader import load_cre_alternatives

cre_data = load_cre_alternatives(years=5)
# Returns data for: commercial, residential, office, industrial, retail REITs
```

## Integration with Existing DMM Code

Your existing `train_dmm.py` can be easily updated:

```python
# OLD: Using CSV files
def load_tokenized_reit_data() -> np.ndarray:
    history = load_twelvedata_api(...)  # Manual API calls
    prices = np.array([p.price for p in history])
    return prices

# NEW: Using qfclient
from dmm.qfclient_data_loader import load_reit_data

def load_tokenized_reit_data() -> np.ndarray:
    prices = load_reit_data("VNQ", years=20, interval="monthly")
    return prices
```

## Troubleshooting

### "qfclient not available"
```bash
cd qfclient-main
pip install -e .
```

### "Rate limit exceeded"
- Use multiple API keys (qfclient will auto-failover)
- Add more providers to your .env file
- Reduce data fetch frequency

### "No data returned"
- Check your API keys are correct
- Verify .env file is in the right location
- Try with Yahoo Finance first (no key needed)
- Check rate limits haven't been exceeded

## Provider Status

Check which providers are configured:

```python
from qfclient import MarketClient

client = MarketClient()
status = client.get_status()

for provider, info in status.items():
    if info['configured']:
        print(f"✓ {provider}: ready")
    else:
        print(f"✗ {provider}: not configured")
```

## Advantages Over CSV Files

1. **Always Up-to-Date**: Fetch latest data on demand
2. **Automatic Failover**: If one API is down, qfclient tries others
3. **No Manual Updates**: No need to download and update CSV files
4. **Multiple Sources**: Access 12+ data providers with one interface
5. **Economic Context**: Easy access to FRED economic indicators
6. **Type Safety**: All data is strictly typed with Pydantic models

## Rate Limits Summary

| Provider | Free Tier | Best For |
|----------|-----------|----------|
| Yahoo Finance | ~30/min | REIT OHLCV (no key needed) |
| FRED | 120/min | Economic indicators |
| Finnhub | 60/min | Company data, news |
| FMP | 250/day | Historical data |
| Alpaca | 200/min | Real-time quotes |

## Next Steps

1. ✅ Create `.env` file with API keys
2. ✅ Test with `python qfclient_data_loader.py`
3. ✅ Train DMM with `python train_dmm_with_qfclient.py`
4. ✅ Integrate with your simulation in `integrate_dmm.py`

## Support

- qfclient docs: `Market_Sim/qfclient-main/README.md`
- Data loader code: `Market_sim/dmm/qfclient_data_loader.py`
- Training script: `Market_sim/dmm/train_dmm_with_qfclient.py`

---

**Pro Tip**: Start with just FRED and Finnhub keys. That gives you:
- Economic indicators (FRED)
- Company profiles and news (Finnhub)
- REIT prices (Yahoo Finance, no key needed)

This covers 90% of what you need for DMM training!
