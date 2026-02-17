# Quick Start Guide

Get up and running with qfclient in 5 minutes.

## Installation

```bash
pip install git+https://github.com/oregonquantgroup/qfclient.git
```

## Minimal Setup (No API Keys)

qfclient works out of the box with Yahoo Finance and CoinGecko - no API keys required:

```python
from qfclient import MarketClient, CryptoClient
from datetime import date

# Market data via Yahoo Finance
market = MarketClient()
quote = market.get_quote("AAPL")
print(f"AAPL: ${quote.price}")

# Historical data
candles = market.get_ohlcv("AAPL", start=date(2024, 1, 1))
df = candles.to_df()
print(df.head())

# Crypto data via CoinGecko
crypto = CryptoClient()
btc = crypto.get_quote("BTC")
print(f"Bitcoin: ${btc.price_usd:,.2f}")
```

## Adding API Keys (Recommended)

For better rate limits and more features, add API keys:

### 1. Create `.env` file

```bash
# In your project directory
touch .env
```

### 2. Add your keys

```bash
# .env
FINNHUB_API_KEY=your-finnhub-key
FMP_API_KEY=your-fmp-key
FRED_API_KEY=your-fred-key
```

### 3. Get free API keys

| Provider | Signup Link | What You Get |
|----------|-------------|--------------|
| Finnhub | [finnhub.io/register](https://finnhub.io/register) | 60 req/min, company profiles, earnings |
| FMP | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer) | 250 req/day, financials, fundamentals |
| FRED | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Economic data (GDP, inflation, rates) |
| Alpaca | [alpaca.markets](https://alpaca.markets) | 200 req/min, real-time quotes |

### 4. Keys are loaded automatically

```python
from qfclient import MarketClient

# .env is loaded automatically on import
client = MarketClient()

# Now you have access to all configured providers
quote = client.get_quote("AAPL")  # Uses best available provider
```

## Common Use Cases

### Get Stock Quotes

```python
from qfclient import MarketClient

client = MarketClient()

# Single quote
quote = client.get_quote("AAPL")
print(f"Price: ${quote.price}, Volume: {quote.volume:,}")

# Multiple quotes
quotes = client.get_quotes_batch(["AAPL", "GOOGL", "MSFT"])
for symbol, q in quotes.items():
    print(f"{symbol}: ${q.price}")
```

### Get Historical Data

```python
from qfclient import MarketClient, Interval
from datetime import date

client = MarketClient()

# Daily candles
candles = client.get_ohlcv(
    "AAPL",
    interval=Interval.DAY_1,
    start=date(2024, 1, 1),
    limit=252  # ~1 year of trading days
)

# Convert to DataFrame
df = candles.to_df()
print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail())
```

### Get Company Information

```python
client = MarketClient()

# Company profile
profile = client.get_company_profile("AAPL")
print(f"{profile.name}")
print(f"Sector: {profile.sector}")
print(f"Market Cap: ${profile.market_cap:,.0f}")

# Earnings calendar
earnings = client.get_earnings(symbol="AAPL")
for e in earnings[:3]:
    print(f"{e.date}: EPS Est: ${e.eps_estimate}, Actual: ${e.eps_actual}")
```

### Get Crypto Data

```python
from qfclient import CryptoClient

client = CryptoClient()

# Bitcoin price
btc = client.get_quote("BTC")
print(f"BTC: ${btc.price_usd:,.2f}")
print(f"24h Change: {btc.percent_change_24h:.2f}%")
print(f"Market Cap: ${btc.market_cap:,.0f}")

# Top 10 coins
top = client.get_top_coins(limit=10)
for coin in top:
    print(f"{coin.symbol}: ${coin.price_usd:,.2f}")
```

### Get Economic Data

```python
from qfclient import MarketClient
from datetime import date

client = MarketClient()

# Federal Funds Rate (last 12 months)
rates = client.get_economic_indicator("FEDFUNDS", limit=12)
for r in rates:
    print(f"{r.date}: {r.value}%")

# Unemployment since 2020
unemployment = client.get_economic_indicator(
    "UNRATE",
    start=date(2020, 1, 1)
)
df = unemployment.to_df()
```

### Get Options Data

```python
from qfclient import MarketClient

client = MarketClient()

# Available expirations
expirations = client.get_option_expirations("AAPL")
print(f"Expirations: {expirations[:5]}")

# Options chain with Greeks
chain = client.get_options_chain("AAPL", expiration=expirations[0])

print(f"Underlying: ${chain.underlying_price}")
print(f"\nCalls (first 5):")
for call in chain.calls[:5]:
    print(f"  Strike: ${call.strike}, Delta: {call.delta:.3f}, IV: {call.implied_volatility:.1%}")
```

## Next Steps

- See the full [README](../README.md) for complete API reference
- Check [.env.example](../.env.example) for all available providers
- Run `client.get_status()` to see which providers are configured
