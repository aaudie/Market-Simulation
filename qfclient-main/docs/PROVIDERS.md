# Provider Reference

Complete guide to all data providers supported by qfclient.

## Overview

qfclient supports 12 market data providers and 4 crypto providers. The client automatically selects the best available provider based on:

1. Whether the provider is configured (API key present)
2. Current rate limit status
3. Provider capabilities for the requested data type
4. User preference (if specified)

## Market Data Providers

### Yahoo Finance

**No API key required** - Works out of the box.

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |
| Company Profiles | Yes |
| Earnings | Yes |
| Options | Yes |
| Dividends | Yes |
| News | No |

**Rate Limit:** ~30 requests/min (IP-based)

**Notes:**
- Best free option for basic market data
- Uses the `yfinance` library
- Rate limits are based on IP address, not API key
- Good for personal projects and backtesting

---

### Alpaca

**Environment Variables:**
```bash
ALPACA_API_KEY_ID=your-key-id
ALPACA_API_SECRET_KEY=your-secret-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |
| Company Profiles | No |
| Options | No |

**Rate Limit:** 200 requests/min

**Signup:** [alpaca.markets](https://alpaca.markets) (free brokerage account)

**Notes:**
- Fast, reliable real-time quotes
- Excellent for high-frequency data needs
- Free tier requires opening a brokerage account (no deposit needed)

---

### Finnhub

**Environment Variable:**
```bash
FINNHUB_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Paid only |
| Company Profiles | Yes |
| Earnings | Yes |
| News | Yes |
| Insider Transactions | Yes |
| Recommendations | Yes |

**Rate Limit:** 60 requests/min

**Signup:** [finnhub.io/register](https://finnhub.io/register)

**Notes:**
- Excellent free tier for fundamental data
- OHLCV (candle data) requires paid subscription
- Great for company profiles, earnings, and news

---

### Financial Modeling Prep (FMP)

**Environment Variable:**
```bash
FMP_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |
| Company Profiles | Yes |
| Earnings | Yes |
| Financials | Yes |
| Dividends | Yes |
| Insider Transactions | Yes |

**Rate Limit:** 250 requests/day (free tier)

**Signup:** [financialmodelingprep.com](https://site.financialmodelingprep.com/developer)

**Notes:**
- Best free source for financial statements
- Income statement, balance sheet, cash flow
- Daily request limit rather than per-minute

---

### FRED (Federal Reserve Economic Data)

**Environment Variable:**
```bash
FRED_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Economic Indicators | Yes |

**Rate Limit:** 120 requests/min

**Signup:** [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

**Notes:**
- Only source for economic data
- 800,000+ data series available
- Essential for macro analysis

**Popular Series IDs:**
```python
# Interest Rates
"FEDFUNDS"  # Federal Funds Rate
"DGS10"     # 10-Year Treasury
"DGS2"      # 2-Year Treasury

# Employment
"UNRATE"    # Unemployment Rate
"PAYEMS"    # Nonfarm Payrolls

# Inflation
"CPIAUCSL"  # Consumer Price Index
"PCEPI"     # PCE Price Index

# GDP
"GDP"       # Gross Domestic Product
"GDPC1"     # Real GDP
```

---

### Tradier

**Environment Variable:**
```bash
TRADIER_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |
| Options | Yes (with Greeks) |

**Rate Limit:** 120 requests/min

**Signup:** [tradier.com](https://tradier.com) (free brokerage account)

**Notes:**
- Best free source for options data with Greeks
- Includes delta, gamma, theta, vega, implied volatility
- Requires opening a brokerage account

---

### Twelve Data

**Environment Variable:**
```bash
TWELVE_DATA_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |

**Rate Limit:** 8 requests/min, 800 requests/day

**Signup:** [twelvedata.com/register](https://twelvedata.com/register)

**Notes:**
- Low rate limit on free tier
- Good as a fallback provider
- Supports forex data

---

### Alpha Vantage

**Environment Variable:**
```bash
ALPHA_VANTAGE_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |
| Company Profiles | Yes |
| Earnings | Yes |

**Rate Limit:** 5 requests/min, 25 requests/day

**Signup:** [alphavantage.co](https://www.alphavantage.co/support/#api-key) (email only)

**Notes:**
- Very low rate limit
- Can create multiple accounts for higher throughput (email+1@gmail.com trick)
- Good historical data coverage

---

### Tiingo

**Environment Variable:**
```bash
TIINGO_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes (IEX) |
| OHLCV | Yes |
| News | Yes |

**Rate Limit:** 1000 requests/hour, 500 unique symbols/month

**Signup:** [tiingo.com](https://www.tiingo.com/account/signup)

**Notes:**
- Good rate limits
- Symbol limit on free tier (500 unique symbols/month)
- Quality news data

---

### Polygon.io

**Environment Variable:**
```bash
POLYGON_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes |

**Rate Limit:** 5 requests/min (free tier)

**Signup:** [polygon.io](https://polygon.io/dashboard/signup)

**Notes:**
- Premium data quality
- Free tier very limited
- Paid tiers excellent for production

---

### EODHD

**Environment Variable:**
```bash
EODHD_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| OHLCV | Yes |
| Company Profiles | Paid only |

**Rate Limit:** 20 requests/day (free tier)

**Signup:** [eodhd.com](https://eodhd.com/register)

**Notes:**
- Global market coverage (70+ exchanges)
- Very limited free tier
- Good for international stocks

---

### Marketstack

**Environment Variable:**
```bash
MARKETSTACK_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| OHLCV | Yes |

**Rate Limit:** 100 requests/month (free tier)

**Signup:** [marketstack.com](https://marketstack.com/signup/free)

**Notes:**
- Global market coverage (72+ exchanges)
- Free tier uses HTTP only (no HTTPS)
- Very limited free tier

---

## Crypto Providers

### CoinGecko

**Environment Variable (optional):**
```bash
COINGECKO_API_KEY=your-key  # Optional, for higher limits
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes (daily only) |
| Asset Profiles | Yes |
| Market Data | Yes |
| Top Coins | Yes |
| Trending | Yes |
| Global Market | Yes |

**Rate Limit:**
- Without key: 10-30 requests/min
- With demo key: 30 requests/min, 10,000/month

**Signup:** [coingecko.com/api](https://www.coingecko.com/en/api/pricing) (optional)

**Notes:**
- Best free crypto data source
- Works without API key
- Daily OHLCV only (no intraday)

---

### CryptoCompare

**Environment Variable (optional):**
```bash
CRYPTOCOMPARE_API_KEY=your-key  # Optional, for higher limits
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| OHLCV | Yes (1m to daily) |
| Top Coins | Yes |

**Rate Limit:**
- Without key: Lower limits
- With key: 50 requests/min, 100,000/month

**Signup:** [min-api.cryptocompare.com](https://min-api.cryptocompare.com)

**Notes:**
- **Best free source for intraday crypto OHLCV**
- Supports 1m, 5m, 15m, 30m, 1h, 4h, 1d intervals
- Works without API key at lower limits

---

### CoinMarketCap

**Environment Variable:**
```bash
COINMARKETCAP_API_KEY=your-key
```

| Feature | Status |
|---------|--------|
| Quotes | Yes |
| Asset Profiles | Yes |
| Top Coins | Yes |

**Rate Limit:** 30 requests/min, 333 requests/day

**Signup:** [coinmarketcap.com/api](https://coinmarketcap.com/api/)

**Notes:**
- Industry standard for crypto rankings
- No OHLCV data on free tier
- Good for market cap and volume data

---

### Messari

**Environment Variable:**
```bash
MESSARI_API_KEY=your-key
```

**Status:** Deprecated

**Notes:**
- The data.messari.io API has been deprecated as of 2024
- Contact sales@messari.io for new API access
- Provider is disabled by default

---

## Provider Selection

### Automatic Selection

By default, qfclient automatically selects the best provider:

```python
from qfclient import MarketClient

client = MarketClient()
quote = client.get_quote("AAPL")  # Uses best available provider
```

### Manual Selection

You can prefer a specific provider:

```python
# Global preference
client = MarketClient(prefer="alpaca")

# Per-request preference
quote = client.get_quote("AAPL", prefer="finnhub")
```

### Check Provider Status

```python
client = MarketClient()
status = client.get_status()

for provider, info in status.items():
    configured = "Yes" if info["configured"] else "No"
    print(f"{provider}: Configured={configured}")
```

### Direct Provider Access

```python
from qfclient.market.providers import AlpacaProvider

alpaca = AlpacaProvider()

# Check if configured
if alpaca.is_configured():
    quote = alpaca.get_quote("AAPL")

# Check capabilities
print(f"Supports OHLCV: {alpaca.supports_ohlcv}")
print(f"Supports options: {alpaca.supports_options}")
print(f"Supports news: {alpaca.supports_news}")
```

## Rate Limit Handling

qfclient handles rate limits automatically:

1. **Tracking:** Monitors requests per provider
2. **Selection:** Avoids rate-limited providers
3. **Failover:** Switches to another provider on 429 errors
4. **Headers:** Parses rate limit headers from responses

```python
from qfclient import RateLimitError

try:
    quote = client.get_quote("AAPL")
except RateLimitError as e:
    print(f"Provider {e.provider} rate limited")
    print(f"Retry after: {e.retry_after} seconds")
```

## Recommended Setup

For most use cases, we recommend configuring these providers:

```bash
# .env

# Primary market data (choose based on needs)
ALPACA_API_KEY_ID=...      # Fast quotes
ALPACA_API_SECRET_KEY=...
FINNHUB_API_KEY=...        # Company data, earnings

# Fundamentals
FMP_API_KEY=...            # Financial statements

# Economic data
FRED_API_KEY=...           # GDP, inflation, rates

# Options (if needed)
TRADIER_API_KEY=...        # Options with Greeks
```

Yahoo Finance and CoinGecko work without keys and serve as good fallbacks.
