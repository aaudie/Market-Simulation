# qfclient

A Python library for fetching market and cryptocurrency data from multiple providers with automatic failover, rate limiting, and strict Pydantic types.

## Features

- **Multi-provider support** - 12 market providers + 4 crypto providers
- **Automatic failover** - Seamlessly switches providers on rate limits or errors
- **Strict typing** - All data returned as Pydantic models
- **DataFrame conversion** - Easy `.to_df()` method on all list results
- **Rate limiting** - Built-in tracking with dynamic header parsing
- **Async support** - High-performance concurrent operations
- **Batch operations** - Fetch multiple symbols in parallel
- **CLI tools** - Built-in commands for diagnostics and quick lookups

## Installation

```bash
# From GitHub
pip install git+https://github.com/oregonquantgroup/qfclient.git

# For development
git clone https://github.com/oregonquantgroup/qfclient.git
cd qfclient
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Core: `pydantic`, `httpx`, `python-dotenv`, `yfinance`, `pandas`

## Quick Start

```python
from qfclient import MarketClient, CryptoClient
from datetime import date

# Market data
market = MarketClient()

# Get a quote
quote = market.get_quote("AAPL")
print(f"AAPL: ${quote.price}")

# Get OHLCV candles
candles = market.get_ohlcv("AAPL", start=date(2024, 1, 1))
df = candles.to_df()

# Crypto data
crypto = CryptoClient()
btc = crypto.get_quote("BTC")
print(f"Bitcoin: ${btc.price_usd:,.2f}")

# Get top coins by market cap
top_coins = crypto.get_top_coins(limit=10)
```

## CLI Commands

After installation, the `qfclient` command is available:

```bash
# List all providers and their configuration status
qfclient list

# Run diagnostic tests on all providers
qfclient test

# Show version
qfclient version

# JSON output (for scripting)
qfclient list -q
qfclient test -q
```

**Example output of `qfclient list`:**
```
==================================================
qfclient Provider Status
==================================================

Market Data Providers:
------------------------------
  [x] alpaca
  [x] finnhub
  [x] yfinance
  [ ] polygon

  Configured: 10/12

Crypto Providers:
------------------------------
  [x] coingecko
  [x] coinmarketcap
  ...
```

**Example output of `qfclient test`:**
```
==================================================
qfclient Diagnostic Tests
==================================================

Market Data Tests:
------------------------------
  [PASS] Quote (AAPL)
         $250.50
  [PASS] OHLCV (AAPL)
         5 candles
  [PASS] Dividends (AAPL)
         89 dividends
  ...

Results: 12/12 passed, 0 failed, 0 skipped
==================================================
```

### Programmatic Access

The CLI functions are also available as Python imports:

```python
from qfclient import list_providers, run_diagnostics

# List configured providers
status = list_providers(verbose=True)  # prints and returns dict
status = list_providers(verbose=False)  # returns dict only

# Run diagnostic tests
results = run_diagnostics(verbose=True)  # prints and returns dict
# results = {"passed": [...], "failed": [...], "skipped": [...]}
```

## Environment Setup

qfclient uses environment variables for API keys. Create a `.env` file in your project root:

```bash
cp .env.example .env
nano .env
```

### Required vs Optional Keys

**Works without any API keys:**
- Yahoo Finance (market data)
- CoinGecko (crypto data)
- CryptoCompare (crypto data)

**Recommended free API keys:**

| Provider | Get Key | Best For |
|----------|---------|----------|
| Finnhub | [finnhub.io](https://finnhub.io/register) | Quotes, profiles, earnings, news |
| FMP | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer) | OHLCV, fundamentals |
| FRED | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Economic indicators |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Real-time quotes, OHLCV |

### Example `.env` File

```bash
# Market Data Providers
ALPACA_API_KEY_ID=your-key-here
ALPACA_API_SECRET_KEY=your-secret-here
FINNHUB_API_KEY=your-key-here
FMP_API_KEY=your-key-here
FRED_API_KEY=your-key-here
TRADIER_API_KEY=your-key-here
TWELVE_DATA_API_KEY=your-key-here

# Crypto Providers (optional - works without keys)
COINGECKO_API_KEY=your-key-here
COINMARKETCAP_API_KEY=your-key-here
CRYPTOCOMPARE_API_KEY=your-key-here
```

See [`.env.example`](.env.example) for all providers and signup links.

## API Reference

### MarketClient

```python
from qfclient import MarketClient, Interval
from datetime import date

client = MarketClient()

# Quotes
quote = client.get_quote("AAPL")

# OHLCV (candlestick data)
candles = client.get_ohlcv(
    "AAPL",
    interval=Interval.DAY_1,  # 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
    limit=100
)

# Company info
profile = client.get_company_profile("AAPL")
earnings = client.get_earnings(symbol="AAPL")
news = client.get_news("AAPL", limit=10)

# Options
expirations = client.get_option_expirations("AAPL")
chain = client.get_options_chain("AAPL", expiration=expirations[0])

# Fundamentals
dividends = client.get_dividends("AAPL")
splits = client.get_stock_splits("AAPL")
recommendations = client.get_recommendations("AAPL")
price_target = client.get_price_target("AAPL")  # requires Finnhub paid

# Financials (requires FMP paid tier)
income = client.get_income_statement("AAPL", period="annual", limit=5)
balance = client.get_balance_sheet("AAPL", period="quarterly", limit=4)
cash_flow = client.get_cash_flow("AAPL")

# Insider activity (from provider APIs - requires paid tier)
insider_txns = client.get_insider_transactions("AAPL")

# SEC Form 4 insider data (free - direct from SEC EDGAR)
# Get complete Form 4 filings with rich data
filings = client.get_sec_filings("AAPL", limit=10)
for f in filings:
    print(f"{f.owner_name} ({f.role.role_description}): {f.net_shares:+,.0f} shares")

# Get individual transactions with position change %
transactions = client.get_sec_transactions("AAPL")
for t in transactions:
    print(f"{t.owner_name} ({t.role.role_type.value}): {t.shares:,.0f} shares")
    if t.position_change_pct:
        print(f"  Position change: {t.position_change_pct:+.1f}%")

# Get aggregated insider sentiment
summary = client.get_insider_summary("AAPL")
print(f"NPR: {summary.net_purchase_ratio:.2f}, Sentiment: {summary.insider_sentiment}")

# Economic data (FRED)
fed_rate = client.get_economic_indicator("FEDFUNDS", limit=12)
unemployment = client.get_economic_indicator("UNRATE", start=date(2020, 1, 1))
```

### CryptoClient

```python
from qfclient import CryptoClient, Interval

client = CryptoClient()

# Quotes
btc = client.get_quote("BTC")
print(f"BTC: ${btc.price_usd:,.2f}, 24h change: {btc.percent_change_24h:.2f}%")

# OHLCV
candles = client.get_ohlcv("ETH", interval=Interval.HOUR_1, limit=24)

# Asset info
asset = client.get_asset("BTC")

# Market data with ATH, supply info
market_data = client.get_market_data("BTC")
print(f"ATH: ${market_data.ath:,.2f}")

# Top coins by market cap
top_100 = client.get_top_coins(limit=100)
df = top_100.to_df()

# Trending coins
trending = client.get_trending()

# Global market overview
global_data = client.get_global_market()
print(f"Total market cap: ${global_data['total_market_cap_usd']:,.0f}")
```

### Batch Operations

```python
# Fetch multiple symbols in parallel
quotes = client.get_quotes_batch(["AAPL", "GOOGL", "MSFT", "AMZN"])
for symbol, quote in quotes.items():
    print(f"{symbol}: ${quote.price}")

ohlcv_data = client.get_ohlcv_batch(["AAPL", "GOOGL", "MSFT"], start=date(2024, 1, 1))
```

### Async Client

```python
import asyncio
from qfclient import AsyncMarketClient

async def main():
    async with AsyncMarketClient() as client:
        quotes = await client.get_quotes_batch([
            "AAPL", "GOOGL", "MSFT", "AMZN", "META"
        ])
        for symbol, quote in quotes.items():
            print(f"{symbol}: ${quote.price}")

asyncio.run(main())
```

### Working with Results

All list results are `ResultList` objects:

```python
candles = client.get_ohlcv("AAPL", limit=100)

df = candles.to_df()           # pandas DataFrame
data = candles.to_dicts()      # list of dicts
json_str = candles.to_json()   # JSON string

# Filter and iterate
high_volume = candles.filter(lambda c: c.volume > 1_000_000)
for candle in candles:
    print(candle.close)
```

### Provider Selection

```python
# Set preferred provider globally
client = MarketClient(prefer="alpaca")

# Or per request
quote = client.get_quote("AAPL", prefer="finnhub")

# Check provider status
status = client.get_status()
for provider, info in status.items():
    if info["configured"]:
        print(f"{provider}: ready")
```

## Data Models

All data is returned as strictly-typed Pydantic models:

### Market Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `Quote` | Real-time price | `price`, `bid`, `ask`, `volume`, `timestamp` |
| `OHLCV` | Candlestick | `open`, `high`, `low`, `close`, `volume`, `timestamp` |
| `CompanyProfile` | Company info | `name`, `sector`, `industry`, `market_cap` |
| `EarningsEvent` | Earnings | `date`, `eps_estimate`, `eps_actual` |
| `OptionChain` | Options | `calls`, `puts`, `expiration`, `underlying_price` |
| `OptionContract` | Single option | `strike`, `delta`, `gamma`, `theta`, `vega`, `iv` |
| `EconomicIndicator` | FRED data | `observation_date`, `value`, `series_id` |
| `Dividend` | Dividends | `ex_date`, `payment_date`, `amount` |

### SEC Form 4 Models (Insider Trading)

| Model | Description | Key Fields |
|-------|-------------|------------|
| `SECInsiderTransaction` | Single transaction | `shares`, `price_per_share`, `position_change_pct`, `role` |
| `Form4Filing` | Complete filing | `transactions`, `net_shares`, `net_value`, `owner_name`, `role` |
| `InsiderSummary` | Aggregated stats | `net_purchase_ratio`, `num_buyers`, `num_sellers`, `insider_sentiment` |
| `InsiderRole` | Role info | `is_ceo`, `is_cfo`, `is_director`, `role_type` |
| `InsiderRoleType` | Role enum | `CEO`, `CFO`, `COO`, `OTHER_OFFICER`, `DIRECTOR`, `TEN_PERCENT_OWNER` |

### Crypto Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `CryptoQuote` | Price | `price_usd`, `market_cap`, `volume_24h`, `percent_change_24h` |
| `CryptoOHLCV` | Candlestick | `open`, `high`, `low`, `close`, `volume`, `timestamp` |
| `CryptoAsset` | Asset info | `name`, `symbol`, `description`, `website` |
| `CryptoMarketData` | Full data | `price`, `ath`, `circulating_supply`, `max_supply` |

## Providers

### Market Data Providers

| Provider | API Key | Rate Limit | Features |
|----------|---------|------------|----------|
| Yahoo Finance | No | ~30/min | Quotes, OHLCV, Options, Dividends |
| Alpaca | Yes | 200/min | Quotes, OHLCV |
| Alpha Vantage | Yes | 5/min | Quotes, OHLCV, Profiles |
| Finnhub | Yes | 60/min | Quotes, Profiles, Earnings, News |
| FMP | Yes | 250/day | Quotes, OHLCV, Profiles |
| FRED | Yes | 120/min | Economic indicators |
| Tiingo | Yes | 1000/hr | Quotes, OHLCV, News |
| Tradier | Yes | 120/min | OHLCV, Options with Greeks |
| Twelve Data | Yes | 8/min | Quotes, OHLCV |
| Polygon | Yes | 5/min | Quotes, OHLCV |
| EODHD | Yes | 20/day | Global EOD data |
| Marketstack | Yes | 100/mo | Global EOD data |
| SEC EDGAR | No | 10/sec | Form 4 insider transactions, filings |

### Crypto Providers

| Provider | API Key | Rate Limit | Features |
|----------|---------|------------|----------|
| CoinGecko | Optional | 30/min | Quotes, OHLCV, Profiles, Rankings |
| CoinMarketCap | Yes | 30/min | Quotes, Profiles, Rankings |
| CryptoCompare | Optional | 50/min | Quotes, OHLCV (best for intraday) |

## Error Handling

```python
from qfclient import MarketClient, ProviderError, RateLimitError

client = MarketClient()

try:
    quote = client.get_quote("AAPL")
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, retry after {e.retry_after}s")
except ProviderError as e:
    print(f"Provider {e.provider} error: {e}")
```

## Common FRED Series IDs

```python
# Interest Rates
"FEDFUNDS"    # Federal Funds Rate
"DGS10"       # 10-Year Treasury
"DGS2"        # 2-Year Treasury

# Employment
"UNRATE"      # Unemployment Rate
"PAYEMS"      # Nonfarm Payrolls

# Inflation
"CPIAUCSL"    # Consumer Price Index
"PCEPI"       # PCE Price Index

# GDP
"GDP"         # Gross Domestic Product
"GDPC1"       # Real GDP
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run diagnostics
qfclient test

# Format code
black .

# Lint
ruff check . --fix

# Type check
mypy .
```

## License

MIT
