# Examples

Real-world usage examples for qfclient.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Portfolio Analysis](#portfolio-analysis)
- [Options Analysis](#options-analysis)
- [Crypto Portfolio](#crypto-portfolio)
- [Economic Dashboard](#economic-dashboard)
- [Async High-Performance](#async-high-performance)
- [Data Export](#data-export)

---

## Basic Usage

### Get Current Prices

```python
from qfclient import MarketClient

client = MarketClient()

# Single stock
quote = client.get_quote("AAPL")
print(f"AAPL: ${quote.price:.2f}")
print(f"  Bid: ${quote.bid} x {quote.bid_size}")
print(f"  Ask: ${quote.ask} x {quote.ask_size}")
print(f"  Volume: {quote.volume:,}")

# Multiple stocks
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
quotes = client.get_quotes_batch(symbols)

for symbol, q in quotes.items():
    print(f"{symbol}: ${q.price:.2f}")
```

### Get Historical Data

```python
from qfclient import MarketClient, Interval
from datetime import date

client = MarketClient()

# Daily data for the year
candles = client.get_ohlcv(
    "AAPL",
    interval=Interval.DAY_1,
    start=date(2024, 1, 1),
    end=date(2024, 12, 31)
)

# Convert to DataFrame
df = candles.to_df()
print(df.tail())

# Calculate returns
df['returns'] = df['close'].pct_change()
print(f"YTD Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
```

### Intraday Data

```python
from qfclient import MarketClient, Interval

client = MarketClient()

# 5-minute candles (last 100)
candles = client.get_ohlcv(
    "SPY",
    interval=Interval.MIN_5,
    limit=100
)

df = candles.to_df()
print(df.tail(10))
```

---

## Portfolio Analysis

### Analyze a Portfolio

```python
from qfclient import MarketClient
from datetime import date

client = MarketClient()

# Portfolio holdings
portfolio = {
    "AAPL": 100,
    "GOOGL": 50,
    "MSFT": 75,
    "AMZN": 25,
    "NVDA": 40
}

# Get current quotes
quotes = client.get_quotes_batch(list(portfolio.keys()))

# Calculate portfolio value
total_value = 0
print("Portfolio Summary:")
print("-" * 50)

for symbol, shares in portfolio.items():
    quote = quotes[symbol]
    value = quote.price * shares
    total_value += value
    print(f"{symbol}: {shares} shares @ ${quote.price:.2f} = ${value:,.2f}")

print("-" * 50)
print(f"Total Value: ${total_value:,.2f}")
```

### Portfolio Historical Performance

```python
from qfclient import MarketClient
from datetime import date
import pandas as pd

client = MarketClient()

portfolio = {"AAPL": 100, "GOOGL": 50, "MSFT": 75}
start_date = date(2024, 1, 1)

# Get historical data for all holdings
ohlcv_data = client.get_ohlcv_batch(
    list(portfolio.keys()),
    start=start_date
)

# Combine into portfolio value
dfs = []
for symbol, candles in ohlcv_data.items():
    df = candles.to_df()[['timestamp', 'close']].copy()
    df = df.rename(columns={'close': symbol})
    df = df.set_index('timestamp')
    dfs.append(df)

# Merge all price series
prices = pd.concat(dfs, axis=1)

# Calculate portfolio value over time
portfolio_value = sum(
    prices[symbol] * shares
    for symbol, shares in portfolio.items()
)

print(f"Start Value: ${portfolio_value.iloc[0]:,.2f}")
print(f"End Value: ${portfolio_value.iloc[-1]:,.2f}")
print(f"Return: {(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100:.2f}%")
```

### Get Company Fundamentals

```python
from qfclient import MarketClient

client = MarketClient()

symbols = ["AAPL", "MSFT", "GOOGL"]
profiles = client.get_profiles_batch(symbols)

for symbol, profile in profiles.items():
    print(f"\n{profile.name} ({symbol})")
    print(f"  Sector: {profile.sector}")
    print(f"  Industry: {profile.industry}")
    print(f"  Market Cap: ${profile.market_cap:,.0f}")
    print(f"  Employees: {profile.employees:,}")
```

---

## Options Analysis

### Options Chain Analysis

```python
from qfclient import MarketClient

client = MarketClient()

# Get available expirations
expirations = client.get_option_expirations("AAPL")
print(f"Available expirations: {len(expirations)}")

# Get nearest expiration chain
chain = client.get_options_chain("AAPL", expiration=expirations[0])

print(f"\nUnderlying Price: ${chain.underlying_price:.2f}")
print(f"Expiration: {chain.expiration}")

# Find ATM options
atm_strike = round(chain.underlying_price / 5) * 5  # Round to nearest $5

print(f"\nNear-ATM Calls (strike ~${atm_strike}):")
for call in chain.calls:
    if abs(call.strike - atm_strike) <= 10:
        print(f"  ${call.strike}: Bid=${call.bid:.2f}, Ask=${call.ask:.2f}, "
              f"Delta={call.delta:.3f}, IV={call.implied_volatility:.1%}")

print(f"\nNear-ATM Puts (strike ~${atm_strike}):")
for put in chain.puts:
    if abs(put.strike - atm_strike) <= 10:
        print(f"  ${put.strike}: Bid=${put.bid:.2f}, Ask=${put.ask:.2f}, "
              f"Delta={put.delta:.3f}, IV={put.implied_volatility:.1%}")
```

### Find High IV Options

```python
from qfclient import MarketClient

client = MarketClient()

chain = client.get_options_chain("AAPL")

# Sort calls by IV
high_iv_calls = sorted(
    [c for c in chain.calls if c.implied_volatility],
    key=lambda x: x.implied_volatility,
    reverse=True
)[:5]

print("Highest IV Calls:")
for call in high_iv_calls:
    print(f"  Strike ${call.strike}: IV={call.implied_volatility:.1%}, "
          f"Delta={call.delta:.3f}")
```

---

## Crypto Portfolio

### Track Crypto Holdings

```python
from qfclient import CryptoClient

client = CryptoClient()

holdings = {
    "BTC": 0.5,
    "ETH": 5.0,
    "SOL": 100,
    "DOGE": 10000
}

quotes = client.get_quotes_batch(list(holdings.keys()))

print("Crypto Portfolio:")
print("-" * 60)

total_usd = 0
for symbol, amount in holdings.items():
    quote = quotes[symbol]
    value = quote.price_usd * amount
    total_usd += value
    change = quote.percent_change_24h
    arrow = "↑" if change > 0 else "↓"
    print(f"{symbol}: {amount} @ ${quote.price_usd:,.2f} = ${value:,.2f} "
          f"({arrow} {abs(change):.2f}%)")

print("-" * 60)
print(f"Total: ${total_usd:,.2f}")
```

### Market Overview

```python
from qfclient import CryptoClient

client = CryptoClient()

# Global market data
global_data = client.get_global_market()
print("Global Crypto Market:")
print(f"  Total Market Cap: ${global_data['total_market_cap_usd']:,.0f}")
print(f"  24h Volume: ${global_data['total_volume_24h_usd']:,.0f}")
print(f"  BTC Dominance: {global_data['btc_dominance']:.1f}%")
print(f"  ETH Dominance: {global_data['eth_dominance']:.1f}%")

# Top 10 coins
print("\nTop 10 by Market Cap:")
top_coins = client.get_top_coins(limit=10)
for i, coin in enumerate(top_coins, 1):
    print(f"  {i}. {coin.symbol}: ${coin.price_usd:,.2f} "
          f"(MCap: ${coin.market_cap:,.0f})")

# Trending
print("\nTrending (24h):")
trending = client.get_trending()
for coin in trending[:5]:
    print(f"  {coin['symbol']}: Rank #{coin['market_cap_rank']}")
```

---

## Economic Dashboard

### Interest Rate Monitor

```python
from qfclient import MarketClient
from datetime import date

client = MarketClient()

print("Interest Rates (Last 12 Months):")
print("-" * 40)

# Fed Funds Rate
fed_rate = client.get_economic_indicator("FEDFUNDS", limit=12)
print("\nFederal Funds Rate:")
for r in fed_rate[-6:]:  # Last 6
    print(f"  {r.date}: {r.value}%")

# 10-Year Treasury
t10 = client.get_economic_indicator("DGS10", limit=30)
print(f"\n10-Year Treasury: {t10[-1].value}%")

# 2-Year Treasury
t2 = client.get_economic_indicator("DGS2", limit=30)
print(f"2-Year Treasury: {t2[-1].value}%")

# Yield curve (10Y - 2Y spread)
spread = float(t10[-1].value) - float(t2[-1].value)
status = "Normal" if spread > 0 else "Inverted"
print(f"Yield Curve Spread: {spread:.2f}% ({status})")
```

### Economic Indicators Dashboard

```python
from qfclient import MarketClient

client = MarketClient()

indicators = {
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI (Inflation)",
    "GDP": "GDP",
    "PAYEMS": "Nonfarm Payrolls (thousands)",
}

print("Economic Dashboard:")
print("=" * 50)

for series_id, name in indicators.items():
    data = client.get_economic_indicator(series_id, limit=1)
    if data:
        latest = data[0]
        print(f"{name}:")
        print(f"  Value: {latest.value}")
        print(f"  Date: {latest.date}")
        print()
```

---

## Async High-Performance

### Fetch Many Symbols Concurrently

```python
import asyncio
from qfclient import AsyncMarketClient

async def main():
    # S&P 500 sample
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "V", "XOM", "JPM", "WMT", "PG",
        "MA", "HD", "CVX", "MRK", "ABBV"
    ]

    async with AsyncMarketClient() as client:
        # Fetch all quotes concurrently
        quotes = await client.get_quotes_batch(symbols)

        # Process results
        for symbol, quote in sorted(quotes.items()):
            if hasattr(quote, 'price'):
                print(f"{symbol}: ${quote.price:.2f}")
            else:
                print(f"{symbol}: Error - {quote}")

asyncio.run(main())
```

### Parallel OHLCV Fetch

```python
import asyncio
from qfclient import AsyncMarketClient, Interval
from datetime import date

async def main():
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

    async with AsyncMarketClient() as client:
        ohlcv_data = await client.get_ohlcv_batch(
            symbols,
            interval=Interval.DAY_1,
            start=date(2024, 1, 1)
        )

        for symbol, candles in ohlcv_data.items():
            if hasattr(candles, 'to_df'):
                df = candles.to_df()
                returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                print(f"{symbol}: {len(df)} candles, YTD return: {returns:.2f}%")

asyncio.run(main())
```

---

## Data Export

### Export to CSV

```python
from qfclient import MarketClient
from datetime import date

client = MarketClient()

# Get data
candles = client.get_ohlcv("AAPL", start=date(2024, 1, 1))
df = candles.to_df()

# Export to CSV
df.to_csv("aapl_2024.csv", index=False)
print(f"Exported {len(df)} rows to aapl_2024.csv")
```

### Export to JSON

```python
from qfclient import MarketClient

client = MarketClient()

# Get company profile
profile = client.get_company_profile("AAPL")

# Export to JSON
import json
with open("aapl_profile.json", "w") as f:
    json.dump(profile.model_dump(), f, indent=2, default=str)

# For list results
candles = client.get_ohlcv("AAPL", limit=10)
json_str = candles.to_json()  # Built-in method
with open("aapl_candles.json", "w") as f:
    f.write(json_str)
```

### Export Multiple Symbols to Excel

```python
from qfclient import MarketClient
from datetime import date
import pandas as pd

client = MarketClient()

symbols = ["AAPL", "GOOGL", "MSFT"]
start = date(2024, 1, 1)

# Get all data
all_data = client.get_ohlcv_batch(symbols, start=start)

# Export each to a sheet
with pd.ExcelWriter("stock_data.xlsx") as writer:
    for symbol, candles in all_data.items():
        df = candles.to_df()
        df.to_excel(writer, sheet_name=symbol, index=False)

print("Exported to stock_data.xlsx")
```

---

## Error Handling

### Robust Data Fetching

```python
from qfclient import MarketClient, ProviderError, RateLimitError
import time

client = MarketClient()

def get_quote_safe(symbol: str, max_retries: int = 3) -> dict:
    """Get quote with retry logic."""
    for attempt in range(max_retries):
        try:
            quote = client.get_quote(symbol)
            return {
                "symbol": symbol,
                "price": quote.price,
                "success": True
            }
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait = e.retry_after or 60
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return {"symbol": symbol, "error": "Rate limited", "success": False}
        except ProviderError as e:
            return {"symbol": symbol, "error": str(e), "success": False}

# Usage
result = get_quote_safe("AAPL")
if result["success"]:
    print(f"{result['symbol']}: ${result['price']}")
else:
    print(f"{result['symbol']}: Failed - {result['error']}")
```

### Batch with Error Handling

```python
from qfclient import MarketClient, ProviderError

client = MarketClient()

symbols = ["AAPL", "INVALID", "GOOGL", "FAKE123", "MSFT"]
quotes = client.get_quotes_batch(symbols)

successful = []
failed = []

for symbol, result in quotes.items():
    if isinstance(result, ProviderError):
        failed.append((symbol, str(result)))
    else:
        successful.append((symbol, result.price))

print(f"Successful: {len(successful)}")
for symbol, price in successful:
    print(f"  {symbol}: ${price:.2f}")

print(f"\nFailed: {len(failed)}")
for symbol, error in failed:
    print(f"  {symbol}: {error}")
```
