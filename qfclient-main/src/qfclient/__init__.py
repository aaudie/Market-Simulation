"""
qfclient - Quantitative Finance Data Client

A Python library for fetching market and cryptocurrency data from multiple providers.

Features:
- Unified API across multiple data providers
- Automatic provider selection and failover
- Rate limiting with dynamic header parsing
- Strict Pydantic types with DataFrame conversion support
- Async support for high-performance concurrent operations

Usage:
    from qfclient import MarketClient, CryptoClient

    # Market data (sync)
    market = MarketClient()
    quote = market.get_quote("AAPL")
    candles = market.get_ohlcv("AAPL", start=date(2024, 1, 1))
    df = candles.to_df()  # Convert to DataFrame

    # Crypto data (sync)
    crypto = CryptoClient()
    btc = crypto.get_quote("BTC")
    top_coins = crypto.get_top_coins(limit=10)

    # Async usage for concurrent requests
    from qfclient import AsyncMarketClient, AsyncCryptoClient

    async with AsyncMarketClient() as client:
        quotes = await client.get_quotes_batch(["AAPL", "GOOGL", "MSFT"])
"""

# Auto-load .env file before any providers are imported
from dotenv import load_dotenv

load_dotenv()  # Searches for .env in current dir and parents

__version__ = "0.1.0"

# Main clients (sync and async)
from .client import MarketClient, CryptoClient, AsyncMarketClient, AsyncCryptoClient

# Common types
from .common import (
    ResultList,
    Interval,
    AssetType,
    ProviderError,
    RateLimitError,
    ConfigurationError,
    RateLimiter,
    get_limiter,
)

# Market models
from .market.models import (
    Quote,
    OHLCV,
    CompanyProfile,
    EarningsEvent,
    OptionContract,
    OptionChain,
    EconomicIndicator,
    # SEC Form 4 models
    TransactionCode,
    OwnershipType,
    InsiderRoleType,
    InsiderRole,
    SECInsiderTransaction,
    Form4Filing,
    InsiderSummary,
)

# Crypto models
from .crypto.models import (
    CryptoQuote,
    CryptoOHLCV,
    CryptoAsset,
    CryptoMarketData,
)

# CLI/diagnostic functions
from .cli import list_providers, run_diagnostics

__all__ = [
    # Version
    "__version__",
    # Clients (sync)
    "MarketClient",
    "CryptoClient",
    # Clients (async)
    "AsyncMarketClient",
    "AsyncCryptoClient",
    # Common
    "ResultList",
    "Interval",
    "AssetType",
    "ProviderError",
    "RateLimitError",
    "ConfigurationError",
    "RateLimiter",
    "get_limiter",
    # Market models
    "Quote",
    "OHLCV",
    "CompanyProfile",
    "EarningsEvent",
    "OptionContract",
    "OptionChain",
    "EconomicIndicator",
    # SEC Form 4 models
    "TransactionCode",
    "OwnershipType",
    "InsiderRoleType",
    "InsiderRole",
    "SECInsiderTransaction",
    "Form4Filing",
    "InsiderSummary",
    # Crypto models
    "CryptoQuote",
    "CryptoOHLCV",
    "CryptoAsset",
    "CryptoMarketData",
    # CLI/diagnostics
    "list_providers",
    "run_diagnostics",
]
