"""
Cryptocurrency data module for qfclient.

Provides access to cryptocurrency quotes, OHLCV, and market data.
"""

from .models import (
    CryptoQuote,
    CryptoOHLCV,
    CryptoAsset,
    CryptoMarketData,
)

from .providers import (
    BaseCryptoProvider,
    CoinGeckoProvider,
    CoinMarketCapProvider,
    PROVIDERS,
    get_provider,
    get_configured_providers,
)

__all__ = [
    # Models
    "CryptoQuote",
    "CryptoOHLCV",
    "CryptoAsset",
    "CryptoMarketData",
    # Providers
    "BaseCryptoProvider",
    "CoinGeckoProvider",
    "CoinMarketCapProvider",
    "PROVIDERS",
    "get_provider",
    "get_configured_providers",
]
