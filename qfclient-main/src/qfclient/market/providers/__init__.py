"""
Market data providers.

Each provider implements the BaseProvider interface and provides
access to specific data capabilities (quotes, OHLCV, options, etc.).
"""

from .base import BaseProvider
from .alpaca import AlpacaProvider
from .alpha_vantage import AlphaVantageProvider
from .eodhd import EODHDProvider
from .finnhub import FinnhubProvider
from .fmp import FMPProvider
from .fred import FREDProvider
from .marketstack import MarketstackProvider
from .polygon import PolygonProvider
from .sec import SECProvider
from .tiingo import TiingoProvider
from .tradier import TradierProvider
from .twelve_data import TwelveDataProvider
from .yfinance import YFinanceProvider

__all__ = [
    "BaseProvider",
    "AlpacaProvider",
    "AlphaVantageProvider",
    "EODHDProvider",
    "FinnhubProvider",
    "FMPProvider",
    "FREDProvider",
    "MarketstackProvider",
    "PolygonProvider",
    "SECProvider",
    "TiingoProvider",
    "TradierProvider",
    "TwelveDataProvider",
    "YFinanceProvider",
]

# Provider registry for easy lookup
PROVIDERS = {
    "alpaca": AlpacaProvider,
    "alpha_vantage": AlphaVantageProvider,
    "eodhd": EODHDProvider,
    "finnhub": FinnhubProvider,
    "fmp": FMPProvider,
    "fred": FREDProvider,
    "marketstack": MarketstackProvider,
    "polygon": PolygonProvider,
    "sec": SECProvider,
    "tiingo": TiingoProvider,
    "tradier": TradierProvider,
    "twelve_data": TwelveDataProvider,
    "yfinance": YFinanceProvider,
}


def get_provider(name: str, **kwargs) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name (e.g., "alpaca", "finnhub")
        **kwargs: Additional arguments to pass to the provider constructor

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[name](**kwargs)


def get_configured_providers() -> list[BaseProvider]:
    """
    Get all providers that are properly configured (have API keys, etc.).

    Returns:
        List of configured provider instances
    """
    configured = []
    for name, provider_class in PROVIDERS.items():
        provider = provider_class()
        if provider.is_configured():
            configured.append(provider)
    return configured
