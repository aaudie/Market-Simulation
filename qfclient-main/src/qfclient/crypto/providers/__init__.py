"""
Cryptocurrency data providers.
"""

from .base import BaseCryptoProvider
from .coingecko import CoinGeckoProvider
from .coinmarketcap import CoinMarketCapProvider
from .cryptocompare import CryptoCompareProvider
from .messari import MessariProvider

__all__ = [
    "BaseCryptoProvider",
    "CoinGeckoProvider",
    "CoinMarketCapProvider",
    "CryptoCompareProvider",
    "MessariProvider",
]

# Provider registry
PROVIDERS = {
    "coingecko": CoinGeckoProvider,
    "coinmarketcap": CoinMarketCapProvider,
    "cryptocompare": CryptoCompareProvider,
    "messari": MessariProvider,
}


def get_provider(name: str, **kwargs) -> BaseCryptoProvider:
    """
    Get a crypto provider instance by name.

    Args:
        name: Provider name (e.g., "coingecko", "coinmarketcap")
        **kwargs: Additional arguments for the provider

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[name](**kwargs)


def get_configured_providers() -> list[BaseCryptoProvider]:
    """Get all properly configured providers."""
    configured = []
    for name, provider_class in PROVIDERS.items():
        provider = provider_class()
        if provider.is_configured():
            configured.append(provider)
    return configured
