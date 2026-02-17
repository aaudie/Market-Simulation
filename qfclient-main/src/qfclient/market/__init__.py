"""
Market data module for qfclient.

Provides access to stock, ETF, options, and economic data.
"""

from .models import (
    Quote,
    OHLCV,
    CompanyProfile,
    FinancialStatement,
    EarningsEvent,
    OptionContract,
    OptionChain,
    EconomicIndicator,
    COMMON_SERIES,
)

from .providers import (
    BaseProvider,
    AlpacaProvider,
    FinnhubProvider,
    FMPProvider,
    FREDProvider,
    TradierProvider,
    TwelveDataProvider,
    YFinanceProvider,
    PROVIDERS,
    get_provider,
    get_configured_providers,
)

__all__ = [
    # Models
    "Quote",
    "OHLCV",
    "CompanyProfile",
    "FinancialStatement",
    "EarningsEvent",
    "OptionContract",
    "OptionChain",
    "EconomicIndicator",
    "COMMON_SERIES",
    # Providers
    "BaseProvider",
    "AlpacaProvider",
    "FinnhubProvider",
    "FMPProvider",
    "FREDProvider",
    "TradierProvider",
    "TwelveDataProvider",
    "YFinanceProvider",
    "PROVIDERS",
    "get_provider",
    "get_configured_providers",
]
