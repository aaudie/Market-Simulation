"""
Main client interfaces for qfclient.

Provides high-level access to market and crypto data with:
- Automatic provider selection and failover
- Rate limiting
- Consistent return types (ResultList with .to_df() support)
- Batch operations for fetching multiple symbols
- Async support for high-performance concurrent requests
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import TypeVar

from .common.base import ResultList, ProviderError, RateLimitError
from .common.types import Interval
from .common.rate_limiter import get_limiter

# Market imports
from .market.models import (
    Quote, OHLCV, CompanyProfile, EarningsEvent,
    OptionChain, EconomicIndicator, NewsArticle,
    InsiderTransaction, AnalystRecommendation, PriceTarget,
    Dividend, StockSplit, FinancialStatement,
    # SEC Form 4 models
    SECInsiderTransaction, Form4Filing, InsiderSummary,
)
from .market.providers import (
    PROVIDERS as MARKET_PROVIDERS,
    get_configured_providers as get_configured_market_providers,
    SECProvider,
)

# Crypto imports
from .crypto.models import (
    CryptoQuote, CryptoOHLCV, CryptoAsset, CryptoMarketData,
)
from .crypto.providers import (
    PROVIDERS as CRYPTO_PROVIDERS,
    get_configured_providers as get_configured_crypto_providers,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MarketClient:
    """
    High-level client for market data (stocks, ETFs, options, economic).

    Automatically handles:
    - Provider selection based on availability and rate limits
    - Failover between providers
    - Rate limiting

    Usage:
        client = MarketClient()

        # Get a quote
        quote = client.get_quote("AAPL")
        print(f"AAPL: ${quote.price}")

        # Get OHLCV data
        candles = client.get_ohlcv("AAPL", start=date(2024, 1, 1))
        df = candles.to_df()  # Convert to DataFrame

        # Get company profile
        profile = client.get_company_profile("AAPL")
        print(f"{profile.name}: {profile.sector}")
    """

    def __init__(self, prefer: str | None = None):
        """
        Initialize the market client.

        Args:
            prefer: Preferred provider name (e.g., "alpaca", "finnhub").
                   If not available, will failover to others.
        """
        self.prefer = prefer
        self._providers = {}
        self._limiter = get_limiter()

    def _get_provider(self, name: str):
        """Get or create a provider instance."""
        if name not in self._providers:
            if name not in MARKET_PROVIDERS:
                raise ValueError(f"Unknown provider: {name}")
            self._providers[name] = MARKET_PROVIDERS[name]()
        return self._providers[name]

    def _select_provider(self, capability: str, prefer: str | None = None, exclude: list[str] | None = None):
        """Select the best available provider for a capability."""
        prefer = prefer or self.prefer
        exclude = exclude or []

        # Get providers that support this capability
        capable_providers = []
        for name, provider_class in MARKET_PROVIDERS.items():
            if name in exclude:
                continue
            provider = self._get_provider(name)
            if provider.is_configured():
                # Check if provider supports the capability
                supports = getattr(provider, f"supports_{capability}", False)
                if supports:
                    capable_providers.append(name)

        if not capable_providers:
            raise ProviderError("none", f"No providers configured for {capability}")

        # Use rate limiter to select best available
        selected = self._limiter.select_provider(capable_providers, prefer)

        if not selected:
            raise ProviderError("none", f"All providers rate limited for {capability}")

        return self._get_provider(selected)

    def _call_with_failover(self, capability: str, method: str, *args, prefer: str | None = None, **kwargs):
        """Call a provider method with automatic failover on failure."""
        tried_providers = []
        last_error = None

        for attempt in range(3):  # Try up to 3 providers
            try:
                provider = self._select_provider(capability, prefer, exclude=tried_providers)
                tried_providers.append(provider.provider_name)
                func = getattr(provider, method)
                result = func(*args, **kwargs)
                return result
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} rate limited, trying another...")
                continue
            except ProviderError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} failed: {e}, trying another...")
                continue

        # All providers failed
        raise last_error or ProviderError("none", f"All providers failed for {capability}")

    def get_quote(self, symbol: str, prefer: str | None = None) -> Quote:
        """
        Get a real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            Quote object with price, bid, ask, etc.
        """
        provider = self._select_provider("quotes", prefer)
        return provider.get_quote(symbol)

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[OHLCV]:
        """
        Get OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval (1m, 5m, 1h, 1d, etc.)
            start: Start date
            end: End date
            limit: Maximum number of candles
            prefer: Preferred provider

        Returns:
            ResultList of OHLCV candles (use .to_df() for DataFrame)
        """
        provider = self._select_provider("ohlcv", prefer)
        return provider.get_ohlcv(symbol, interval, start, end, limit)

    def get_company_profile(self, symbol: str, prefer: str | None = None) -> CompanyProfile:
        """
        Get company profile and fundamental information.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            CompanyProfile with name, sector, market cap, etc.
        """
        provider = self._select_provider("company_profile", prefer)
        return provider.get_company_profile(symbol)

    def get_earnings(
        self,
        symbol: str | None = None,
        start: date | None = None,
        end: date | None = None,
        prefer: str | None = None,
    ) -> ResultList[EarningsEvent]:
        """
        Get earnings calendar.

        Args:
            symbol: Filter by symbol (optional)
            start: Start date for calendar
            end: End date for calendar
            prefer: Preferred provider

        Returns:
            ResultList of EarningsEvent
        """
        provider = self._select_provider("earnings", prefer)
        return provider.get_earnings(symbol, start, end)

    def get_option_expirations(
        self,
        symbol: str,
        prefer: str | None = None,
    ) -> list[date]:
        """
        Get available option expiration dates for a symbol.

        Args:
            symbol: Underlying stock symbol
            prefer: Preferred provider

        Returns:
            List of expiration dates
        """
        provider = self._select_provider("options", prefer)
        return provider.get_expirations(symbol)

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
        prefer: str | None = None,
    ) -> OptionChain:
        """
        Get options chain for a symbol.

        Args:
            symbol: Underlying stock symbol
            expiration: Option expiration date (uses nearest if not specified)
            prefer: Preferred provider

        Returns:
            OptionChain with calls, puts, and Greeks (delta, gamma, theta, vega)

        Example:
            chain = client.get_options_chain("AAPL")
            for call in chain.calls:
                print(f"Strike: {call.strike}, Delta: {call.delta}, IV: {call.implied_volatility}")
        """
        provider = self._select_provider("options", prefer)
        return provider.get_options_chain(symbol, expiration)

    def get_economic_indicator(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
        limit: int | None = None,
        prefer: str | None = None,
    ) -> ResultList[EconomicIndicator]:
        """
        Get economic indicator data (e.g., FEDFUNDS, GDP, UNRATE).

        Args:
            series_id: FRED series ID (see COMMON_SERIES for popular IDs)
            start: Start date
            end: End date
            limit: Maximum number of observations (most recent first)
            prefer: Preferred provider

        Returns:
            ResultList of EconomicIndicator observations

        Example:
            # Get last 12 months of Fed Funds rate
            rates = client.get_economic_indicator("FEDFUNDS", limit=12)

            # Get unemployment since 2020
            unemployment = client.get_economic_indicator("UNRATE", start=date(2020, 1, 1))
        """
        provider = self._select_provider("economic", prefer)
        return provider.get_economic_indicator(series_id, start, end, limit)

    def get_news(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
        prefer: str | None = None,
    ) -> ResultList[NewsArticle]:
        """
        Get news articles for a symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date for news (default: 7 days ago)
            end: End date for news (default: today)
            limit: Maximum number of articles to return
            prefer: Preferred provider

        Returns:
            ResultList of NewsArticle

        Example:
            news = client.get_news("AAPL", limit=10)
            for article in news:
                print(f"{article.published_at}: {article.headline}")
        """
        return self._call_with_failover(
            "news", "get_news", symbol, start, end, limit, prefer=prefer
        )

    def get_insider_transactions(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[InsiderTransaction]:
        """
        Get insider transactions (Form 4 filings) for a symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date filter
            end: End date filter
            limit: Maximum transactions to return
            prefer: Preferred provider

        Returns:
            ResultList of InsiderTransaction

        Example:
            transactions = client.get_insider_transactions("AAPL")
            for tx in transactions:
                print(f"{tx.insider_name}: {tx.transaction_type} {tx.shares} shares @ ${tx.price}")
        """
        return self._call_with_failover(
            "insider_transactions", "get_insider_transactions", symbol, start, end, limit, prefer=prefer
        )

    # ==================== SEC Form 4 Methods ====================

    def get_sec_filings(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
    ) -> ResultList[Form4Filing]:
        """
        Get SEC Form 4 filings with rich insider data.

        Fetches complete Form 4 filings directly from SEC EDGAR with
        comprehensive data including:
        - All transactions (purchases, sales, grants, exercises)
        - Position changes (shares before/after, % change)
        - Insider role classification (CEO, CFO, Director, 10% owner)

        Args:
            symbol: Stock ticker symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)
            limit: Maximum filings to return

        Returns:
            ResultList of Form4Filing

        Example:
            filings = client.get_sec_filings("AAPL", limit=10)
            for f in filings:
                print(f"{f.owner_name} ({f.role.role_type.value})")
                print(f"  Net shares: {f.net_shares:+,.0f}")
                print(f"  Net value: ${f.net_value:+,.2f}")
                for t in f.transactions:
                    if t.position_change_pct:
                        print(f"    Position change: {t.position_change_pct:+.1f}%")
        """
        sec = self._get_provider("sec")
        return sec.get_form4_filings(symbol, start, end, limit)

    def get_sec_transactions(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[SECInsiderTransaction]:
        """
        Get SEC Form 4 transactions with rich metadata.

        Returns individual transactions from Form 4 filings with
        detailed data including:
        - Transaction details (shares, price, date, code)
        - Position change calculations (shares before/after, % change)
        - Insider role classification (CEO, CFO, Director, etc.)
        - Open market vs derivative classification

        Args:
            symbol: Stock ticker symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)
            limit: Maximum transactions to return

        Returns:
            ResultList of SECInsiderTransaction

        Example:
            transactions = client.get_sec_transactions("AAPL")
            for t in transactions:
                print(f"{t.owner_name} ({t.role.role_type.value})")
                print(f"  {t.transaction_code.value}: {t.shares:,.0f} @ ${t.price_per_share:.2f}")
                if t.position_change_pct:
                    print(f"  Position change: {t.position_change_pct:+.1f}%")
        """
        sec = self._get_provider("sec")
        return sec.get_insider_transactions(symbol, start, end, limit)

    def get_insider_summary(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> InsiderSummary:
        """
        Get aggregated insider trading summary for a symbol.

        Computes summary statistics from Form 4 filings including:
        - Net Purchase Ratio (NPR) - Lakonishok & Lee 2001
        - Buyer/seller counts
        - Total shares and value traded
        - Insider sentiment classification

        Args:
            symbol: Stock ticker symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)

        Returns:
            InsiderSummary with aggregated statistics

        Example:
            summary = client.get_insider_summary("AAPL")
            print(f"Unique insiders: {summary.unique_insiders}")
            print(f"Buyers: {summary.num_buyers}, Sellers: {summary.num_sellers}")
            print(f"NPR: {summary.net_purchase_ratio:.2f}")
            print(f"Sentiment: {summary.insider_sentiment}")
        """
        sec = self._get_provider("sec")
        return sec.get_insider_summary(symbol, start, end)

    def get_recommendations(
        self,
        symbol: str,
        prefer: str | None = None,
    ) -> ResultList[AnalystRecommendation]:
        """
        Get analyst recommendation trends for a symbol.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            ResultList of AnalystRecommendation (aggregated by period)

        Example:
            recs = client.get_recommendations("AAPL")
            latest = recs[0]
            print(f"Buy: {latest.buy}, Hold: {latest.hold}, Sell: {latest.sell}")
        """
        return self._call_with_failover(
            "recommendations", "get_recommendations", symbol, prefer=prefer
        )

    def get_price_target(
        self,
        symbol: str,
        prefer: str | None = None,
    ) -> PriceTarget:
        """
        Get analyst price targets for a symbol.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            PriceTarget with high, low, mean, median targets

        Example:
            target = client.get_price_target("AAPL")
            print(f"Target range: ${target.target_low} - ${target.target_high}")
            print(f"Mean target: ${target.target_mean} ({target.num_analysts} analysts)")
        """
        return self._call_with_failover(
            "recommendations", "get_price_target", symbol, prefer=prefer
        )

    def get_dividends(
        self,
        symbol: str,
        prefer: str | None = None,
    ) -> ResultList[Dividend]:
        """
        Get dividend history for a symbol.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            ResultList of Dividend

        Example:
            divs = client.get_dividends("AAPL")
            for div in divs[-4:]:  # Last 4 dividends
                print(f"{div.ex_date}: ${div.amount}")
        """
        return self._call_with_failover(
            "dividends", "get_dividends", symbol, prefer=prefer
        )

    def get_stock_splits(
        self,
        symbol: str,
        prefer: str | None = None,
    ) -> ResultList[StockSplit]:
        """
        Get stock split history for a symbol.

        Args:
            symbol: Stock ticker symbol
            prefer: Preferred provider

        Returns:
            ResultList of StockSplit

        Example:
            splits = client.get_stock_splits("AAPL")
            for split in splits:
                print(f"{split.split_date}: {split.ratio}:1 split")
        """
        return self._call_with_failover(
            "dividends", "get_stock_splits", symbol, prefer=prefer
        )

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        prefer: str | None = None,
    ) -> ResultList[FinancialStatement]:
        """
        Get income statement data for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return
            prefer: Preferred provider

        Returns:
            ResultList of FinancialStatement with income data

        Example:
            income = client.get_income_statement("AAPL", period="annual", limit=5)
            for stmt in income:
                print(f"{stmt.fiscal_date}: Revenue=${stmt.data['revenue']:,}")
        """
        return self._call_with_failover(
            "financials", "get_income_statement", symbol, period, limit, prefer=prefer
        )

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        prefer: str | None = None,
    ) -> ResultList[FinancialStatement]:
        """
        Get balance sheet data for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return
            prefer: Preferred provider

        Returns:
            ResultList of FinancialStatement with balance sheet data

        Example:
            balance = client.get_balance_sheet("AAPL", period="annual", limit=5)
            for stmt in balance:
                print(f"{stmt.fiscal_date}: Assets=${stmt.data['total_assets']:,}")
        """
        return self._call_with_failover(
            "financials", "get_balance_sheet", symbol, period, limit, prefer=prefer
        )

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        prefer: str | None = None,
    ) -> ResultList[FinancialStatement]:
        """
        Get cash flow statement data for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return
            prefer: Preferred provider

        Returns:
            ResultList of FinancialStatement with cash flow data

        Example:
            cash_flow = client.get_cash_flow("AAPL", period="annual", limit=5)
            for stmt in cash_flow:
                print(f"{stmt.fiscal_date}: FCF=${stmt.data['free_cash_flow']:,}")
        """
        return self._call_with_failover(
            "financials", "get_cash_flow", symbol, period, limit, prefer=prefer
        )

    def get_status(self) -> dict:
        """Get status of all configured providers."""
        status = {}
        for name in MARKET_PROVIDERS:
            provider = self._get_provider(name)
            status[name] = {
                "configured": provider.is_configured(),
                "rate_limit": self._limiter.get_status(name),
            }
        return status

    # ==================== Batch Methods ====================

    def get_quotes_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, Quote | ProviderError]:
        """
        Get quotes for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to Quote or ProviderError if failed
        """
        results = {}

        def fetch_quote(symbol: str):
            try:
                return symbol, self._call_with_failover("quotes", "get_quote", symbol, prefer=prefer)
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_quote, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_ohlcv_batch(
        self,
        symbols: list[str],
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, ResultList[OHLCV] | ProviderError]:
        """
        Get OHLCV data for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum candles per symbol
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to ResultList[OHLCV] or ProviderError if failed
        """
        results = {}

        def fetch_ohlcv(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "ohlcv", "get_ohlcv", symbol, interval, start, end, limit, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_ohlcv, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_profiles_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, CompanyProfile | ProviderError]:
        """
        Get company profiles for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to CompanyProfile or ProviderError if failed
        """
        results = {}

        def fetch_profile(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "company_profile", "get_company_profile", symbol, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_profile, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_news_batch(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
        limit: int = 10,
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, ResultList[NewsArticle] | ProviderError]:
        """
        Get news for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            start: Start date for news
            end: End date for news
            limit: Maximum articles per symbol
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to ResultList[NewsArticle] or ProviderError if failed
        """
        results = {}

        def fetch_news(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "news", "get_news", symbol, start, end, limit, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_news, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_dividends_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, ResultList[Dividend] | ProviderError]:
        """
        Get dividend history for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to ResultList[Dividend] or ProviderError if failed
        """
        results = {}

        def fetch_dividends(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "dividends", "get_dividends", symbol, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_dividends, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_recommendations_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, ResultList[AnalystRecommendation] | ProviderError]:
        """
        Get analyst recommendations for multiple symbols in parallel.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to ResultList[AnalystRecommendation] or ProviderError if failed
        """
        results = {}

        def fetch_recommendations(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "recommendations", "get_recommendations", symbol, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_recommendations, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results


class CryptoClient:
    """
    High-level client for cryptocurrency data.

    Automatically handles:
    - Provider selection based on availability and rate limits
    - Failover between providers
    - Rate limiting

    Usage:
        client = CryptoClient()

        # Get a quote
        quote = client.get_quote("BTC")
        print(f"Bitcoin: ${quote.price_usd}")

        # Get market data with more details
        data = client.get_market_data("ETH")
        print(f"ETH ATH: ${data.ath}")

        # Get top coins
        top = client.get_top_coins(limit=10)
        df = top.to_df()
    """

    def __init__(self, prefer: str | None = None):
        """
        Initialize the crypto client.

        Args:
            prefer: Preferred provider name (e.g., "coingecko", "coinmarketcap").
        """
        self.prefer = prefer
        self._providers = {}
        self._limiter = get_limiter()

    def _get_provider(self, name: str):
        """Get or create a provider instance."""
        if name not in self._providers:
            if name not in CRYPTO_PROVIDERS:
                raise ValueError(f"Unknown provider: {name}")
            self._providers[name] = CRYPTO_PROVIDERS[name]()
        return self._providers[name]

    def _select_provider(self, capability: str, prefer: str | None = None, exclude: list[str] | None = None):
        """Select the best available provider for a capability."""
        prefer = prefer or self.prefer
        exclude = exclude or []

        capable_providers = []
        for name, provider_class in CRYPTO_PROVIDERS.items():
            if name in exclude:
                continue
            provider = self._get_provider(name)
            if provider.is_configured():
                supports = getattr(provider, f"supports_{capability}", False)
                if supports:
                    capable_providers.append(name)

        if not capable_providers:
            raise ProviderError("none", f"No providers configured for {capability}")

        selected = self._limiter.select_provider(capable_providers, prefer)

        if not selected:
            raise ProviderError("none", f"All providers rate limited for {capability}")

        return self._get_provider(selected)

    def _call_with_failover(self, capability: str, method: str, *args, prefer: str | None = None, **kwargs):
        """Call a provider method with automatic failover on failure."""
        tried_providers = []
        last_error = None

        for attempt in range(3):  # Try up to 3 providers
            try:
                provider = self._select_provider(capability, prefer, exclude=tried_providers)
                tried_providers.append(provider.provider_name)
                func = getattr(provider, method)
                result = func(*args, **kwargs)
                return result
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} rate limited, trying another...")
                continue
            except ProviderError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} failed: {e}, trying another...")
                continue

        # All providers failed
        raise last_error or ProviderError("none", f"All providers failed for {capability}")

    def get_quote(self, symbol: str, prefer: str | None = None) -> CryptoQuote:
        """
        Get a real-time quote for a cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., BTC, ETH)
            prefer: Preferred provider

        Returns:
            CryptoQuote with price, market cap, volume, etc.
        """
        provider = self._select_provider("quotes", prefer)
        return provider.get_quote(symbol)

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[CryptoOHLCV]:
        """
        Get OHLCV (candlestick) data for a cryptocurrency.

        Args:
            symbol: Crypto symbol
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum number of candles
            prefer: Preferred provider

        Returns:
            ResultList of CryptoOHLCV candles
        """
        provider = self._select_provider("ohlcv", prefer)
        return provider.get_ohlcv(symbol, interval, start, end, limit)

    def get_asset(self, symbol: str, prefer: str | None = None) -> CryptoAsset:
        """
        Get asset profile and metadata.

        Args:
            symbol: Crypto symbol
            prefer: Preferred provider

        Returns:
            CryptoAsset with name, description, links, etc.
        """
        provider = self._select_provider("asset", prefer)
        return provider.get_asset(symbol)

    def get_market_data(self, symbol: str, prefer: str | None = None) -> CryptoMarketData:
        """
        Get comprehensive market data for a cryptocurrency.

        Args:
            symbol: Crypto symbol
            prefer: Preferred provider

        Returns:
            CryptoMarketData with price, ATH, supply, rankings, etc.
        """
        provider = self._select_provider("market_data", prefer)
        return provider.get_market_data(symbol)

    def get_top_coins(
        self,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[CryptoMarketData]:
        """
        Get top cryptocurrencies by market cap.

        Args:
            limit: Number of coins to return
            prefer: Preferred provider

        Returns:
            ResultList of CryptoMarketData
        """
        provider = self._select_provider("market_data", prefer)
        return provider.get_top_coins(limit)

    def get_trending(self) -> list[dict]:
        """
        Get trending cryptocurrencies (most searched in last 24 hours).

        Returns:
            List of trending coins with name, symbol, market_cap_rank, and score

        Example:
            trending = client.get_trending()
            for coin in trending:
                print(f"{coin['symbol']}: rank #{coin['market_cap_rank']}")
        """
        provider = self._get_provider("coingecko")
        return provider.get_trending()

    def get_global_market(self) -> dict:
        """
        Get global cryptocurrency market data.

        Returns:
            Dict with:
            - total_market_cap_usd: Total crypto market cap
            - total_volume_24h_usd: 24h trading volume
            - btc_dominance: Bitcoin market share %
            - eth_dominance: Ethereum market share %
            - active_cryptocurrencies: Number of active coins
            - market_cap_change_24h_pct: 24h market cap change

        Example:
            global_data = client.get_global_market()
            print(f"Total market cap: ${global_data['total_market_cap_usd']:,.0f}")
            print(f"BTC dominance: {global_data['btc_dominance']:.1f}%")
        """
        provider = self._get_provider("coingecko")
        return provider.get_global()

    def get_status(self) -> dict:
        """Get status of all configured providers."""
        status = {}
        for name in CRYPTO_PROVIDERS:
            provider = self._get_provider(name)
            status[name] = {
                "configured": provider.is_configured(),
                "rate_limit": self._limiter.get_status(name),
            }
        return status

    # ==================== Batch Methods ====================

    def get_quotes_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, CryptoQuote | ProviderError]:
        """
        Get quotes for multiple cryptocurrencies in parallel.

        Args:
            symbols: List of crypto symbols (e.g., ["BTC", "ETH", "SOL"])
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to CryptoQuote or ProviderError if failed
        """
        results = {}

        def fetch_quote(symbol: str):
            try:
                return symbol, self._call_with_failover("quotes", "get_quote", symbol, prefer=prefer)
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_quote, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results

    def get_ohlcv_batch(
        self,
        symbols: list[str],
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
        max_workers: int = 5,
    ) -> dict[str, ResultList[CryptoOHLCV] | ProviderError]:
        """
        Get OHLCV data for multiple cryptocurrencies in parallel.

        Args:
            symbols: List of crypto symbols
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum candles per symbol
            prefer: Preferred provider
            max_workers: Maximum parallel requests (default 5)

        Returns:
            Dict mapping symbol to ResultList[CryptoOHLCV] or ProviderError if failed
        """
        results = {}

        def fetch_ohlcv(symbol: str):
            try:
                return symbol, self._call_with_failover(
                    "ohlcv", "get_ohlcv", symbol, interval, start, end, limit, prefer=prefer
                )
            except ProviderError as e:
                return symbol, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_ohlcv, s): s for s in symbols}
            for future in as_completed(futures):
                symbol, result = future.result()
                results[symbol] = result

        return results


# ==================== Async Clients ====================


class AsyncMarketClient:
    """
    Async client for market data (stocks, ETFs, options, economic).

    Uses true async HTTP requests for high-performance concurrent operations.
    Ideal for fetching data for many symbols simultaneously.

    Usage:
        async with AsyncMarketClient() as client:
            # Fetch multiple quotes concurrently
            quotes = await client.get_quotes_batch(["AAPL", "GOOGL", "MSFT"])

            # Or use individual methods
            quote = await client.get_quote("AAPL")
    """

    def __init__(self, prefer: str | None = None):
        """
        Initialize the async market client.

        Args:
            prefer: Preferred provider name (e.g., "alpaca", "finnhub").
        """
        self.prefer = prefer
        self._providers = {}
        self._limiter = get_limiter()

    def _get_provider(self, name: str):
        """Get or create a provider instance."""
        if name not in self._providers:
            if name not in MARKET_PROVIDERS:
                raise ValueError(f"Unknown provider: {name}")
            self._providers[name] = MARKET_PROVIDERS[name]()
        return self._providers[name]

    def _select_provider(self, capability: str, prefer: str | None = None, exclude: list[str] | None = None):
        """Select the best available provider for a capability."""
        prefer = prefer or self.prefer
        exclude = exclude or []

        capable_providers = []
        for name in MARKET_PROVIDERS:
            if name in exclude:
                continue
            provider = self._get_provider(name)
            if provider.is_configured():
                supports = getattr(provider, f"supports_{capability}", False)
                if supports:
                    capable_providers.append(name)

        if not capable_providers:
            raise ProviderError("none", f"No providers configured for {capability}")

        selected = self._limiter.select_provider(capable_providers, prefer)

        if not selected:
            raise ProviderError("none", f"All providers rate limited for {capability}")

        return self._get_provider(selected)

    async def _call_with_failover_async(self, capability: str, method: str, *args, prefer: str | None = None, **kwargs):
        """Call a provider async method with automatic failover on failure."""
        tried_providers = []
        last_error = None

        for attempt in range(3):
            try:
                provider = self._select_provider(capability, prefer, exclude=tried_providers)
                tried_providers.append(provider.provider_name)
                func = getattr(provider, f"{method}_async", None)
                if func is None:
                    # Fall back to sync method if async not available
                    func = getattr(provider, method)
                    result = func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                return result
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} rate limited, trying another...")
                continue
            except ProviderError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} failed: {e}, trying another...")
                continue

        raise last_error or ProviderError("none", f"All providers failed for {capability}")

    async def get_quote(self, symbol: str, prefer: str | None = None) -> Quote:
        """Get a real-time quote for a symbol."""
        return await self._call_with_failover_async("quotes", "get_quote", symbol, prefer=prefer)

    async def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[OHLCV]:
        """Get OHLCV (candlestick) data for a symbol."""
        return await self._call_with_failover_async(
            "ohlcv", "get_ohlcv", symbol, interval, start, end, limit, prefer=prefer
        )

    async def get_company_profile(self, symbol: str, prefer: str | None = None) -> CompanyProfile:
        """Get company profile and fundamental information."""
        return await self._call_with_failover_async("company_profile", "get_company_profile", symbol, prefer=prefer)

    async def get_news(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
        prefer: str | None = None,
    ) -> ResultList[NewsArticle]:
        """Get news articles for a symbol."""
        return await self._call_with_failover_async("news", "get_news", symbol, start, end, limit, prefer=prefer)

    async def get_dividends(self, symbol: str, prefer: str | None = None) -> ResultList[Dividend]:
        """Get dividend history for a symbol."""
        return await self._call_with_failover_async("dividends", "get_dividends", symbol, prefer=prefer)

    async def get_recommendations(self, symbol: str, prefer: str | None = None) -> ResultList[AnalystRecommendation]:
        """Get analyst recommendation trends for a symbol."""
        return await self._call_with_failover_async("recommendations", "get_recommendations", symbol, prefer=prefer)

    # ==================== Async Batch Methods ====================

    async def get_quotes_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
    ) -> dict[str, Quote | ProviderError]:
        """
        Get quotes for multiple symbols concurrently using asyncio.gather.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to Quote or ProviderError if failed
        """
        async def fetch_quote(symbol: str):
            try:
                quote = await self.get_quote(symbol, prefer)
                return symbol, quote
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_quote(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def get_ohlcv_batch(
        self,
        symbols: list[str],
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> dict[str, ResultList[OHLCV] | ProviderError]:
        """
        Get OHLCV data for multiple symbols concurrently using asyncio.gather.

        Args:
            symbols: List of stock ticker symbols
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum candles per symbol
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to ResultList[OHLCV] or ProviderError if failed
        """
        async def fetch_ohlcv(symbol: str):
            try:
                ohlcv = await self.get_ohlcv(symbol, interval, start, end, limit, prefer)
                return symbol, ohlcv
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_ohlcv(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def get_profiles_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
    ) -> dict[str, CompanyProfile | ProviderError]:
        """
        Get company profiles for multiple symbols concurrently.

        Args:
            symbols: List of stock ticker symbols
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to CompanyProfile or ProviderError if failed
        """
        async def fetch_profile(symbol: str):
            try:
                profile = await self.get_company_profile(symbol, prefer)
                return symbol, profile
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_profile(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def get_news_batch(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
        limit: int = 10,
        prefer: str | None = None,
    ) -> dict[str, ResultList[NewsArticle] | ProviderError]:
        """
        Get news for multiple symbols concurrently.

        Args:
            symbols: List of stock ticker symbols
            start: Start date for news
            end: End date for news
            limit: Maximum articles per symbol
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to ResultList[NewsArticle] or ProviderError if failed
        """
        async def fetch_news(symbol: str):
            try:
                news = await self.get_news(symbol, start, end, limit, prefer)
                return symbol, news
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_news(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def aclose(self):
        """Close all provider async clients."""
        for provider in self._providers.values():
            await provider.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


class AsyncCryptoClient:
    """
    Async client for cryptocurrency data.

    Uses true async HTTP requests for high-performance concurrent operations.
    Ideal for fetching data for many symbols simultaneously.

    Usage:
        async with AsyncCryptoClient() as client:
            # Fetch multiple quotes concurrently
            quotes = await client.get_quotes_batch(["BTC", "ETH", "SOL"])

            # Or use individual methods
            quote = await client.get_quote("BTC")
    """

    def __init__(self, prefer: str | None = None):
        """
        Initialize the async crypto client.

        Args:
            prefer: Preferred provider name (e.g., "coingecko", "coinmarketcap").
        """
        self.prefer = prefer
        self._providers = {}
        self._limiter = get_limiter()

    def _get_provider(self, name: str):
        """Get or create a provider instance."""
        if name not in self._providers:
            if name not in CRYPTO_PROVIDERS:
                raise ValueError(f"Unknown provider: {name}")
            self._providers[name] = CRYPTO_PROVIDERS[name]()
        return self._providers[name]

    def _select_provider(self, capability: str, prefer: str | None = None, exclude: list[str] | None = None):
        """Select the best available provider for a capability."""
        prefer = prefer or self.prefer
        exclude = exclude or []

        capable_providers = []
        for name in CRYPTO_PROVIDERS:
            if name in exclude:
                continue
            provider = self._get_provider(name)
            if provider.is_configured():
                supports = getattr(provider, f"supports_{capability}", False)
                if supports:
                    capable_providers.append(name)

        if not capable_providers:
            raise ProviderError("none", f"No providers configured for {capability}")

        selected = self._limiter.select_provider(capable_providers, prefer)

        if not selected:
            raise ProviderError("none", f"All providers rate limited for {capability}")

        return self._get_provider(selected)

    async def _call_with_failover_async(self, capability: str, method: str, *args, prefer: str | None = None, **kwargs):
        """Call a provider async method with automatic failover on failure."""
        tried_providers = []
        last_error = None

        for attempt in range(3):
            try:
                provider = self._select_provider(capability, prefer, exclude=tried_providers)
                tried_providers.append(provider.provider_name)
                func = getattr(provider, f"{method}_async", None)
                if func is None:
                    func = getattr(provider, method)
                    result = func(*args, **kwargs)
                else:
                    result = await func(*args, **kwargs)
                return result
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} rate limited, trying another...")
                continue
            except ProviderError as e:
                last_error = e
                logger.warning(f"Provider {e.provider} failed: {e}, trying another...")
                continue

        raise last_error or ProviderError("none", f"All providers failed for {capability}")

    async def get_quote(self, symbol: str, prefer: str | None = None) -> CryptoQuote:
        """Get a real-time quote for a cryptocurrency."""
        return await self._call_with_failover_async("quotes", "get_quote", symbol, prefer=prefer)

    async def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> ResultList[CryptoOHLCV]:
        """Get OHLCV (candlestick) data for a cryptocurrency."""
        return await self._call_with_failover_async(
            "ohlcv", "get_ohlcv", symbol, interval, start, end, limit, prefer=prefer
        )

    async def get_asset(self, symbol: str, prefer: str | None = None) -> CryptoAsset:
        """Get asset profile and metadata."""
        return await self._call_with_failover_async("asset", "get_asset", symbol, prefer=prefer)

    async def get_market_data(self, symbol: str, prefer: str | None = None) -> CryptoMarketData:
        """Get comprehensive market data for a cryptocurrency."""
        return await self._call_with_failover_async("market_data", "get_market_data", symbol, prefer=prefer)

    # ==================== Async Batch Methods ====================

    async def get_quotes_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
    ) -> dict[str, CryptoQuote | ProviderError]:
        """
        Get quotes for multiple cryptocurrencies concurrently using asyncio.gather.

        Args:
            symbols: List of crypto symbols (e.g., ["BTC", "ETH", "SOL"])
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to CryptoQuote or ProviderError if failed
        """
        async def fetch_quote(symbol: str):
            try:
                quote = await self.get_quote(symbol, prefer)
                return symbol, quote
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_quote(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def get_ohlcv_batch(
        self,
        symbols: list[str],
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        prefer: str | None = None,
    ) -> dict[str, ResultList[CryptoOHLCV] | ProviderError]:
        """
        Get OHLCV data for multiple cryptocurrencies concurrently using asyncio.gather.

        Args:
            symbols: List of crypto symbols
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum candles per symbol
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to ResultList[CryptoOHLCV] or ProviderError if failed
        """
        async def fetch_ohlcv(symbol: str):
            try:
                ohlcv = await self.get_ohlcv(symbol, interval, start, end, limit, prefer)
                return symbol, ohlcv
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_ohlcv(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def get_market_data_batch(
        self,
        symbols: list[str],
        prefer: str | None = None,
    ) -> dict[str, CryptoMarketData | ProviderError]:
        """
        Get market data for multiple cryptocurrencies concurrently.

        Args:
            symbols: List of crypto symbols
            prefer: Preferred provider

        Returns:
            Dict mapping symbol to CryptoMarketData or ProviderError if failed
        """
        async def fetch_market_data(symbol: str):
            try:
                data = await self.get_market_data(symbol, prefer)
                return symbol, data
            except ProviderError as e:
                return symbol, e

        tasks = [fetch_market_data(s) for s in symbols]
        results_list = await asyncio.gather(*tasks)
        return dict(results_list)

    async def aclose(self):
        """Close all provider async clients."""
        for provider in self._providers.values():
            await provider.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
