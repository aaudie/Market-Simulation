"""
Base provider class for market data providers.

Supports both synchronous and asynchronous operations.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import date
from typing import Generic, TypeVar, Any

import httpx

from ...common.base import ResultList, ProviderError, RateLimitError
from ...common.types import Interval
from ...common.rate_limiter import get_limiter
from ..models import (
    Quote, OHLCV, CompanyProfile, EarningsEvent, OptionChain, EconomicIndicator,
    NewsArticle, InsiderTransaction, AnalystRecommendation, PriceTarget,
    Dividend, StockSplit, FinancialStatement,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseProvider(ABC):
    """
    Abstract base class for market data providers.

    Subclasses must implement:
    - provider_name: str class attribute
    - is_configured(): bool method
    - At least one fetch method (get_quote, get_ohlcv, etc.)
    """

    provider_name: str = ""
    base_url: str = ""

    def __init__(self):
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self.last_response_headers: dict[str, str] = {}
        self.last_status_code: int = 0

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured (API key, etc.)."""
        pass

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30)
        return self._client

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests. Override in subclasses."""
        return {}

    def _request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request with rate limiting and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL or path to append to base_url
            params: Query parameters
            json: JSON body for POST requests
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            ProviderError: On API errors
            RateLimitError: When rate limited
        """
        limiter = get_limiter()

        # Check rate limit before request
        if not limiter.can_request(self.provider_name):
            wait_time = limiter.time_until_available(self.provider_name)
            raise RateLimitError(self.provider_name, retry_after=wait_time)

        # Build full URL if needed
        if not url.startswith("http"):
            url = f"{self.base_url}{url}"

        # Merge headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        try:
            # Record the request
            limiter.record_request(self.provider_name)

            resp = self.client.request(
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
            )

            # Store response info for rate limit updates
            self.last_response_headers = dict(resp.headers)
            self.last_status_code = resp.status_code

            # Update rate limiter from response headers
            limiter.update_from_headers(
                self.provider_name,
                self.last_response_headers,
                self.last_status_code,
            )

            # Handle HTTP errors
            if resp.status_code == 429:
                raise RateLimitError(self.provider_name)

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            limiter.record_failure(self.provider_name)
            raise ProviderError(
                self.provider_name,
                f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code,
            )
        except httpx.RequestError as e:
            limiter.record_failure(self.provider_name)
            raise ProviderError(self.provider_name, f"Request failed: {str(e)}")

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict[str, Any]:
        """Convenience method for GET requests."""
        return self._request("GET", url, params=params, **kwargs)

    def post(self, url: str, json: dict | None = None, **kwargs) -> dict[str, Any]:
        """Convenience method for POST requests."""
        return self._request("POST", url, json=json, **kwargs)

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Async Support ====================

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Lazy-initialized async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=30)
        return self._async_client

    async def _request_async(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """
        Make an async HTTP request with rate limiting and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL or path to append to base_url
            params: Query parameters
            json: JSON body for POST requests
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            ProviderError: On API errors
            RateLimitError: When rate limited
        """
        limiter = get_limiter()

        # Check rate limit before request
        if not limiter.can_request(self.provider_name):
            wait_time = limiter.time_until_available(self.provider_name)
            raise RateLimitError(self.provider_name, retry_after=wait_time)

        # Build full URL if needed
        if not url.startswith("http"):
            url = f"{self.base_url}{url}"

        # Merge headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        try:
            # Record the request
            limiter.record_request(self.provider_name)

            resp = await self.async_client.request(
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
            )

            # Store response info for rate limit updates
            self.last_response_headers = dict(resp.headers)
            self.last_status_code = resp.status_code

            # Update rate limiter from response headers
            limiter.update_from_headers(
                self.provider_name,
                self.last_response_headers,
                self.last_status_code,
            )

            # Handle HTTP errors
            if resp.status_code == 429:
                raise RateLimitError(self.provider_name)

            resp.raise_for_status()
            return resp.json()

        except httpx.HTTPStatusError as e:
            limiter.record_failure(self.provider_name)
            raise ProviderError(
                self.provider_name,
                f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code,
            )
        except httpx.RequestError as e:
            limiter.record_failure(self.provider_name)
            raise ProviderError(self.provider_name, f"Request failed: {str(e)}")

    async def get_async(self, url: str, params: dict | None = None, **kwargs) -> dict[str, Any]:
        """Async convenience method for GET requests."""
        return await self._request_async("GET", url, params=params, **kwargs)

    async def post_async(self, url: str, json: dict | None = None, **kwargs) -> dict[str, Any]:
        """Async convenience method for POST requests."""
        return await self._request_async("POST", url, json=json, **kwargs)

    async def aclose(self):
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    # ==================== Async Data Methods ====================
    # Default implementations that call sync methods
    # Providers can override these for true async implementations

    async def get_quote_async(self, symbol: str) -> Quote:
        """Async version of get_quote. Override for true async implementation."""
        return self.get_quote(symbol)

    async def get_ohlcv_async(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[OHLCV]:
        """Async version of get_ohlcv. Override for true async implementation."""
        return self.get_ohlcv(symbol, interval, start, end, limit)

    async def get_company_profile_async(self, symbol: str) -> CompanyProfile:
        """Async version of get_company_profile. Override for true async implementation."""
        return self.get_company_profile(symbol)

    async def get_news_async(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
    ) -> ResultList[NewsArticle]:
        """Async version of get_news. Override for true async implementation."""
        return self.get_news(symbol, start, end, limit)

    async def get_dividends_async(self, symbol: str) -> ResultList[Dividend]:
        """Async version of get_dividends. Override for true async implementation."""
        return self.get_dividends(symbol)

    async def get_recommendations_async(self, symbol: str) -> ResultList[AnalystRecommendation]:
        """Async version of get_recommendations. Override for true async implementation."""
        return self.get_recommendations(symbol)

    # Optional capability methods - override in subclasses
    def get_quote(self, symbol: str) -> Quote:
        """Get a real-time quote. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support quotes")

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[OHLCV]:
        """Get OHLCV candle data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support OHLCV")

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company profile. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support company profiles")

    def get_earnings(
        self,
        symbol: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[EarningsEvent]:
        """Get earnings calendar. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support earnings")

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChain:
        """Get options chain. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support options")

    def get_economic_indicator(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[EconomicIndicator]:
        """Get economic indicator data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support economic indicators")

    # Capability flags
    @property
    def supports_quotes(self) -> bool:
        return False

    @property
    def supports_ohlcv(self) -> bool:
        return False

    @property
    def supports_company_profile(self) -> bool:
        return False

    @property
    def supports_earnings(self) -> bool:
        return False

    @property
    def supports_options(self) -> bool:
        return False

    @property
    def supports_economic(self) -> bool:
        return False

    @property
    def supports_news(self) -> bool:
        return False

    @property
    def supports_insider_transactions(self) -> bool:
        return False

    @property
    def supports_recommendations(self) -> bool:
        return False

    @property
    def supports_dividends(self) -> bool:
        return False

    @property
    def supports_financials(self) -> bool:
        return False

    # Additional optional methods - override in subclasses
    def get_news(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
    ) -> ResultList[NewsArticle]:
        """Get news articles for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support news")

    def get_insider_transactions(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[InsiderTransaction]:
        """Get insider transactions for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support insider transactions")

    def get_recommendations(self, symbol: str) -> ResultList[AnalystRecommendation]:
        """Get analyst recommendations for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support recommendations")

    def get_price_target(self, symbol: str) -> PriceTarget:
        """Get analyst price targets for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support price targets")

    def get_dividends(self, symbol: str) -> ResultList[Dividend]:
        """Get dividend history for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support dividends")

    def get_stock_splits(self, symbol: str) -> ResultList[StockSplit]:
        """Get stock split history for a symbol. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support stock splits")

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """Get income statement data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support financial statements")

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """Get balance sheet data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support financial statements")

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """Get cash flow statement data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support financial statements")
