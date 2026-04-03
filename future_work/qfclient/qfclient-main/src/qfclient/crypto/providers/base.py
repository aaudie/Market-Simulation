"""
Base provider class for cryptocurrency data providers.

Supports both synchronous and asynchronous operations.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import httpx

from ...common.base import ResultList, ProviderError, RateLimitError
from ...common.types import Interval
from ...common.rate_limiter import get_limiter
from ..models import CryptoQuote, CryptoOHLCV, CryptoAsset, CryptoMarketData

logger = logging.getLogger(__name__)


class BaseCryptoProvider(ABC):
    """
    Abstract base class for cryptocurrency data providers.

    Supports both synchronous and asynchronous operations.

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
        """Check if the provider is properly configured."""
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
        """Make an HTTP request with rate limiting and error handling."""
        limiter = get_limiter()

        if not limiter.can_request(self.provider_name):
            wait_time = limiter.time_until_available(self.provider_name)
            raise RateLimitError(self.provider_name, retry_after=wait_time)

        if not url.startswith("http"):
            url = f"{self.base_url}{url}"

        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        try:
            limiter.record_request(self.provider_name)

            resp = self.client.request(
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
            )

            self.last_response_headers = dict(resp.headers)
            self.last_status_code = resp.status_code

            limiter.update_from_headers(
                self.provider_name,
                self.last_response_headers,
                self.last_status_code,
            )

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
        """Make an async HTTP request with rate limiting and error handling."""
        limiter = get_limiter()

        if not limiter.can_request(self.provider_name):
            wait_time = limiter.time_until_available(self.provider_name)
            raise RateLimitError(self.provider_name, retry_after=wait_time)

        if not url.startswith("http"):
            url = f"{self.base_url}{url}"

        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        try:
            limiter.record_request(self.provider_name)

            resp = await self.async_client.request(
                method,
                url,
                params=params,
                json=json,
                headers=request_headers,
            )

            self.last_response_headers = dict(resp.headers)
            self.last_status_code = resp.status_code

            limiter.update_from_headers(
                self.provider_name,
                self.last_response_headers,
                self.last_status_code,
            )

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

    async def get_quote_async(self, symbol: str) -> CryptoQuote:
        """Async version of get_quote. Override for true async implementation."""
        return self.get_quote(symbol)

    async def get_ohlcv_async(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[CryptoOHLCV]:
        """Async version of get_ohlcv. Override for true async implementation."""
        return self.get_ohlcv(symbol, interval, start, end, limit)

    async def get_asset_async(self, symbol: str) -> CryptoAsset:
        """Async version of get_asset. Override for true async implementation."""
        return self.get_asset(symbol)

    async def get_market_data_async(self, symbol: str) -> CryptoMarketData:
        """Async version of get_market_data. Override for true async implementation."""
        return self.get_market_data(symbol)

    # Optional capability methods - override in subclasses
    def get_quote(self, symbol: str) -> CryptoQuote:
        """Get a real-time quote. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support quotes")

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[CryptoOHLCV]:
        """Get OHLCV candle data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support OHLCV")

    def get_asset(self, symbol: str) -> CryptoAsset:
        """Get asset profile. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support asset profiles")

    def get_market_data(self, symbol: str) -> CryptoMarketData:
        """Get comprehensive market data. Override if supported."""
        raise NotImplementedError(f"{self.provider_name} does not support market data")

    # Capability flags
    @property
    def supports_quotes(self) -> bool:
        return False

    @property
    def supports_ohlcv(self) -> bool:
        return False

    @property
    def supports_asset(self) -> bool:
        return False

    @property
    def supports_market_data(self) -> bool:
        return False
