"""Tests for async client functionality."""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from qfclient import AsyncMarketClient, AsyncCryptoClient
from qfclient.common.base import ResultList, ProviderError


# ============================================================================
# AsyncMarketClient Tests
# ============================================================================

class TestAsyncMarketClientInit:
    """Tests for AsyncMarketClient initialization."""

    def test_init_default(self):
        """Should initialize with no preferred provider."""
        client = AsyncMarketClient()
        assert client.prefer is None
        assert client._providers == {}

    def test_init_with_prefer(self):
        """Should initialize with preferred provider."""
        client = AsyncMarketClient(prefer="alpaca")
        assert client.prefer == "alpaca"


class TestAsyncMarketClientProviderSelection:
    """Tests for AsyncMarketClient provider selection."""

    def test_get_provider_unknown(self):
        """Should raise error for unknown provider."""
        client = AsyncMarketClient()
        with pytest.raises(ValueError, match="Unknown provider"):
            client._get_provider("nonexistent")

    def test_select_provider_no_capability(self):
        """Should raise error when no providers support capability."""
        client = AsyncMarketClient()
        with pytest.raises(ProviderError, match="No providers configured"):
            client._select_provider("nonexistent_capability")


class TestAsyncMarketClientMethods:
    """Tests for AsyncMarketClient async methods."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with async methods."""
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.is_configured.return_value = True
        provider.supports_quotes = True
        provider.supports_ohlcv = True
        provider.supports_company_profile = True
        provider.supports_news = True
        return provider

    @pytest.mark.asyncio
    async def test_get_quote(self, mock_provider):
        """Should call provider's get_quote method."""
        from qfclient.market.models import Quote

        mock_quote = Quote(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(timezone.utc),
        )
        mock_provider.get_quote_async = AsyncMock(return_value=mock_quote)

        client = AsyncMarketClient()
        client._providers["mock"] = mock_provider

        with patch.object(client, '_select_provider', return_value=mock_provider):
            result = await client.get_quote("AAPL")
            assert result.symbol == "AAPL"
            assert result.price == 150.0

    @pytest.mark.asyncio
    async def test_get_quotes_batch(self, mock_provider):
        """Should fetch multiple quotes concurrently."""
        from qfclient.market.models import Quote

        async def mock_get_quote(symbol, prefer=None):
            return Quote(
                symbol=symbol,
                price=100.0 + len(symbol),
                timestamp=datetime.now(timezone.utc),
            )

        client = AsyncMarketClient()
        client.get_quote = mock_get_quote

        results = await client.get_quotes_batch(["AAPL", "GOOGL", "MSFT"])

        assert len(results) == 3
        assert "AAPL" in results
        assert "GOOGL" in results
        assert "MSFT" in results
        assert all(isinstance(v, Quote) for v in results.values())

    @pytest.mark.asyncio
    async def test_get_ohlcv_batch(self, mock_provider):
        """Should fetch OHLCV for multiple symbols concurrently."""
        from qfclient.market.models import OHLCV

        async def mock_get_ohlcv(symbol, interval=None, start=None, end=None, limit=100, prefer=None):
            return ResultList(
                [OHLCV(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=1000000,
                )],
                provider="mock"
            )

        client = AsyncMarketClient()
        client.get_ohlcv = mock_get_ohlcv

        results = await client.get_ohlcv_batch(["AAPL", "GOOGL"])

        assert len(results) == 2
        assert "AAPL" in results
        assert "GOOGL" in results
        assert all(isinstance(v, ResultList) for v in results.values())

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self, mock_provider):
        """Should handle errors in batch operations."""
        from qfclient.market.models import Quote

        async def mock_get_quote(symbol, prefer=None):
            if symbol == "BAD":
                raise ProviderError("mock", "Symbol not found")
            return Quote(
                symbol=symbol,
                price=100.0,
                timestamp=datetime.now(timezone.utc),
            )

        client = AsyncMarketClient()
        client.get_quote = mock_get_quote

        results = await client.get_quotes_batch(["AAPL", "BAD", "MSFT"])

        assert len(results) == 3
        assert isinstance(results["AAPL"], Quote)
        assert isinstance(results["BAD"], ProviderError)
        assert isinstance(results["MSFT"], Quote)


class TestAsyncMarketClientContextManager:
    """Tests for AsyncMarketClient context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Should work as async context manager."""
        async with AsyncMarketClient() as client:
            assert isinstance(client, AsyncMarketClient)

    @pytest.mark.asyncio
    async def test_aclose(self):
        """Should close all provider clients."""
        client = AsyncMarketClient()
        mock_provider = MagicMock()
        mock_provider.aclose = AsyncMock()
        client._providers["mock"] = mock_provider

        await client.aclose()
        mock_provider.aclose.assert_called_once()


# ============================================================================
# AsyncCryptoClient Tests
# ============================================================================

class TestAsyncCryptoClientInit:
    """Tests for AsyncCryptoClient initialization."""

    def test_init_default(self):
        """Should initialize with no preferred provider."""
        client = AsyncCryptoClient()
        assert client.prefer is None
        assert client._providers == {}

    def test_init_with_prefer(self):
        """Should initialize with preferred provider."""
        client = AsyncCryptoClient(prefer="coingecko")
        assert client.prefer == "coingecko"


class TestAsyncCryptoClientProviderSelection:
    """Tests for AsyncCryptoClient provider selection."""

    def test_get_provider_unknown(self):
        """Should raise error for unknown provider."""
        client = AsyncCryptoClient()
        with pytest.raises(ValueError, match="Unknown provider"):
            client._get_provider("nonexistent")

    def test_select_provider_no_capability(self):
        """Should raise error when no providers support capability."""
        client = AsyncCryptoClient()
        with pytest.raises(ProviderError, match="No providers configured"):
            client._select_provider("nonexistent_capability")


class TestAsyncCryptoClientMethods:
    """Tests for AsyncCryptoClient async methods."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with async methods."""
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.is_configured.return_value = True
        provider.supports_quotes = True
        provider.supports_ohlcv = True
        provider.supports_market_data = True
        return provider

    @pytest.mark.asyncio
    async def test_get_quote(self, mock_provider):
        """Should call provider's get_quote method."""
        from qfclient.crypto.models import CryptoQuote

        mock_quote = CryptoQuote(
            symbol="BTC",
            price_usd=50000.0,
            timestamp=datetime.now(timezone.utc),
        )
        mock_provider.get_quote_async = AsyncMock(return_value=mock_quote)

        client = AsyncCryptoClient()
        client._providers["mock"] = mock_provider

        with patch.object(client, '_select_provider', return_value=mock_provider):
            result = await client.get_quote("BTC")
            assert result.symbol == "BTC"
            assert result.price_usd == 50000.0

    @pytest.mark.asyncio
    async def test_get_quotes_batch(self, mock_provider):
        """Should fetch multiple quotes concurrently."""
        from qfclient.crypto.models import CryptoQuote

        async def mock_get_quote(symbol, prefer=None):
            return CryptoQuote(
                symbol=symbol,
                price_usd=1000.0 * len(symbol),
                timestamp=datetime.now(timezone.utc),
            )

        client = AsyncCryptoClient()
        client.get_quote = mock_get_quote

        results = await client.get_quotes_batch(["BTC", "ETH", "SOL"])

        assert len(results) == 3
        assert "BTC" in results
        assert "ETH" in results
        assert "SOL" in results
        assert all(isinstance(v, CryptoQuote) for v in results.values())

    @pytest.mark.asyncio
    async def test_get_market_data_batch(self, mock_provider):
        """Should fetch market data for multiple symbols concurrently."""
        from qfclient.crypto.models import CryptoMarketData

        async def mock_get_market_data(symbol, prefer=None):
            return CryptoMarketData(
                symbol=symbol,
                price_usd=1000.0,
                market_cap_rank=1,
            )

        client = AsyncCryptoClient()
        client.get_market_data = mock_get_market_data

        results = await client.get_market_data_batch(["BTC", "ETH"])

        assert len(results) == 2
        assert "BTC" in results
        assert "ETH" in results
        assert all(isinstance(v, CryptoMarketData) for v in results.values())

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self, mock_provider):
        """Should handle errors in batch operations."""
        from qfclient.crypto.models import CryptoQuote

        async def mock_get_quote(symbol, prefer=None):
            if symbol == "BAD":
                raise ProviderError("mock", "Symbol not found")
            return CryptoQuote(
                symbol=symbol,
                price_usd=1000.0,
                timestamp=datetime.now(timezone.utc),
            )

        client = AsyncCryptoClient()
        client.get_quote = mock_get_quote

        results = await client.get_quotes_batch(["BTC", "BAD", "ETH"])

        assert len(results) == 3
        assert isinstance(results["BTC"], CryptoQuote)
        assert isinstance(results["BAD"], ProviderError)
        assert isinstance(results["ETH"], CryptoQuote)


class TestAsyncCryptoClientContextManager:
    """Tests for AsyncCryptoClient context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Should work as async context manager."""
        async with AsyncCryptoClient() as client:
            assert isinstance(client, AsyncCryptoClient)

    @pytest.mark.asyncio
    async def test_aclose(self):
        """Should close all provider clients."""
        client = AsyncCryptoClient()
        mock_provider = MagicMock()
        mock_provider.aclose = AsyncMock()
        client._providers["mock"] = mock_provider

        await client.aclose()
        mock_provider.aclose.assert_called_once()


# ============================================================================
# Integration Tests (using real providers, but mocked HTTP)
# ============================================================================

class TestAsyncProviderIntegration:
    """Tests for async provider base class integration."""

    @pytest.mark.asyncio
    async def test_provider_async_client_lifecycle(self):
        """Provider async client should be properly initialized and closed."""
        from qfclient.market.providers.yfinance import YFinanceProvider

        provider = YFinanceProvider()

        # Initially no async client
        assert provider._async_client is None

        # Access creates it
        _ = provider.async_client
        assert provider._async_client is not None

        # Close it
        await provider.aclose()
        assert provider._async_client is None

    @pytest.mark.asyncio
    async def test_provider_async_context_manager(self):
        """Provider should work as async context manager."""
        from qfclient.crypto.providers.coingecko import CoinGeckoProvider

        async with CoinGeckoProvider() as provider:
            assert provider._async_client is None  # Not used yet
            _ = provider.async_client  # Now initialized
            assert provider._async_client is not None

        # After exit, client should be closed
        assert provider._async_client is None
