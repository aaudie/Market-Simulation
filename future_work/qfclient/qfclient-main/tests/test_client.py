"""Tests for MarketClient and CryptoClient."""

from datetime import datetime, date
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from qfclient import MarketClient, CryptoClient, Quote, OHLCV, Interval
from qfclient.common.base import ResultList, ProviderError, RateLimitError


# ============================================================================
# MarketClient Tests
# ============================================================================

class TestMarketClientInit:
    """Tests for MarketClient initialization."""

    def test_client_creation(self):
        """Client should be created successfully."""
        client = MarketClient()
        assert client is not None

    def test_client_with_preference(self):
        """Client should accept provider preference."""
        client = MarketClient(prefer="alpaca")
        assert client.prefer == "alpaca"


class TestMarketClientProviderSelection:
    """Tests for provider selection logic."""

    def test_select_provider_with_configured_provider(self, market_client):
        """Should select a configured provider."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = True
        mock_provider.supports_quotes = True
        mock_provider.provider_name = "mock_provider"

        with patch.dict(
            'qfclient.client.MARKET_PROVIDERS',
            {'mock_provider': lambda: mock_provider}
        ):
            market_client._providers['mock_provider'] = mock_provider

            # Mock the rate limiter to return our provider
            with patch.object(market_client._limiter, 'select_provider', return_value='mock_provider'):
                provider = market_client._select_provider("quotes")

        assert provider == mock_provider

    def test_select_provider_no_configured_raises(self, market_client):
        """Should raise when no providers are configured."""
        # Mock all providers as unconfigured
        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = False

        with patch.dict(
            'qfclient.client.MARKET_PROVIDERS',
            {'mock_provider': lambda: mock_provider},
            clear=True
        ):
            market_client._providers = {'mock_provider': mock_provider}

            with pytest.raises(ProviderError) as exc_info:
                market_client._select_provider("quotes")

        assert "No providers configured" in str(exc_info.value)


class TestMarketClientFailover:
    """Tests for failover behavior."""

    def test_failover_on_provider_error(self, market_client):
        """Should try another provider when one fails."""
        # Create mock providers
        failing_provider = MagicMock()
        failing_provider.provider_name = "failing"
        failing_provider.get_quote.side_effect = ProviderError("failing", "API Error")

        working_provider = MagicMock()
        working_provider.provider_name = "working"
        working_provider.get_quote.return_value = Quote(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now()
        )

        # Track which providers we've tried
        tried = []

        def select_provider(capability, prefer=None, exclude=None):
            exclude = exclude or []
            if "failing" not in exclude:
                tried.append("failing")
                return failing_provider
            tried.append("working")
            return working_provider

        with patch.object(market_client, '_select_provider', side_effect=select_provider):
            result = market_client._call_with_failover("quotes", "get_quote", "AAPL")

        assert result.symbol == "AAPL"
        assert "failing" in tried
        assert "working" in tried

    def test_failover_on_rate_limit(self, market_client):
        """Should try another provider when rate limited."""
        rate_limited_provider = MagicMock()
        rate_limited_provider.provider_name = "rate_limited"
        rate_limited_provider.get_quote.side_effect = RateLimitError("rate_limited", retry_after=60)

        working_provider = MagicMock()
        working_provider.provider_name = "working"
        working_provider.get_quote.return_value = Quote(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now()
        )

        tried = []

        def select_provider(capability, prefer=None, exclude=None):
            exclude = exclude or []
            if "rate_limited" not in exclude:
                tried.append("rate_limited")
                return rate_limited_provider
            tried.append("working")
            return working_provider

        with patch.object(market_client, '_select_provider', side_effect=select_provider):
            result = market_client._call_with_failover("quotes", "get_quote", "AAPL")

        assert result.symbol == "AAPL"
        assert "rate_limited" in tried

    def test_all_providers_fail_raises_last_error(self, market_client):
        """Should raise the last error when all providers fail."""
        failing_provider = MagicMock()
        failing_provider.provider_name = "failing"
        failing_provider.get_quote.side_effect = ProviderError("failing", "All failed")

        call_count = [0]

        def select_provider(capability, prefer=None, exclude=None):
            call_count[0] += 1
            if call_count[0] > 3:
                raise ProviderError("none", "No more providers")
            return failing_provider

        with patch.object(market_client, '_select_provider', side_effect=select_provider):
            with pytest.raises(ProviderError) as exc_info:
                market_client._call_with_failover("quotes", "get_quote", "AAPL")

        assert "All failed" in str(exc_info.value) or "failing" in str(exc_info.value)


class TestMarketClientMethods:
    """Tests for MarketClient methods."""

    def test_get_quote(self, market_client):
        """get_quote should return Quote."""
        mock_quote = Quote(symbol="AAPL", price=150.0, timestamp=datetime.now())

        mock_provider = MagicMock()
        mock_provider.get_quote.return_value = mock_quote

        with patch.object(market_client, '_select_provider', return_value=mock_provider):
            result = market_client.get_quote("AAPL")

        assert result.symbol == "AAPL"
        assert result.price == 150.0

    def test_get_ohlcv(self, market_client):
        """get_ohlcv should return ResultList of OHLCV."""
        mock_candles = ResultList([
            OHLCV(symbol="AAPL", timestamp=datetime.now(), open=150, high=155, low=149, close=154, volume=1000000),
        ])

        mock_provider = MagicMock()
        mock_provider.get_ohlcv.return_value = mock_candles

        with patch.object(market_client, '_select_provider', return_value=mock_provider):
            result = market_client.get_ohlcv("AAPL", Interval.DAY_1)

        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_get_news(self, market_client):
        """get_news should call provider with correct args."""
        from qfclient.market.models import NewsArticle

        mock_news = ResultList([
            NewsArticle(headline="Test headline", published_at=datetime.now()),
        ])

        with patch.object(market_client, '_call_with_failover', return_value=mock_news) as mock_call:
            result = market_client.get_news("AAPL", limit=10)

        mock_call.assert_called_once()
        assert "news" in mock_call.call_args[0]
        assert result == mock_news

    def test_get_dividends(self, market_client):
        """get_dividends should call provider with correct args."""
        from qfclient.market.models import Dividend

        mock_divs = ResultList([
            Dividend(symbol="AAPL", ex_date=date.today(), amount=0.24),
        ])

        with patch.object(market_client, '_call_with_failover', return_value=mock_divs) as mock_call:
            result = market_client.get_dividends("AAPL")

        mock_call.assert_called_once()
        assert "dividends" in mock_call.call_args[0]
        assert result == mock_divs

    def test_get_recommendations(self, market_client):
        """get_recommendations should call provider with correct args."""
        from qfclient.market.models import AnalystRecommendation

        mock_recs = ResultList([
            AnalystRecommendation(symbol="AAPL", buy=10, hold=5, sell=2),
        ])

        with patch.object(market_client, '_call_with_failover', return_value=mock_recs) as mock_call:
            result = market_client.get_recommendations("AAPL")

        mock_call.assert_called_once()
        assert "recommendations" in mock_call.call_args[0]

    def test_get_income_statement(self, market_client):
        """get_income_statement should call provider with correct args."""
        from qfclient.market.models import FinancialStatement

        mock_statements = ResultList([
            FinancialStatement(
                symbol="AAPL",
                statement_type="income",
                period="annual",
                fiscal_date=date.today(),
                currency="USD",
                data={"revenue": 100000}
            ),
        ])

        with patch.object(market_client, '_call_with_failover', return_value=mock_statements) as mock_call:
            result = market_client.get_income_statement("AAPL", period="annual", limit=5)

        mock_call.assert_called_once()
        assert "financials" in mock_call.call_args[0]


class TestMarketClientBatchMethods:
    """Tests for batch methods."""

    def test_get_quotes_batch(self, market_client):
        """get_quotes_batch should fetch multiple quotes in parallel."""
        def mock_failover(capability, method, symbol, **kwargs):
            return Quote(symbol=symbol, price=100.0 + len(symbol), timestamp=datetime.now())

        with patch.object(market_client, '_call_with_failover', side_effect=mock_failover):
            results = market_client.get_quotes_batch(["AAPL", "MSFT", "GOOG"])

        assert len(results) == 3
        assert "AAPL" in results
        assert "MSFT" in results
        assert "GOOG" in results
        assert isinstance(results["AAPL"], Quote)

    def test_get_quotes_batch_with_errors(self, market_client):
        """get_quotes_batch should handle partial failures."""
        def mock_failover(capability, method, symbol, **kwargs):
            if symbol == "INVALID":
                raise ProviderError("test", "Symbol not found")
            return Quote(symbol=symbol, price=100.0, timestamp=datetime.now())

        with patch.object(market_client, '_call_with_failover', side_effect=mock_failover):
            results = market_client.get_quotes_batch(["AAPL", "INVALID"])

        assert isinstance(results["AAPL"], Quote)
        assert isinstance(results["INVALID"], ProviderError)

    def test_get_news_batch(self, market_client):
        """get_news_batch should fetch news for multiple symbols."""
        from qfclient.market.models import NewsArticle

        def mock_failover(capability, method, symbol, *args, **kwargs):
            return ResultList([
                NewsArticle(headline=f"News for {symbol}", published_at=datetime.now())
            ])

        with patch.object(market_client, '_call_with_failover', side_effect=mock_failover):
            results = market_client.get_news_batch(["AAPL", "MSFT"])

        assert len(results) == 2
        assert len(results["AAPL"]) == 1


class TestMarketClientStatus:
    """Tests for status methods."""

    def test_get_status(self, market_client):
        """get_status should return provider status dict."""
        status = market_client.get_status()

        assert isinstance(status, dict)
        # Should have entries for configured providers
        for name, info in status.items():
            assert "configured" in info
            assert "rate_limit" in info


# ============================================================================
# CryptoClient Tests
# ============================================================================

class TestCryptoClientInit:
    """Tests for CryptoClient initialization."""

    def test_client_creation(self):
        """Client should be created successfully."""
        client = CryptoClient()
        assert client is not None

    def test_client_with_preference(self):
        """Client should accept provider preference."""
        client = CryptoClient(prefer="coingecko")
        assert client.prefer == "coingecko"


class TestCryptoClientMethods:
    """Tests for CryptoClient methods."""

    def test_get_quote(self, crypto_client):
        """get_quote should return CryptoQuote."""
        from qfclient.crypto.models import CryptoQuote

        mock_quote = CryptoQuote(
            symbol="BTC",
            price_usd=45000.0,
            timestamp=datetime.now()
        )

        mock_provider = MagicMock()
        mock_provider.get_quote.return_value = mock_quote

        with patch.object(crypto_client, '_select_provider', return_value=mock_provider):
            result = crypto_client.get_quote("BTC")

        assert result.symbol == "BTC"
        assert result.price_usd == 45000.0

    def test_get_market_data(self, crypto_client):
        """get_market_data should return CryptoMarketData."""
        from qfclient.crypto.models import CryptoMarketData

        mock_data = CryptoMarketData(
            symbol="BTC",
            price_usd=45000.0,
            market_cap=900000000000,
        )

        mock_provider = MagicMock()
        mock_provider.get_market_data.return_value = mock_data

        with patch.object(crypto_client, '_select_provider', return_value=mock_provider):
            result = crypto_client.get_market_data("BTC")

        assert result.symbol == "BTC"
        assert result.market_cap == 900000000000

    def test_get_top_coins(self, crypto_client):
        """get_top_coins should return ResultList of CryptoMarketData."""
        from qfclient.crypto.models import CryptoMarketData

        mock_coins = ResultList([
            CryptoMarketData(symbol="BTC", price_usd=45000.0, market_cap_rank=1),
            CryptoMarketData(symbol="ETH", price_usd=2500.0, market_cap_rank=2),
        ])

        mock_provider = MagicMock()
        mock_provider.get_top_coins.return_value = mock_coins

        with patch.object(crypto_client, '_select_provider', return_value=mock_provider):
            result = crypto_client.get_top_coins(limit=10)

        assert len(result) == 2
        assert result[0].symbol == "BTC"


class TestCryptoClientBatchMethods:
    """Tests for crypto batch methods."""

    def test_get_quotes_batch(self, crypto_client):
        """get_quotes_batch should fetch multiple crypto quotes."""
        from qfclient.crypto.models import CryptoQuote

        def mock_failover(capability, method, symbol, **kwargs):
            return CryptoQuote(symbol=symbol, price_usd=1000.0 * len(symbol), timestamp=datetime.now())

        with patch.object(crypto_client, '_call_with_failover', side_effect=mock_failover):
            results = crypto_client.get_quotes_batch(["BTC", "ETH", "SOL"])

        assert len(results) == 3
        assert "BTC" in results
        assert isinstance(results["BTC"], CryptoQuote)
