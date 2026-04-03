"""Tests for market and crypto providers with mocked HTTP responses."""

from datetime import datetime, date
from unittest.mock import patch, MagicMock

import pytest

from qfclient.common.base import ResultList, ProviderError, RateLimitError
from qfclient.common.types import Interval


# ============================================================================
# Alpaca Provider Tests
# ============================================================================

class TestAlpacaProvider:
    """Tests for Alpaca provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.alpaca import AlpacaProvider
        return AlpacaProvider(api_key="test_key", api_secret="test_secret")

    def test_is_configured_with_keys(self, provider):
        """Provider should be configured when keys are provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_keys(self):
        """Provider should not be configured without keys."""
        from qfclient.market.providers.alpaca import AlpacaProvider
        with patch.dict("os.environ", {}, clear=True):
            p = AlpacaProvider(api_key=None, api_secret=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Alpaca should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Alpaca should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "quote": {
                "ap": 150.25,
                "bp": 150.20,
                "as": 100,
                "bs": 200,
                "t": "2024-01-15T14:30:00Z"
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.bid == 150.20
        assert quote.ask == 150.25

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = {
            "bars": [
                {"t": "2024-01-15T00:00:00Z", "o": 150.0, "h": 155.0, "l": 149.0, "c": 154.0, "v": 1000000},
                {"t": "2024-01-16T00:00:00Z", "o": 154.0, "h": 156.0, "l": 152.0, "c": 155.0, "v": 1100000},
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0
        assert candles[0].close == 154.0
        assert candles[0].volume == 1000000


# ============================================================================
# Finnhub Provider Tests
# ============================================================================

class TestFinnhubProvider:
    """Tests for Finnhub provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.finnhub import FinnhubProvider
        return FinnhubProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_supports_quotes(self, provider):
        """Finnhub should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_company_profile(self, provider):
        """Finnhub should support company profiles."""
        assert provider.supports_company_profile is True

    def test_supports_earnings(self, provider):
        """Finnhub should support earnings."""
        assert provider.supports_earnings is True

    def test_supports_news(self, provider):
        """Finnhub should support news."""
        assert provider.supports_news is True

    def test_supports_insider_transactions(self, provider):
        """Finnhub should support insider transactions."""
        assert provider.supports_insider_transactions is True

    def test_supports_recommendations(self, provider):
        """Finnhub should support recommendations."""
        assert provider.supports_recommendations is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "c": 150.25,  # Current price
            "o": 149.00,  # Open
            "h": 151.00,  # High
            "l": 148.50,  # Low
            "pc": 148.00,  # Previous close
            "d": 2.25,  # Change
            "dp": 1.52,  # Change percent
            "t": 1705330200
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.previous_close == 148.00

    def test_get_company_profile_success(self, provider):
        """get_company_profile should return CompanyProfile on success."""
        mock_response = {
            "name": "Apple Inc",
            "exchange": "NASDAQ",
            "finnhubIndustry": "Technology",
            "country": "US",
            "currency": "USD",
            "marketCapitalization": 3000000,
            "shareOutstanding": 15000,
            "weburl": "https://apple.com",
            "logo": "https://logo.url",
            "ipo": "1980-12-12"
        }

        with patch.object(provider, 'get', return_value=mock_response):
            profile = provider.get_company_profile("AAPL")

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc"
        assert profile.exchange == "NASDAQ"
        assert profile.country == "US"

    def test_get_news_success(self, provider):
        """get_news should return ResultList of NewsArticle on success."""
        mock_response = [
            {
                "headline": "Apple announces new product",
                "summary": "Apple has announced...",
                "source": "Reuters",
                "url": "https://news.url",
                "image": "https://image.url",
                "datetime": 1705330200,
                "category": "technology"
            }
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            news = provider.get_news("AAPL", limit=10)

        assert len(news) == 1
        assert news[0].headline == "Apple announces new product"
        assert news[0].source == "Reuters"

    def test_get_recommendations_success(self, provider):
        """get_recommendations should return ResultList of AnalystRecommendation."""
        mock_response = [
            {
                "period": "2024-01-01",
                "strongBuy": 10,
                "buy": 15,
                "hold": 5,
                "sell": 2,
                "strongSell": 1
            }
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            recs = provider.get_recommendations("AAPL")

        assert len(recs) == 1
        assert recs[0].strong_buy == 10
        assert recs[0].buy == 15
        assert recs[0].hold == 5


# ============================================================================
# FMP Provider Tests
# ============================================================================

class TestFMPProvider:
    """Tests for Financial Modeling Prep provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.fmp import FMPProvider
        return FMPProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_supports_quotes(self, provider):
        """FMP should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """FMP should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_company_profile(self, provider):
        """FMP should support company profiles."""
        assert provider.supports_company_profile is True

    def test_supports_earnings(self, provider):
        """FMP should support earnings."""
        assert provider.supports_earnings is True

    def test_supports_financials(self, provider):
        """FMP should support financial statements."""
        assert provider.supports_financials is True

    def test_supports_dividends(self, provider):
        """FMP should support dividends."""
        assert provider.supports_dividends is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = [{
            "price": 150.25,
            "open": 149.00,
            "dayHigh": 151.00,
            "dayLow": 148.50,
            "volume": 50000000,
            "previousClose": 148.00,
            "change": 2.25,
            "changePercentage": 1.52,
            "marketCap": 3000000000000,
            "timestamp": 1705330200
        }]

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.volume == 50000000

    def test_get_dividends_success(self, provider):
        """get_dividends should return ResultList of Dividend."""
        mock_response = {
            "historical": [
                {
                    "date": "2024-01-15",
                    "adjDividend": 0.24,
                    "paymentDate": "2024-02-15",
                    "recordDate": "2024-01-16",
                    "declarationDate": "2024-01-01"
                }
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            divs = provider.get_dividends("AAPL")

        assert len(divs) == 1
        assert divs[0].amount == 0.24
        assert divs[0].ex_date == date(2024, 1, 15)

    def test_get_income_statement_success(self, provider):
        """get_income_statement should return ResultList of FinancialStatement."""
        mock_response = [
            {
                "date": "2023-12-31",
                "reportedCurrency": "USD",
                "revenue": 100000000000,
                "grossProfit": 40000000000,
                "netIncome": 20000000000,
                "eps": 1.50
            }
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            statements = provider.get_income_statement("AAPL", period="annual", limit=1)

        assert len(statements) == 1
        assert statements[0].statement_type == "income"
        assert statements[0].data["revenue"] == 100000000000


# ============================================================================
# Yahoo Finance Provider Tests
# ============================================================================

class TestYFinanceProvider:
    """Tests for Yahoo Finance provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.yfinance import YFinanceProvider
        return YFinanceProvider()

    def test_is_configured_always_true(self, provider):
        """YFinance should always be configured (no API key needed)."""
        # Skip if yfinance not installed
        try:
            import yfinance
            assert provider.is_configured() is True
        except ImportError:
            pytest.skip("yfinance not installed")

    def test_supports_quotes(self, provider):
        """YFinance should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """YFinance should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_options(self, provider):
        """YFinance should support options."""
        assert provider.supports_options is True

    def test_supports_dividends(self, provider):
        """YFinance should support dividends."""
        assert provider.supports_dividends is True

    def test_supports_recommendations(self, provider):
        """YFinance should support recommendations."""
        assert provider.supports_recommendations is True

    def test_supports_news(self, provider):
        """YFinance should support news."""
        assert provider.supports_news is True


# ============================================================================
# FRED Provider Tests
# ============================================================================

class TestFREDProvider:
    """Tests for FRED provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.fred import FREDProvider
        return FREDProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_supports_economic(self, provider):
        """FRED should support economic indicators."""
        assert provider.supports_economic is True

    def test_get_economic_indicator_success(self, provider):
        """get_economic_indicator should return ResultList of EconomicIndicator."""
        mock_series_response = {
            "seriess": [{
                "title": "Federal Funds Rate",
                "units": "Percent",
                "frequency": "Monthly"
            }]
        }

        mock_observations_response = {
            "observations": [
                {"date": "2024-01-01", "value": "5.33"},
                {"date": "2024-02-01", "value": "5.33"},
            ]
        }

        def mock_get(url, params=None):
            if "series/observations" in url:
                return mock_observations_response
            return mock_series_response

        with patch.object(provider, 'get', side_effect=mock_get):
            indicators = provider.get_economic_indicator("FEDFUNDS", limit=2)

        assert len(indicators) == 2
        assert indicators[0].series_id == "FEDFUNDS"
        assert indicators[0].value == 5.33


# ============================================================================
# CoinGecko Provider Tests
# ============================================================================

class TestCoinGeckoProvider:
    """Tests for CoinGecko provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.coingecko import CoinGeckoProvider
        return CoinGeckoProvider()

    def test_is_configured_always_true(self, provider):
        """CoinGecko should always be configured (API key optional)."""
        assert provider.is_configured() is True

    def test_supports_quotes(self, provider):
        """CoinGecko should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """CoinGecko should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_asset(self, provider):
        """CoinGecko should support asset profiles."""
        assert provider.supports_asset is True

    def test_supports_market_data(self, provider):
        """CoinGecko should support market data."""
        assert provider.supports_market_data is True

    def test_get_quote_success(self, provider):
        """get_quote should return CryptoQuote on success."""
        mock_response = {
            "bitcoin": {
                "usd": 45000.00,
                "btc": 1.0,
                "usd_market_cap": 900000000000,
                "usd_24h_vol": 20000000000,
                "usd_24h_change": 2.5
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd == 45000.00
        assert quote.market_cap == 900000000000

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of CryptoOHLCV."""
        # CoinGecko OHLC format: [timestamp_ms, open, high, low, close]
        mock_response = [
            [1705276800000, 44000, 45000, 43500, 44800],
            [1705363200000, 44800, 46000, 44500, 45500],
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("BTC", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "BTC"
        assert candles[0].open == 44000
        assert candles[0].close == 44800


# ============================================================================
# CryptoCompare Provider Tests
# ============================================================================

class TestCryptoCompareProvider:
    """Tests for CryptoCompare provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.cryptocompare import CryptoCompareProvider
        return CryptoCompareProvider()

    def test_is_configured_always_true(self, provider):
        """CryptoCompare should always be configured (API key optional)."""
        assert provider.is_configured() is True

    def test_supports_quotes(self, provider):
        """CryptoCompare should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """CryptoCompare should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_get_quote_success(self, provider):
        """get_quote should return CryptoQuote on success."""
        mock_response = {
            "RAW": {
                "BTC": {
                    "USD": {
                        "PRICE": 45000.00,
                        "MKTCAP": 900000000000,
                        "TOTALVOLUME24HTO": 20000000000,
                        "CHANGEPCT24HOUR": 2.5,
                        "HIGH24HOUR": 46000,
                        "LOW24HOUR": 44000,
                        "OPEN24HOUR": 44500,
                        "CIRCULATINGSUPPLY": 19500000,
                        "SUPPLY": 21000000,
                        "LASTUPDATE": 1705330200
                    },
                    "BTC": {
                        "PRICE": 1.0
                    }
                }
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd == 45000.00
        assert quote.market_cap == 900000000000

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of CryptoOHLCV."""
        mock_response = {
            "Data": {
                "Data": [
                    {"time": 1705276800, "open": 44000, "high": 45000, "low": 43500, "close": 44800, "volumeto": 1000000, "volumefrom": 22},
                    {"time": 1705363200, "open": 44800, "high": 46000, "low": 44500, "close": 45500, "volumeto": 1100000, "volumefrom": 24},
                ]
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("BTC", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "BTC"
        assert candles[0].volume == 1000000


# ============================================================================
# Polygon Provider Tests
# ============================================================================

class TestPolygonProvider:
    """Tests for Polygon.io provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.polygon import PolygonProvider
        return PolygonProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.polygon import PolygonProvider
        with patch.dict("os.environ", {}, clear=True):
            p = PolygonProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Polygon should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Polygon should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_company_profile(self, provider):
        """Polygon should support company profiles."""
        assert provider.supports_company_profile is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "status": "OK",
            "results": [{
                "T": "AAPL",
                "c": 150.25,
                "o": 149.00,
                "h": 151.00,
                "l": 148.50,
                "v": 50000000,
                "vw": 150.00,
                "t": 1705330200000
            }]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.high == 151.00

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = {
            "status": "OK",
            "results": [
                {"t": 1705276800000, "o": 150.0, "h": 155.0, "l": 149.0, "c": 154.0, "v": 1000000, "vw": 152.0, "n": 5000},
                {"t": 1705363200000, "o": 154.0, "h": 156.0, "l": 152.0, "c": 155.0, "v": 1100000, "vw": 154.0, "n": 5500},
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0
        assert candles[0].close == 154.0
        assert candles[0].volume == 1000000

    def test_get_company_profile_success(self, provider):
        """get_company_profile should return CompanyProfile on success."""
        mock_response = {
            "status": "OK",
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "primary_exchange": "XNAS",
                "description": "Apple Inc. designs and manufactures...",
                "market_cap": 3000000000000,
                "share_class_shares_outstanding": 15000000000,
                "homepage_url": "https://www.apple.com",
                "sic_code": "3571",
                "sic_description": "Electronic Computers",
                "cik": "0000320193",
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            profile = provider.get_company_profile("AAPL")

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc."
        assert profile.exchange == "XNAS"
        assert profile.market_cap == 3000000000000
        assert profile.website == "https://www.apple.com"


# ============================================================================
# Tiingo Provider Tests
# ============================================================================

class TestTiingoProvider:
    """Tests for Tiingo provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.tiingo import TiingoProvider
        return TiingoProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.tiingo import TiingoProvider
        with patch.dict("os.environ", {}, clear=True):
            p = TiingoProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Tiingo should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Tiingo should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_company_profile(self, provider):
        """Tiingo should support company profiles."""
        assert provider.supports_company_profile is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = [{
            "ticker": "AAPL",
            "last": 150.25,
            "open": 149.00,
            "high": 151.00,
            "low": 148.50,
            "volume": 50000000,
            "bidPrice": 150.20,
            "askPrice": 150.30,
            "bidSize": 100,
            "askSize": 200,
            "prevClose": 148.00,
            "timestamp": "2024-01-15T14:30:00+00:00"
        }]

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.bid == 150.20
        assert quote.ask == 150.30

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = [
            {"date": "2024-01-15T00:00:00+00:00", "adjOpen": 150.0, "adjHigh": 155.0, "adjLow": 149.0, "adjClose": 154.0, "adjVolume": 1000000},
            {"date": "2024-01-16T00:00:00+00:00", "adjOpen": 154.0, "adjHigh": 156.0, "adjLow": 152.0, "adjClose": 155.0, "adjVolume": 1100000},
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0
        assert candles[0].close == 154.0

    def test_get_company_profile_success(self, provider):
        """get_company_profile should return CompanyProfile on success."""
        mock_response = {
            "ticker": "AAPL",
            "name": "Apple Inc",
            "exchangeCode": "NASDAQ",
            "description": "Apple Inc. designs and manufactures...",
            "startDate": "1980-12-12"
        }

        with patch.object(provider, 'get', return_value=mock_response):
            profile = provider.get_company_profile("AAPL")

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc"
        assert profile.exchange == "NASDAQ"


# ============================================================================
# Alpha Vantage Provider Tests
# ============================================================================

class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.alpha_vantage import AlphaVantageProvider
        return AlphaVantageProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.alpha_vantage import AlphaVantageProvider
        with patch.dict("os.environ", {}, clear=True):
            p = AlphaVantageProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Alpha Vantage should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Alpha Vantage should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_company_profile(self, provider):
        """Alpha Vantage should support company profiles."""
        assert provider.supports_company_profile is True

    def test_supports_earnings(self, provider):
        """Alpha Vantage should support earnings."""
        assert provider.supports_earnings is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "149.00",
                "03. high": "151.00",
                "04. low": "148.50",
                "05. price": "150.25",
                "06. volume": "50000000",
                "07. latest trading day": "2024-01-15",
                "08. previous close": "148.00",
                "09. change": "2.25",
                "10. change percent": "1.52%"
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.volume == 50000000
        assert quote.change == 2.25
        assert quote.change_percent == 1.52

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = {
            "Time Series (Daily)": {
                "2024-01-16": {"1. open": "154.0", "2. high": "156.0", "3. low": "152.0", "4. close": "155.0", "5. volume": "1100000"},
                "2024-01-15": {"1. open": "150.0", "2. high": "155.0", "3. low": "149.0", "4. close": "154.0", "5. volume": "1000000"},
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        # Returned in chronological order
        assert candles[0].open == 150.0
        assert candles[1].close == 155.0

    def test_get_company_profile_success(self, provider):
        """get_company_profile should return CompanyProfile on success."""
        mock_response = {
            "Symbol": "AAPL",
            "Name": "Apple Inc",
            "Exchange": "NASDAQ",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Country": "USA",
            "Currency": "USD",
            "MarketCapitalization": "3000000000000",
            "SharesOutstanding": "15000000000",
            "PERatio": "28.5",
            "EPS": "6.05",
            "DividendYield": "0.005",
            "Beta": "1.2",
            "52WeekHigh": "200.00",
            "52WeekLow": "130.00",
            "Description": "Apple Inc. designs and manufactures..."
        }

        with patch.object(provider, 'get', return_value=mock_response):
            profile = provider.get_company_profile("AAPL")

        assert profile.symbol == "AAPL"
        assert profile.name == "Apple Inc"
        assert profile.sector == "Technology"
        assert profile.pe_ratio == 28.5
        assert profile.eps == 6.05

    def test_get_earnings_success(self, provider):
        """get_earnings should return ResultList of EarningsEvent on success."""
        mock_response = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2024-01-15",
                    "reportedEPS": "2.18",
                    "estimatedEPS": "2.10",
                    "surprise": "0.08",
                    "surprisePercentage": "3.81"
                }
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            earnings = provider.get_earnings("AAPL")

        assert len(earnings) == 1
        assert earnings[0].symbol == "AAPL"
        assert earnings[0].eps_actual == 2.18
        assert earnings[0].eps_estimate == 2.10
        assert earnings[0].surprise == 0.08


# ============================================================================
# Twelve Data Provider Tests
# ============================================================================

class TestTwelveDataProvider:
    """Tests for Twelve Data provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.twelve_data import TwelveDataProvider
        return TwelveDataProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.twelve_data import TwelveDataProvider
        with patch.dict("os.environ", {}, clear=True):
            p = TwelveDataProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Twelve Data should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Twelve Data should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "exchange": "NASDAQ",
            "open": "149.00",
            "high": "151.00",
            "low": "148.50",
            "close": "150.25",
            "volume": "50000000",
            "previous_close": "148.00",
            "change": "2.25",
            "percent_change": "1.52",
            "datetime": "2024-01-15"
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.change == 2.25
        assert quote.change_percent == 1.52

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = {
            "values": [
                {"datetime": "2024-01-16", "open": "154.0", "high": "156.0", "low": "152.0", "close": "155.0", "volume": "1100000"},
                {"datetime": "2024-01-15", "open": "150.0", "high": "155.0", "low": "149.0", "close": "154.0", "volume": "1000000"},
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 154.0
        assert candles[0].close == 155.0

    def test_get_quote_error_response(self, provider):
        """get_quote should raise ProviderError on error response."""
        mock_response = {
            "status": "error",
            "message": "Invalid API key"
        }

        with patch.object(provider, 'get', return_value=mock_response):
            with pytest.raises(ProviderError) as exc_info:
                provider.get_quote("AAPL")

        assert "Invalid API key" in str(exc_info.value)


# ============================================================================
# Marketstack Provider Tests
# ============================================================================

class TestMarketstackProvider:
    """Tests for Marketstack provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.marketstack import MarketstackProvider
        return MarketstackProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.marketstack import MarketstackProvider
        with patch.dict("os.environ", {}, clear=True):
            p = MarketstackProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """Marketstack should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """Marketstack should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "data": [{
                "symbol": "AAPL",
                "open": 149.00,
                "high": 151.00,
                "low": 148.50,
                "close": 150.25,
                "volume": 50000000,
                "adj_close": 150.25,
                "exchange": "XNAS",
                "date": "2024-01-15T00:00:00+0000"
            }]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.volume == 50000000

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = {
            "data": [
                {"symbol": "AAPL", "date": "2024-01-16T00:00:00+0000", "open": 154.0, "high": 156.0, "low": 152.0, "close": 155.0, "volume": 1100000},
                {"symbol": "AAPL", "date": "2024-01-15T00:00:00+0000", "open": 150.0, "high": 155.0, "low": 149.0, "close": 154.0, "volume": 1000000},
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        # Marketstack returns newest first, provider reverses to chronological
        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0  # Oldest first after reverse
        assert candles[1].close == 155.0

    def test_get_ohlcv_rejects_intraday(self, provider):
        """get_ohlcv should reject intraday intervals."""
        with pytest.raises(ProviderError) as exc_info:
            provider.get_ohlcv("AAPL", Interval.MINUTE_1)

        assert "not supported" in str(exc_info.value)


# ============================================================================
# EODHD Provider Tests
# ============================================================================

class TestEODHDProvider:
    """Tests for EOD Historical Data provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.eodhd import EODHDProvider
        return EODHDProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.eodhd import EODHDProvider
        with patch.dict("os.environ", {}, clear=True):
            p = EODHDProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """EODHD should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_ohlcv(self, provider):
        """EODHD should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_get_quote_success(self, provider):
        """get_quote should return Quote on success."""
        mock_response = {
            "code": "AAPL.US",
            "open": 149.00,
            "high": 151.00,
            "low": 148.50,
            "close": 150.25,
            "volume": 50000000,
            "previousClose": 148.00,
            "change": 2.25,
            "change_p": 1.52,
            "timestamp": 1705330200
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price == 150.25
        assert quote.open == 149.00
        assert quote.previous_close == 148.00

    def test_get_ohlcv_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV on success."""
        mock_response = [
            {"date": "2024-01-15", "open": 150.0, "high": 155.0, "low": 149.0, "close": 154.0, "adjusted_close": 154.0, "volume": 1000000},
            {"date": "2024-01-16", "open": 154.0, "high": 156.0, "low": 152.0, "close": 155.0, "adjusted_close": 155.0, "volume": 1100000},
        ]

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0
        assert candles[0].close == 154.0

    def test_get_ohlcv_rejects_intraday(self, provider):
        """get_ohlcv should reject intraday intervals."""
        with pytest.raises(ProviderError) as exc_info:
            provider.get_ohlcv("AAPL", Interval.MINUTE_1)

        assert "not supported" in str(exc_info.value)


# ============================================================================
# Tradier Provider Tests
# ============================================================================

class TestTradierProvider:
    """Tests for Tradier provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.tradier import TradierProvider
        return TradierProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.market.providers.tradier import TradierProvider
        with patch.dict("os.environ", {}, clear=True):
            p = TradierProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_ohlcv(self, provider):
        """Tradier should support OHLCV."""
        assert provider.supports_ohlcv is True

    def test_supports_options(self, provider):
        """Tradier should support options."""
        assert provider.supports_options is True

    def test_uses_sandbox_by_default(self, provider):
        """Tradier should use sandbox URL by default."""
        assert "sandbox" in provider.base_url

    def test_uses_production_when_specified(self):
        """Tradier should use production URL when specified."""
        from qfclient.market.providers.tradier import TradierProvider
        p = TradierProvider(api_key="test_key", use_production=True)
        assert "sandbox" not in p.base_url
        assert "api.tradier.com" in p.base_url

    def test_get_ohlcv_daily_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV for daily data."""
        mock_response = {
            "history": {
                "day": [
                    {"date": "2024-01-15", "open": 150.0, "high": 155.0, "low": 149.0, "close": 154.0, "volume": 1000000},
                    {"date": "2024-01-16", "open": 154.0, "high": 156.0, "low": 152.0, "close": 155.0, "volume": 1100000},
                ]
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.DAY_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].open == 150.0
        assert candles[0].close == 154.0

    def test_get_ohlcv_intraday_success(self, provider):
        """get_ohlcv should return ResultList of OHLCV for intraday data."""
        mock_response = {
            "series": {
                "data": [
                    {"time": "2024-01-15T09:30:00", "open": 150.0, "high": 150.5, "low": 149.5, "close": 150.2, "volume": 10000, "vwap": 150.1},
                    {"time": "2024-01-15T09:31:00", "open": 150.2, "high": 150.8, "low": 150.0, "close": 150.5, "volume": 12000, "vwap": 150.3},
                ]
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            candles = provider.get_ohlcv("AAPL", Interval.MINUTE_1, limit=10)

        assert len(candles) == 2
        assert candles[0].symbol == "AAPL"
        assert candles[0].vwap == 150.1

    def test_get_expirations_success(self, provider):
        """get_expirations should return list of dates."""
        mock_response = {
            "expirations": {
                "date": ["2024-01-19", "2024-01-26", "2024-02-02"]
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            expirations = provider.get_expirations("AAPL")

        assert len(expirations) == 3
        assert expirations[0] == date(2024, 1, 19)

    def test_get_options_chain_success(self, provider):
        """get_options_chain should return OptionChain with calls and puts."""
        mock_expirations = {"expirations": {"date": ["2024-01-19"]}}
        mock_quote = {"quotes": {"quote": {"last": 150.0}}}
        mock_chain = {
            "options": {
                "option": [
                    {
                        "symbol": "AAPL240119C00150000",
                        "option_type": "call",
                        "strike": 150.0,
                        "bid": 2.50,
                        "ask": 2.55,
                        "last": 2.52,
                        "volume": 1000,
                        "open_interest": 5000,
                        "greeks": {"delta": 0.5, "gamma": 0.05, "theta": -0.02, "vega": 0.15, "mid_iv": 0.25}
                    },
                    {
                        "symbol": "AAPL240119P00150000",
                        "option_type": "put",
                        "strike": 150.0,
                        "bid": 2.45,
                        "ask": 2.50,
                        "last": 2.48,
                        "volume": 800,
                        "open_interest": 4000,
                        "greeks": {"delta": -0.5, "gamma": 0.05, "theta": -0.02, "vega": 0.15, "mid_iv": 0.25}
                    },
                ]
            }
        }

        def mock_get(url, params=None):
            if "expirations" in url:
                return mock_expirations
            elif "quotes" in url:
                return mock_quote
            elif "chains" in url:
                return mock_chain
            return {}

        with patch.object(provider, 'get', side_effect=mock_get):
            chain = provider.get_options_chain("AAPL")

        assert chain.underlying == "AAPL"
        assert chain.underlying_price == 150.0
        assert len(chain.calls) == 1
        assert len(chain.puts) == 1
        assert chain.calls[0].strike == 150.0
        assert chain.calls[0].delta == 0.5
        assert chain.calls[0].implied_volatility == 0.25


# ============================================================================
# CoinMarketCap Provider Tests
# ============================================================================

class TestCoinMarketCapProvider:
    """Tests for CoinMarketCap provider."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.coinmarketcap import CoinMarketCapProvider
        return CoinMarketCapProvider(api_key="test_key")

    def test_is_configured_with_key(self, provider):
        """Provider should be configured when key is provided."""
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Provider should not be configured without key."""
        from qfclient.crypto.providers.coinmarketcap import CoinMarketCapProvider
        with patch.dict("os.environ", {}, clear=True):
            p = CoinMarketCapProvider(api_key=None)
            assert p.is_configured() is False

    def test_supports_quotes(self, provider):
        """CoinMarketCap should support quotes."""
        assert provider.supports_quotes is True

    def test_supports_asset(self, provider):
        """CoinMarketCap should support asset profiles."""
        assert provider.supports_asset is True

    def test_supports_market_data(self, provider):
        """CoinMarketCap should support market data."""
        assert provider.supports_market_data is True

    def test_get_quote_success(self, provider):
        """get_quote should return CryptoQuote on success."""
        mock_response = {
            "data": {
                "BTC": {
                    "name": "Bitcoin",
                    "symbol": "BTC",
                    "cmc_rank": 1,
                    "circulating_supply": 19500000,
                    "total_supply": 19500000,
                    "max_supply": 21000000,
                    "quote": {
                        "USD": {
                            "price": 45000.00,
                            "market_cap": 900000000000,
                            "volume_24h": 20000000000,
                            "percent_change_1h": 0.5,
                            "percent_change_24h": 2.5,
                            "percent_change_7d": 5.0,
                            "percent_change_30d": 10.0,
                            "last_updated": "2024-01-15T14:30:00.000Z"
                        }
                    }
                }
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.name == "Bitcoin"
        assert quote.price_usd == 45000.00
        assert quote.market_cap == 900000000000
        assert quote.change_24h == 2.5
        assert quote.market_cap_rank == 1

    def test_get_asset_success(self, provider):
        """get_asset should return CryptoAsset on success."""
        mock_response = {
            "data": {
                "BTC": {
                    "name": "Bitcoin",
                    "symbol": "BTC",
                    "slug": "bitcoin",
                    "description": "Bitcoin is a decentralized cryptocurrency...",
                    "category": "coin",
                    "tags": [{"name": "mineable"}, {"name": "pow"}],
                    "urls": {
                        "website": ["https://bitcoin.org"],
                        "source_code": ["https://github.com/bitcoin/bitcoin"],
                        "technical_doc": ["https://bitcoin.org/bitcoin.pdf"],
                        "twitter": ["https://twitter.com/bitcoin"],
                        "reddit": ["https://reddit.com/r/bitcoin"]
                    },
                    "platform": None
                }
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            asset = provider.get_asset("BTC")

        assert asset.symbol == "BTC"
        assert asset.name == "Bitcoin"
        assert asset.slug == "bitcoin"
        assert asset.website == "https://bitcoin.org"
        assert asset.github == "https://github.com/bitcoin/bitcoin"
        assert "mineable" in asset.tags

    def test_get_market_data_success(self, provider):
        """get_market_data should return CryptoMarketData on success."""
        mock_response = {
            "data": {
                "BTC": {
                    "name": "Bitcoin",
                    "symbol": "BTC",
                    "cmc_rank": 1,
                    "circulating_supply": 19500000,
                    "total_supply": 19500000,
                    "max_supply": 21000000,
                    "quote": {
                        "USD": {
                            "price": 45000.00,
                            "market_cap": 900000000000,
                            "fully_diluted_market_cap": 945000000000,
                            "volume_24h": 20000000000,
                            "percent_change_1h": 0.5,
                            "percent_change_24h": 2.5,
                            "percent_change_7d": 5.0,
                            "percent_change_30d": 10.0,
                            "last_updated": "2024-01-15T14:30:00.000Z"
                        }
                    }
                }
            }
        }

        with patch.object(provider, 'get', return_value=mock_response):
            data = provider.get_market_data("BTC")

        assert data.symbol == "BTC"
        assert data.price_usd == 45000.00
        assert data.fully_diluted_valuation == 945000000000
        assert data.change_7d == 5.0

    def test_get_top_coins_success(self, provider):
        """get_top_coins should return ResultList of CryptoMarketData."""
        mock_response = {
            "data": [
                {
                    "name": "Bitcoin",
                    "symbol": "BTC",
                    "cmc_rank": 1,
                    "circulating_supply": 19500000,
                    "quote": {
                        "USD": {
                            "price": 45000.00,
                            "market_cap": 900000000000,
                            "volume_24h": 20000000000,
                            "percent_change_24h": 2.5,
                            "last_updated": "2024-01-15T14:30:00.000Z"
                        }
                    }
                },
                {
                    "name": "Ethereum",
                    "symbol": "ETH",
                    "cmc_rank": 2,
                    "circulating_supply": 120000000,
                    "quote": {
                        "USD": {
                            "price": 2500.00,
                            "market_cap": 300000000000,
                            "volume_24h": 10000000000,
                            "percent_change_24h": 3.0,
                            "last_updated": "2024-01-15T14:30:00.000Z"
                        }
                    }
                }
            ]
        }

        with patch.object(provider, 'get', return_value=mock_response):
            coins = provider.get_top_coins(limit=10)

        assert len(coins) == 2
        assert coins[0].symbol == "BTC"
        assert coins[0].market_cap_rank == 1
        assert coins[1].symbol == "ETH"
        assert coins[1].price_usd == 2500.00
