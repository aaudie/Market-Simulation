"""
Integration tests that make REAL API calls.

These tests verify that:
1. API endpoints are reachable
2. Responses parse correctly into our Pydantic models
3. Data types are properly coerced

Run with: pytest tests/test_integration.py -v

Note: Tests are skipped if the provider is not configured (missing API key).
Free-tier providers (Yahoo Finance, CoinGecko, CryptoCompare) should always work.
"""

import pytest
from datetime import date, timedelta
from functools import wraps

from qfclient.common.base import ProviderError, RateLimitError
from qfclient.common.types import Interval


def skip_on_rate_limit(func):
    """Decorator to skip test if rate limit is hit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            pytest.skip(f"Rate limit hit: {e}")
    return wrapper


# ============================================================================
# Market Provider Integration Tests
# ============================================================================

class TestYahooFinanceIntegration:
    """Integration tests for Yahoo Finance (free, no API key needed)."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.yfinance import YFinanceProvider
        p = YFinanceProvider()
        if not p.is_configured():
            pytest.skip("yfinance package not installed")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Yahoo Finance."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert isinstance(quote.price, (int, float))
        assert quote.price > 0
        print(f"\n  AAPL price: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Yahoo Finance."""
        end = date.today()
        start = end - timedelta(days=30)

        candles = provider.get_ohlcv("MSFT", Interval.DAY_1, start=start, end=end, limit=10)

        assert len(candles) > 0
        assert candles[0].symbol == "MSFT"
        assert candles[0].open > 0
        assert candles[0].close > 0
        assert candles[0].volume > 0
        print(f"\n  Got {len(candles)} MSFT candles, latest close: ${candles[-1].close:.2f}")

    def test_get_company_profile_real(self, provider):
        """Fetch real company profile from Yahoo Finance."""
        profile = provider.get_company_profile("GOOGL")

        assert profile.symbol == "GOOGL"
        assert profile.name is not None
        assert len(profile.name) > 0
        print(f"\n  {profile.symbol}: {profile.name}, Sector: {profile.sector}")

    def test_get_dividends_real(self, provider):
        """Fetch real dividend data from Yahoo Finance."""
        divs = provider.get_dividends("AAPL")

        assert len(divs) > 0
        assert divs[0].amount > 0
        print(f"\n  AAPL has {len(divs)} dividend records, latest: ${divs[-1].amount:.4f}")

    def test_get_recommendations_real(self, provider):
        """Fetch real analyst recommendations from Yahoo Finance."""
        recs = provider.get_recommendations("NVDA")

        # May be empty for some symbols
        if len(recs) > 0:
            assert recs[0].buy is not None or recs[0].hold is not None
            print(f"\n  NVDA recommendations: Buy={recs[0].buy}, Hold={recs[0].hold}, Sell={recs[0].sell}")
        else:
            print("\n  No recommendations available for NVDA")


class TestAlpacaIntegration:
    """Integration tests for Alpaca."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.alpaca import AlpacaProvider
        p = AlpacaProvider()
        if not p.is_configured():
            pytest.skip("Alpaca not configured (missing ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY)")
        return p

    @skip_on_rate_limit
    def test_get_quote_real(self, provider):
        """Fetch a real quote from Alpaca."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert isinstance(quote.price, (int, float))
        assert quote.price > 0
        print(f"\n  AAPL quote from Alpaca: ${quote.price:.2f}")

    @skip_on_rate_limit
    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Alpaca."""
        end = date.today()
        start = end - timedelta(days=7)

        try:
            candles = provider.get_ohlcv("SPY", Interval.DAY_1, start=start, end=end, limit=5)
            assert len(candles) > 0
            assert candles[0].symbol == "SPY"
            assert candles[0].close > 0
            print(f"\n  Got {len(candles)} SPY candles from Alpaca")
        except ProviderError as e:
            if "403" in str(e):
                pytest.skip("Alpaca OHLCV requires paid subscription")
            raise


class TestFinnhubIntegration:
    """Integration tests for Finnhub."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.finnhub import FinnhubProvider
        p = FinnhubProvider()
        if not p.is_configured():
            pytest.skip("Finnhub not configured (missing FINNHUB_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Finnhub."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL from Finnhub: ${quote.price:.2f}")

    def test_get_company_profile_real(self, provider):
        """Fetch real company profile from Finnhub."""
        profile = provider.get_company_profile("MSFT")

        assert profile.symbol == "MSFT"
        assert profile.name is not None
        print(f"\n  {profile.name}, Market Cap: ${profile.market_cap:,.0f}" if profile.market_cap else f"\n  {profile.name}")

    def test_get_news_real(self, provider):
        """Fetch real news from Finnhub."""
        news = provider.get_news("TSLA", limit=5)

        assert len(news) > 0
        assert news[0].headline is not None
        print(f"\n  Got {len(news)} news articles for TSLA")
        print(f"  Latest: {news[0].headline[:60]}...")


class TestFMPIntegration:
    """Integration tests for Financial Modeling Prep."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.fmp import FMPProvider
        p = FMPProvider()
        if not p.is_configured():
            pytest.skip("FMP not configured (missing FMP_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from FMP."""
        quote = provider.get_quote("AMZN")

        assert quote.symbol == "AMZN"
        assert quote.price > 0
        print(f"\n  AMZN from FMP: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from FMP."""
        candles = provider.get_ohlcv("META", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        assert candles[0].close > 0
        print(f"\n  Got {len(candles)} META candles from FMP")

    def test_get_dividends_real(self, provider):
        """Fetch real dividend data from FMP."""
        try:
            divs = provider.get_dividends("JNJ")
            assert len(divs) > 0
            print(f"\n  JNJ has {len(divs)} dividend records")
        except ProviderError as e:
            if "404" in str(e):
                pytest.skip("FMP dividends endpoint may have changed - needs investigation")
            raise


class TestPolygonIntegration:
    """Integration tests for Polygon.io."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.polygon import PolygonProvider
        p = PolygonProvider()
        if not p.is_configured():
            pytest.skip("Polygon not configured (missing POLYGON_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Polygon."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL from Polygon: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Polygon."""
        end = date.today() - timedelta(days=1)  # Yesterday (free tier is delayed)
        start = end - timedelta(days=30)

        candles = provider.get_ohlcv("AAPL", Interval.DAY_1, start=start, end=end, limit=10)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} AAPL candles from Polygon")

    def test_get_company_profile_real(self, provider):
        """Fetch real company profile from Polygon."""
        profile = provider.get_company_profile("TSLA")

        assert profile.symbol == "TSLA"
        assert profile.name is not None
        print(f"\n  {profile.name} from Polygon")


class TestTiingoIntegration:
    """Integration tests for Tiingo."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.tiingo import TiingoProvider
        p = TiingoProvider()
        if not p.is_configured():
            pytest.skip("Tiingo not configured (missing TIINGO_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Tiingo."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL from Tiingo: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Tiingo."""
        end = date.today()
        start = end - timedelta(days=30)

        candles = provider.get_ohlcv("GOOGL", Interval.DAY_1, start=start, end=end, limit=10)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} GOOGL candles from Tiingo")


class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.alpha_vantage import AlphaVantageProvider
        p = AlphaVantageProvider()
        if not p.is_configured():
            pytest.skip("Alpha Vantage not configured (missing ALPHA_VANTAGE_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Alpha Vantage."""
        quote = provider.get_quote("IBM")

        assert quote.symbol == "IBM"
        assert quote.price > 0
        print(f"\n  IBM from Alpha Vantage: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Alpha Vantage."""
        import time
        time.sleep(1)  # Alpha Vantage free tier: 1 req/sec

        try:
            candles = provider.get_ohlcv("IBM", Interval.DAY_1, limit=10)
            assert len(candles) > 0
            print(f"\n  Got {len(candles)} IBM candles from Alpha Vantage")
        except ProviderError as e:
            if "Rate limited" in str(e) or "spreading out" in str(e):
                pytest.skip("Alpha Vantage rate limited (free tier: 1 req/sec, 25/day)")
            raise


class TestTwelveDataIntegration:
    """Integration tests for Twelve Data."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.twelve_data import TwelveDataProvider
        p = TwelveDataProvider()
        if not p.is_configured():
            pytest.skip("Twelve Data not configured (missing TWELVE_DATA_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from Twelve Data."""
        quote = provider.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL from Twelve Data: ${quote.price:.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Twelve Data."""
        candles = provider.get_ohlcv("MSFT", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} MSFT candles from Twelve Data")


class TestFREDIntegration:
    """Integration tests for FRED."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.fred import FREDProvider
        p = FREDProvider()
        if not p.is_configured():
            pytest.skip("FRED not configured (missing FRED_API_KEY)")
        return p

    def test_get_economic_indicator_real(self, provider):
        """Fetch real economic data from FRED."""
        indicators = provider.get_economic_indicator("FEDFUNDS", limit=12)

        assert len(indicators) > 0
        assert indicators[0].series_id == "FEDFUNDS"
        assert indicators[0].value is not None
        print(f"\n  Fed Funds Rate: {indicators[-1].value}% (as of {indicators[-1].observation_date})")

    def test_get_gdp_real(self, provider):
        """Fetch real GDP data from FRED."""
        gdp = provider.get_economic_indicator("GDP", limit=4)

        assert len(gdp) > 0
        print(f"\n  Latest GDP: ${gdp[-1].value:,.0f}B")


class TestTradierIntegration:
    """Integration tests for Tradier."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.tradier import TradierProvider
        p = TradierProvider()
        if not p.is_configured():
            pytest.skip("Tradier not configured (missing TRADIER_API_KEY)")
        return p

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from Tradier."""
        end = date.today()
        start = end - timedelta(days=30)

        candles = provider.get_ohlcv("AAPL", Interval.DAY_1, start=start, end=end, limit=10)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} AAPL candles from Tradier")

    def test_get_expirations_real(self, provider):
        """Fetch real option expirations from Tradier."""
        expirations = provider.get_expirations("SPY")

        assert len(expirations) > 0
        print(f"\n  SPY has {len(expirations)} option expirations, next: {expirations[0]}")

    def test_get_options_chain_real(self, provider):
        """Fetch real options chain from Tradier."""
        chain = provider.get_options_chain("SPY")

        assert chain.underlying == "SPY"
        assert len(chain.calls) > 0
        assert len(chain.puts) > 0
        print(f"\n  SPY options chain: {len(chain.calls)} calls, {len(chain.puts)} puts")
        if chain.calls:
            c = chain.calls[0]
            print(f"  Sample call: Strike=${c.strike}, Delta={c.delta}, IV={c.implied_volatility}")


# ============================================================================
# Crypto Provider Integration Tests
# ============================================================================

class TestCoinGeckoIntegration:
    """Integration tests for CoinGecko (free, no API key needed)."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.coingecko import CoinGeckoProvider
        return CoinGeckoProvider()

    @skip_on_rate_limit
    def test_get_quote_real(self, provider):
        """Fetch a real quote from CoinGecko."""
        quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC from CoinGecko: ${quote.price_usd:,.2f}")

    @skip_on_rate_limit
    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from CoinGecko."""
        candles = provider.get_ohlcv("ETH", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        assert candles[0].close > 0
        print(f"\n  Got {len(candles)} ETH candles from CoinGecko")

    @skip_on_rate_limit
    def test_get_asset_real(self, provider):
        """Fetch real asset profile from CoinGecko."""
        asset = provider.get_asset("BTC")

        assert asset.symbol == "BTC"
        assert asset.name == "Bitcoin"
        print(f"\n  {asset.name}: {asset.description[:80] if asset.description else 'No description'}...")

    @skip_on_rate_limit
    def test_get_market_data_real(self, provider):
        """Fetch real market data from CoinGecko."""
        data = provider.get_market_data("ETH")

        assert data.symbol == "ETH"
        assert data.price_usd > 0
        assert data.market_cap > 0
        print(f"\n  ETH: ${data.price_usd:,.2f}, MCap: ${data.market_cap:,.0f}, Rank: #{data.market_cap_rank}")

    @skip_on_rate_limit
    def test_get_top_coins_real(self, provider):
        """Fetch real top coins from CoinGecko."""
        coins = provider.get_top_coins(limit=10)

        assert len(coins) == 10
        assert coins[0].market_cap_rank == 1
        print(f"\n  Top 3 coins:")
        for coin in coins[:3]:
            print(f"    #{coin.market_cap_rank} {coin.symbol}: ${coin.price_usd:,.2f}")

    @skip_on_rate_limit
    def test_get_trending_real(self, provider):
        """Fetch real trending coins from CoinGecko."""
        trending = provider.get_trending()

        assert len(trending) > 0
        print(f"\n  Trending coins:")
        for coin in trending[:3]:
            print(f"    {coin['symbol']}: rank #{coin['market_cap_rank']}")

    @skip_on_rate_limit
    def test_get_global_real(self, provider):
        """Fetch real global market data from CoinGecko."""
        data = provider.get_global()

        assert data["total_market_cap_usd"] > 0
        assert data["btc_dominance"] > 0
        print(f"\n  Total Crypto Market Cap: ${data['total_market_cap_usd']:,.0f}")
        print(f"  BTC Dominance: {data['btc_dominance']:.1f}%")


class TestCryptoCompareIntegration:
    """Integration tests for CryptoCompare (free, no API key needed for basic)."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.cryptocompare import CryptoCompareProvider
        return CryptoCompareProvider()

    def test_get_quote_real(self, provider):
        """Fetch a real quote from CryptoCompare."""
        quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC from CryptoCompare: ${quote.price_usd:,.2f}")

    def test_get_ohlcv_real(self, provider):
        """Fetch real OHLCV data from CryptoCompare."""
        candles = provider.get_ohlcv("ETH", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        assert candles[0].close > 0
        print(f"\n  Got {len(candles)} ETH candles from CryptoCompare")

    def test_get_ohlcv_intraday_real(self, provider):
        """Fetch real intraday OHLCV data from CryptoCompare."""
        candles = provider.get_ohlcv("BTC", Interval.HOUR_1, limit=24)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} hourly BTC candles from CryptoCompare")

    def test_get_market_data_real(self, provider):
        """Fetch real market data from CryptoCompare."""
        data = provider.get_market_data("SOL")

        assert data.symbol == "SOL"
        assert data.price_usd > 0
        print(f"\n  SOL from CryptoCompare: ${data.price_usd:,.2f}")

    def test_get_top_coins_real(self, provider):
        """Fetch real top coins from CryptoCompare."""
        coins = provider.get_top_coins(limit=10)

        assert len(coins) > 0
        print(f"\n  Got {len(coins)} top coins from CryptoCompare")


class TestCoinMarketCapIntegration:
    """Integration tests for CoinMarketCap."""

    @pytest.fixture
    def provider(self):
        from qfclient.crypto.providers.coinmarketcap import CoinMarketCapProvider
        p = CoinMarketCapProvider()
        if not p.is_configured():
            pytest.skip("CoinMarketCap not configured (missing COINMARKETCAP_API_KEY)")
        return p

    def test_get_quote_real(self, provider):
        """Fetch a real quote from CoinMarketCap."""
        quote = provider.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC from CoinMarketCap: ${quote.price_usd:,.2f}")

    def test_get_asset_real(self, provider):
        """Fetch real asset profile from CoinMarketCap."""
        asset = provider.get_asset("ETH")

        assert asset.symbol == "ETH"
        assert asset.name == "Ethereum"
        print(f"\n  {asset.name}: {asset.description[:80] if asset.description else 'No description'}...")

    def test_get_top_coins_real(self, provider):
        """Fetch real top coins from CoinMarketCap."""
        coins = provider.get_top_coins(limit=10)

        assert len(coins) == 10
        print(f"\n  Top coin from CMC: {coins[0].symbol} at ${coins[0].price_usd:,.2f}")


# ============================================================================
# Client Integration Tests
# ============================================================================

class TestMarketClientIntegration:
    """Integration tests for MarketClient with real API calls."""

    @pytest.fixture
    def client(self):
        from qfclient import MarketClient
        return MarketClient()

    def test_get_quote_with_failover(self, client):
        """Test MarketClient fetches real quote with provider selection."""
        quote = client.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL via MarketClient: ${quote.price:.2f}")

    def test_get_ohlcv_with_failover(self, client):
        """Test MarketClient fetches real OHLCV with provider selection."""
        candles = client.get_ohlcv("MSFT", limit=5)

        assert len(candles) > 0
        print(f"\n  Got {len(candles)} MSFT candles via MarketClient")

    def test_get_quotes_batch(self, client):
        """Test MarketClient batch quote fetching."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = client.get_quotes_batch(symbols)

        assert len(results) == 3
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"\n  {sym}: ${results[sym].price:.2f}")


class TestCryptoClientIntegration:
    """Integration tests for CryptoClient with real API calls."""

    @pytest.fixture
    def client(self):
        from qfclient import CryptoClient
        return CryptoClient()

    def test_get_quote_with_failover(self, client):
        """Test CryptoClient fetches real quote with provider selection."""
        quote = client.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC via CryptoClient: ${quote.price_usd:,.2f}")

    def test_get_top_coins(self, client):
        """Test CryptoClient fetches real top coins."""
        coins = client.get_top_coins(limit=5)

        assert len(coins) == 5
        print(f"\n  Top 5 coins via CryptoClient:")
        for coin in coins:
            print(f"    {coin.symbol}: ${coin.price_usd:,.2f}")

    def test_get_quotes_batch(self, client):
        """Test CryptoClient batch quote fetching."""
        symbols = ["BTC", "ETH", "SOL"]
        results = client.get_quotes_batch(symbols)

        assert len(results) == 3
        for sym in symbols:
            assert sym in results


# ============================================================================
# SEC EDGAR Integration Tests
# ============================================================================

class TestSECIntegration:
    """Integration tests for SEC EDGAR (free, no API key needed)."""

    @pytest.fixture
    def provider(self):
        from qfclient.market.providers.sec import SECProvider
        return SECProvider()

    @pytest.fixture
    def client(self):
        from qfclient import MarketClient
        return MarketClient()

    def test_provider_is_configured(self, provider):
        """SEC provider should always be configured."""
        assert provider.is_configured() is True
        assert provider.provider_name == "sec"

    def test_get_insider_summary(self, client):
        """Test fetching insider trading summary via MarketClient."""
        from qfclient import InsiderSummary

        summary = client.get_insider_summary("AAPL")

        assert isinstance(summary, InsiderSummary)
        assert summary.symbol == "AAPL"
        print(f"\n  AAPL insider summary:")
        print(f"    Unique insiders: {summary.unique_insiders}")
        print(f"    Buyers: {summary.num_buyers}, Sellers: {summary.num_sellers}")
        if summary.net_purchase_ratio is not None:
            print(f"    NPR: {summary.net_purchase_ratio:.2f}")
            print(f"    Sentiment: {summary.insider_sentiment}")

    def test_get_sec_transactions(self, client):
        """Test fetching SEC Form 4 transactions via MarketClient."""
        transactions = client.get_sec_transactions("MSFT", limit=10)

        print(f"\n  MSFT SEC transactions: {len(transactions)} found")
        for txn in transactions[:3]:
            print(f"    {txn.owner_name}: {txn.shares:,.0f} shares @ ${txn.price_per_share:.2f}")
            print(f"      Role: {txn.role.role_description}, Type: {txn.role.role_type.value}")
            if txn.position_change_pct:
                print(f"      Position change: {txn.position_change_pct:+.1f}%")

    def test_get_sec_filings(self, client):
        """Test fetching complete Form 4 filings via MarketClient."""
        filings = client.get_sec_filings("GOOGL", limit=5)

        print(f"\n  GOOGL SEC filings: {len(filings)} found")
        for filing in filings[:2]:
            print(f"    {filing.owner_name} ({filing.role.role_description})")
            print(f"      Transactions: {filing.transaction_count}")
            print(f"      Net shares: {filing.net_shares:+,.0f}")
            print(f"      Net value: ${filing.net_value:+,.2f}")

    def test_transaction_data_quality(self, client):
        """Test that SEC transactions have expected data populated."""
        transactions = client.get_sec_transactions("NVDA", limit=20)

        for txn in transactions:
            # Required fields
            assert txn.symbol == "NVDA"
            assert txn.owner_name is not None
            assert txn.transaction_date is not None
            assert txn.shares >= 0
            assert txn.acquired_or_disposed in ("A", "D")

            # Role should always be present
            assert txn.role is not None
            assert 0.0 <= txn.role.importance <= 1.0

    def test_filing_aggregation_correctness(self, client):
        """Test that filing aggregations are computed correctly."""
        filings = client.get_sec_filings("AMZN", limit=5)

        for filing in filings:
            # Transaction count should match
            assert filing.transaction_count == len(filing.transactions)

            # Net shares should equal acquired - disposed
            computed_net = filing.total_shares_acquired - filing.total_shares_disposed
            assert filing.net_shares == computed_net

            # Net value should equal acquired - disposed
            computed_value = filing.total_value_acquired - filing.total_value_disposed
            assert abs(filing.net_value - computed_value) < 0.01  # Float tolerance
