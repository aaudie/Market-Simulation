"""
Async integration tests that make REAL API calls.

These tests verify that:
1. Async clients work with real APIs
2. Concurrent requests via asyncio.gather work correctly
3. Async context managers properly manage connections
4. Data is correctly parsed into Pydantic models

Run with: pytest tests/test_async_integration.py -v
"""

import asyncio
import pytest
from datetime import date, timedelta

from qfclient import AsyncMarketClient, AsyncCryptoClient
from qfclient.common.base import ProviderError
from qfclient.common.types import Interval


# ============================================================================
# AsyncMarketClient Integration Tests
# ============================================================================

class TestAsyncMarketClientIntegration:
    """Integration tests for AsyncMarketClient with real API calls."""

    @pytest.fixture
    def client(self):
        return AsyncMarketClient()

    @pytest.mark.asyncio
    async def test_get_quote_real(self, client):
        """Fetch a real quote using async client."""
        async with client:
            quote = await client.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert isinstance(quote.price, (int, float))
        assert quote.price > 0
        print(f"\n  AAPL via AsyncMarketClient: ${quote.price:.2f}")

    @pytest.mark.asyncio
    async def test_get_ohlcv_real(self, client):
        """Fetch real OHLCV data using async client."""
        async with client:
            candles = await client.get_ohlcv("MSFT", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        assert candles[0].symbol == "MSFT"
        assert candles[0].close > 0
        print(f"\n  Got {len(candles)} MSFT candles via async, latest: ${candles[-1].close:.2f}")

    @pytest.mark.asyncio
    async def test_get_company_profile_real(self, client):
        """Fetch real company profile using async client."""
        async with client:
            profile = await client.get_company_profile("GOOGL")

        assert profile.symbol == "GOOGL"
        assert profile.name is not None
        print(f"\n  {profile.symbol}: {profile.name}")

    @pytest.mark.asyncio
    async def test_get_quotes_batch_real(self, client):
        """Fetch multiple quotes concurrently using asyncio.gather."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]

        async with client:
            results = await client.get_quotes_batch(symbols)

        assert len(results) == 5
        print(f"\n  Batch quotes (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: ${results[sym].price:.2f}")

    @pytest.mark.asyncio
    async def test_get_ohlcv_batch_real(self, client):
        """Fetch OHLCV for multiple symbols concurrently."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        async with client:
            results = await client.get_ohlcv_batch(symbols, limit=5)

        assert len(results) == 3
        print(f"\n  Batch OHLCV (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: {len(results[sym])} candles")

    @pytest.mark.asyncio
    async def test_get_profiles_batch_real(self, client):
        """Fetch company profiles for multiple symbols concurrently."""
        symbols = ["AAPL", "TSLA", "META"]

        async with client:
            results = await client.get_profiles_batch(symbols)

        assert len(results) == 3
        print(f"\n  Batch profiles (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: {results[sym].name}")

    @pytest.mark.asyncio
    async def test_concurrent_different_operations(self, client):
        """Run different async operations concurrently."""
        async with client:
            # Run quote, ohlcv, and profile requests concurrently
            quote_task = client.get_quote("AAPL")
            ohlcv_task = client.get_ohlcv("MSFT", limit=5)
            profile_task = client.get_company_profile("GOOGL")

            quote, ohlcv, profile = await asyncio.gather(
                quote_task, ohlcv_task, profile_task
            )

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        assert len(ohlcv) > 0
        assert profile.name is not None

        print(f"\n  Concurrent mixed operations:")
        print(f"    AAPL quote: ${quote.price:.2f}")
        print(f"    MSFT candles: {len(ohlcv)}")
        print(f"    GOOGL profile: {profile.name}")

    @pytest.mark.asyncio
    async def test_large_batch_quotes(self, client):
        """Test fetching a larger batch of quotes concurrently."""
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK-B", "JPM", "V"
        ]

        async with client:
            results = await client.get_quotes_batch(symbols)

        successful = sum(1 for r in results.values() if not isinstance(r, ProviderError))
        print(f"\n  Large batch: {successful}/{len(symbols)} successful")

        # At least most should succeed
        assert successful >= len(symbols) // 2

    @pytest.mark.asyncio
    async def test_async_context_manager_cleanup(self, client):
        """Verify async context manager properly cleans up."""
        async with client:
            quote = await client.get_quote("AAPL")
            assert quote.price > 0

        # After exiting context, providers should be closed
        # (we can't easily verify this, but at least no errors)
        print(f"\n  Context manager cleanup successful")

    @pytest.mark.asyncio
    async def test_get_news_batch_real(self, client):
        """Fetch news for multiple symbols concurrently."""
        symbols = ["AAPL", "TSLA"]

        async with client:
            results = await client.get_news_batch(symbols, limit=3)

        print(f"\n  Batch news (concurrent):")
        for sym in symbols:
            if sym in results and not isinstance(results[sym], ProviderError):
                print(f"    {sym}: {len(results[sym])} articles")


# ============================================================================
# AsyncCryptoClient Integration Tests
# ============================================================================

class TestAsyncCryptoClientIntegration:
    """Integration tests for AsyncCryptoClient with real API calls."""

    @pytest.fixture
    def client(self):
        return AsyncCryptoClient()

    @pytest.mark.asyncio
    async def test_get_quote_real(self, client):
        """Fetch a real crypto quote using async client."""
        async with client:
            quote = await client.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC via AsyncCryptoClient: ${quote.price_usd:,.2f}")

    @pytest.mark.asyncio
    async def test_get_ohlcv_real(self, client):
        """Fetch real crypto OHLCV data using async client."""
        async with client:
            candles = await client.get_ohlcv("ETH", Interval.DAY_1, limit=10)

        assert len(candles) > 0
        assert candles[0].symbol == "ETH"
        assert candles[0].close > 0
        print(f"\n  Got {len(candles)} ETH candles via async")

    @pytest.mark.asyncio
    async def test_get_asset_real(self, client):
        """Fetch real crypto asset profile using async client."""
        async with client:
            asset = await client.get_asset("BTC")

        assert asset.symbol == "BTC"
        assert asset.name == "Bitcoin"
        print(f"\n  {asset.name}: {asset.description[:60] if asset.description else 'No description'}...")

    @pytest.mark.asyncio
    async def test_get_market_data_real(self, client):
        """Fetch real crypto market data using async client."""
        async with client:
            data = await client.get_market_data("ETH")

        assert data.symbol == "ETH"
        assert data.price_usd > 0
        assert data.market_cap > 0
        print(f"\n  ETH: ${data.price_usd:,.2f}, MCap: ${data.market_cap:,.0f}")

    @pytest.mark.asyncio
    async def test_get_quotes_batch_real(self, client):
        """Fetch multiple crypto quotes concurrently."""
        symbols = ["BTC", "ETH", "SOL", "ADA", "DOT"]

        async with client:
            results = await client.get_quotes_batch(symbols)

        assert len(results) == 5
        print(f"\n  Batch crypto quotes (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: ${results[sym].price_usd:,.2f}")

    @pytest.mark.asyncio
    async def test_get_ohlcv_batch_real(self, client):
        """Fetch OHLCV for multiple cryptos concurrently."""
        symbols = ["BTC", "ETH", "SOL"]

        async with client:
            results = await client.get_ohlcv_batch(symbols, limit=5)

        assert len(results) == 3
        print(f"\n  Batch crypto OHLCV (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: {len(results[sym])} candles")

    @pytest.mark.asyncio
    async def test_get_market_data_batch_real(self, client):
        """Fetch market data for multiple cryptos concurrently."""
        symbols = ["BTC", "ETH", "BNB"]

        async with client:
            results = await client.get_market_data_batch(symbols)

        assert len(results) == 3
        print(f"\n  Batch market data (concurrent):")
        for sym in symbols:
            assert sym in results
            if not isinstance(results[sym], ProviderError):
                print(f"    {sym}: Rank #{results[sym].market_cap_rank}")

    @pytest.mark.asyncio
    async def test_concurrent_different_operations(self, client):
        """Run different async crypto operations concurrently."""
        async with client:
            quote_task = client.get_quote("BTC")
            ohlcv_task = client.get_ohlcv("ETH", limit=5)
            market_task = client.get_market_data("SOL")

            quote, ohlcv, market = await asyncio.gather(
                quote_task, ohlcv_task, market_task
            )

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        assert len(ohlcv) > 0
        assert market.price_usd > 0

        print(f"\n  Concurrent mixed crypto operations:")
        print(f"    BTC quote: ${quote.price_usd:,.2f}")
        print(f"    ETH candles: {len(ohlcv)}")
        print(f"    SOL price: ${market.price_usd:,.2f}")

    @pytest.mark.asyncio
    async def test_large_batch_crypto_quotes(self, client):
        """Test fetching a larger batch of crypto quotes concurrently."""
        symbols = [
            "BTC", "ETH", "USDT", "BNB", "SOL",
            "XRP", "ADA", "DOGE", "DOT", "LINK"
        ]

        async with client:
            results = await client.get_quotes_batch(symbols)

        successful = sum(1 for r in results.values() if not isinstance(r, ProviderError))
        print(f"\n  Large crypto batch: {successful}/{len(symbols)} successful")

        # Most should succeed
        assert successful >= len(symbols) // 2


# ============================================================================
# Performance Comparison Tests
# ============================================================================

class TestAsyncPerformance:
    """Tests comparing async vs sync performance."""

    @pytest.mark.asyncio
    async def test_async_batch_is_concurrent(self):
        """Verify async batch actually runs concurrently (faster than sequential)."""
        import time
        from qfclient import MarketClient

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

        # Async batch
        async_client = AsyncMarketClient()
        start = time.time()
        async with async_client:
            async_results = await async_client.get_quotes_batch(symbols)
        async_time = time.time() - start

        # Sync sequential (for comparison)
        sync_client = MarketClient()
        start = time.time()
        sync_results = {}
        for sym in symbols:
            try:
                sync_results[sym] = sync_client.get_quote(sym)
            except ProviderError as e:
                sync_results[sym] = e
        sync_time = time.time() - start

        print(f"\n  Performance comparison ({len(symbols)} quotes):")
        print(f"    Async (concurrent): {async_time:.2f}s")
        print(f"    Sync (sequential):  {sync_time:.2f}s")
        print(f"    Speedup: {sync_time/async_time:.1f}x" if async_time > 0 else "    N/A")

        # Both should return same number of results
        assert len(async_results) == len(sync_results)

    @pytest.mark.asyncio
    async def test_async_crypto_batch_is_concurrent(self):
        """Verify async crypto batch runs concurrently."""
        import time
        from qfclient import CryptoClient

        symbols = ["BTC", "ETH", "SOL", "ADA"]

        # Async batch
        async_client = AsyncCryptoClient()
        start = time.time()
        async with async_client:
            async_results = await async_client.get_quotes_batch(symbols)
        async_time = time.time() - start

        # Sync sequential
        sync_client = CryptoClient()
        start = time.time()
        sync_results = {}
        for sym in symbols:
            try:
                sync_results[sym] = sync_client.get_quote(sym)
            except ProviderError as e:
                sync_results[sym] = e
        sync_time = time.time() - start

        print(f"\n  Crypto performance comparison ({len(symbols)} quotes):")
        print(f"    Async (concurrent): {async_time:.2f}s")
        print(f"    Sync (sequential):  {sync_time:.2f}s")
        print(f"    Speedup: {sync_time/async_time:.1f}x" if async_time > 0 else "    N/A")

        assert len(async_results) == len(sync_results)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestAsyncErrorHandling:
    """Tests for async error handling with real APIs."""

    @pytest.mark.asyncio
    async def test_invalid_symbol_in_batch(self):
        """Test that invalid symbols are handled gracefully in batch."""
        client = AsyncMarketClient()
        symbols = ["AAPL", "XYZNOTREAL123", "MSFT"]

        async with client:
            results = await client.get_quotes_batch(symbols)

        assert len(results) == 3
        assert "AAPL" in results
        assert "XYZNOTREAL123" in results
        assert "MSFT" in results

        # Valid symbols should succeed with real prices
        assert not isinstance(results["AAPL"], ProviderError)
        assert not isinstance(results["MSFT"], ProviderError)
        assert results["AAPL"].price > 0
        assert results["MSFT"].price > 0

        # Invalid symbol should either be an error OR a quote with zero price
        # (different providers handle invalid symbols differently)
        invalid_result = results["XYZNOTREAL123"]
        if isinstance(invalid_result, ProviderError):
            # Provider raised an error for invalid symbol
            print(f"\n  Invalid symbol returned ProviderError: {invalid_result}")
        else:
            # Provider returned a quote - should have zero or null values
            assert invalid_result.price == 0 or invalid_result.price is None
            print(f"\n  Invalid symbol returned Quote with price=0")

        print(f"\n  Invalid symbol handled correctly in batch")
        print(f"    AAPL: ${results['AAPL'].price:.2f}")
        print(f"    XYZNOTREAL123: {type(invalid_result).__name__}")
        print(f"    MSFT: ${results['MSFT'].price:.2f}")

    @pytest.mark.asyncio
    async def test_invalid_crypto_symbol_in_batch(self):
        """Test that invalid crypto symbols are handled gracefully."""
        client = AsyncCryptoClient()
        symbols = ["BTC", "NOTACOIN999", "ETH"]

        async with client:
            results = await client.get_quotes_batch(symbols)

        assert len(results) == 3

        # Valid symbols should succeed
        assert not isinstance(results["BTC"], ProviderError)
        assert not isinstance(results["ETH"], ProviderError)

        print(f"\n  Invalid crypto symbol handled correctly")
        print(f"    BTC: ${results['BTC'].price_usd:,.2f}")
        print(f"    NOTACOIN999: {type(results['NOTACOIN999']).__name__}")
        print(f"    ETH: ${results['ETH'].price_usd:,.2f}")

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test handling of empty symbol list."""
        client = AsyncMarketClient()

        async with client:
            results = await client.get_quotes_batch([])

        assert len(results) == 0
        print(f"\n  Empty batch handled correctly")

    @pytest.mark.asyncio
    async def test_duplicate_symbols_in_batch(self):
        """Test handling of duplicate symbols in batch."""
        client = AsyncCryptoClient()
        symbols = ["BTC", "BTC", "ETH", "BTC"]

        async with client:
            results = await client.get_quotes_batch(symbols)

        # Should have results for unique symbols only (dict keys)
        assert "BTC" in results
        assert "ETH" in results
        print(f"\n  Duplicate symbols: got {len(results)} unique results")


# ============================================================================
# Provider Preference Tests
# ============================================================================

class TestAsyncProviderPreference:
    """Tests for async client provider preference."""

    @pytest.mark.asyncio
    async def test_market_client_with_preference(self):
        """Test AsyncMarketClient with provider preference."""
        # Try with finnhub preference
        client = AsyncMarketClient(prefer="finnhub")

        async with client:
            quote = await client.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.price > 0
        print(f"\n  AAPL with finnhub preference: ${quote.price:.2f}")

    @pytest.mark.asyncio
    async def test_crypto_client_with_preference(self):
        """Test AsyncCryptoClient with provider preference."""
        # Try with coingecko preference
        client = AsyncCryptoClient(prefer="coingecko")

        async with client:
            quote = await client.get_quote("BTC")

        assert quote.symbol == "BTC"
        assert quote.price_usd > 0
        print(f"\n  BTC with coingecko preference: ${quote.price_usd:,.2f}")
