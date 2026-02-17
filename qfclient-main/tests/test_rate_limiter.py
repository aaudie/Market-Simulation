"""Tests for rate limiter functionality."""

import time
from unittest.mock import patch

import pytest

from qfclient.common.rate_limiter import (
    RateLimiter,
    ProviderLimits,
    RateLimitHeaderParser,
    ParsedRateLimit,
    RequestWindow,
    get_limiter,
    reset_limiter,
)


# ============================================================================
# ProviderLimits Tests
# ============================================================================

class TestProviderLimits:
    """Tests for ProviderLimits dataclass."""

    def test_default_values(self):
        """ProviderLimits should have sensible defaults."""
        limits = ProviderLimits()
        assert limits.requests_per_minute is None
        assert limits.requests_per_hour is None
        assert limits.requests_per_day is None
        assert limits.priority == 50
        assert limits.enabled is True

    def test_update_dynamic(self):
        """update_dynamic should update dynamic limit fields."""
        limits = ProviderLimits()
        limits.update_dynamic(remaining=10, limit=100, reset_at=1000.0, window="minute")

        assert limits.dynamic_remaining == 10
        assert limits.dynamic_limit == 100
        assert limits.dynamic_reset_at == 1000.0
        assert limits.dynamic_window == "minute"

    def test_clear_dynamic(self):
        """clear_dynamic should clear all dynamic fields."""
        limits = ProviderLimits()
        limits.update_dynamic(remaining=10, limit=100, reset_at=1000.0)
        limits.clear_dynamic()

        assert limits.dynamic_remaining is None
        assert limits.dynamic_limit is None
        assert limits.dynamic_reset_at is None

    def test_has_dynamic_info(self):
        """has_dynamic_info should return True when dynamic info is set."""
        limits = ProviderLimits()
        assert limits.has_dynamic_info is False

        limits.update_dynamic(remaining=10)
        assert limits.has_dynamic_info is True


# ============================================================================
# RequestWindow Tests
# ============================================================================

class TestRequestWindow:
    """Tests for RequestWindow sliding window."""

    def test_add_timestamp(self):
        """add should add timestamps to the window."""
        window = RequestWindow()
        window.add(100.0)
        window.add(200.0)

        assert len(window.timestamps) == 2

    def test_count_in_window(self):
        """count_in_window should count requests within time period."""
        window = RequestWindow()
        now = time.time()

        # Add requests in chronological order (oldest first)
        # Note: The deque-based removal only works when entries are in order
        window.add(now - 70)  # 70 seconds ago (outside 1-minute window)
        window.add(now - 30)  # 30 seconds ago
        window.add(now - 10)  # 10 seconds ago

        # Count within 60 second window
        count = window.count_in_window(60)
        assert count == 2  # Only the two within 60 seconds

    def test_count_removes_old_entries(self):
        """count_in_window should remove old entries."""
        window = RequestWindow()
        now = time.time()

        window.add(now - 120)  # 2 minutes ago
        window.add(now - 10)   # 10 seconds ago

        # After counting, old entry should be removed
        window.count_in_window(60)
        assert len(window.timestamps) == 1


# ============================================================================
# RateLimitHeaderParser Tests
# ============================================================================

class TestRateLimitHeaderParser:
    """Tests for rate limit header parsing."""

    def test_parse_standard_headers(self):
        """Should parse standard rate limit headers."""
        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "95",
            "X-RateLimit-Reset": str(time.time() + 60),
        }

        result = RateLimitHeaderParser.parse(headers)

        assert result is not None
        assert result.limit == 100
        assert result.remaining == 95
        assert result.reset_at is not None

    def test_parse_lowercase_headers(self):
        """Should handle lowercase header names."""
        headers = {
            "x-ratelimit-limit": "100",
            "x-ratelimit-remaining": "50",
        }

        result = RateLimitHeaderParser.parse(headers)

        assert result is not None
        assert result.limit == 100
        assert result.remaining == 50

    def test_parse_tradier_headers(self):
        """Should parse Tradier-specific headers."""
        headers = {
            "X-Ratelimit-Allowed": "120",
            "X-Ratelimit-Available": "100",
            "X-Ratelimit-Used": "20",
        }

        result = RateLimitHeaderParser.parse(headers, provider="tradier")

        assert result is not None
        assert result.limit == 120
        assert result.remaining == 100

    def test_parse_twelve_data_headers(self):
        """Should parse Twelve Data credit headers."""
        headers = {
            "api-credits-left": "500",
            "api-credits-used": "300",
        }

        result = RateLimitHeaderParser.parse(headers, provider="twelve_data")

        assert result is not None
        assert result.remaining == 500
        assert result.window == "credits"

    def test_parse_429_with_retry_after(self):
        """Should parse Retry-After header on 429 response."""
        headers = {
            "Retry-After": "30",
        }

        result = RateLimitHeaderParser.parse(headers, status_code=429)

        assert result is not None
        assert result.remaining == 0
        assert result.reset_at is not None
        # Reset should be ~30 seconds from now
        assert result.reset_at > time.time()
        assert result.reset_at < time.time() + 35

    def test_parse_returns_none_for_empty_headers(self):
        """Should return None when no rate limit headers found."""
        headers = {"Content-Type": "application/json"}

        result = RateLimitHeaderParser.parse(headers)

        assert result is None


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def limiter(self):
        """Create a fresh rate limiter for testing."""
        return RateLimiter(limits={
            "test_provider": ProviderLimits(
                requests_per_minute=10,
                requests_per_day=100,
                priority=50
            ),
            "high_priority": ProviderLimits(
                requests_per_minute=100,
                priority=90
            ),
            "low_priority": ProviderLimits(
                requests_per_minute=5,
                priority=10
            ),
        })

    def test_can_request_with_no_limits(self, limiter):
        """Should allow requests for unknown providers."""
        assert limiter.can_request("unknown_provider") is True

    def test_can_request_within_limits(self, limiter):
        """Should allow requests within rate limits."""
        assert limiter.can_request("test_provider") is True

    def test_can_request_at_limit(self, limiter):
        """Should deny requests when at rate limit."""
        # Make requests up to the limit
        for _ in range(10):
            limiter.record_request("test_provider")

        assert limiter.can_request("test_provider") is False

    def test_record_request(self, limiter):
        """record_request should track requests."""
        limiter.record_request("test_provider")
        limiter.record_request("test_provider")

        status = limiter.get_status("test_provider")
        assert status["requests_minute"] == 2

    def test_record_failure_disables_provider(self, limiter):
        """record_failure should temporarily disable provider."""
        limiter.record_failure("test_provider", disable_seconds=5)

        assert limiter.can_request("test_provider") is False

    def test_select_provider_prefers_requested(self, limiter):
        """select_provider should prefer requested provider if available."""
        providers = ["test_provider", "high_priority"]

        selected = limiter.select_provider(providers, prefer="test_provider")

        assert selected == "test_provider"

    def test_select_provider_by_priority(self, limiter):
        """select_provider should select by priority when no preference."""
        providers = ["test_provider", "high_priority", "low_priority"]

        selected = limiter.select_provider(providers)

        assert selected == "high_priority"

    def test_select_provider_skips_rate_limited(self, limiter):
        """select_provider should skip rate-limited providers."""
        # Rate limit high priority provider
        for _ in range(100):
            limiter.record_request("high_priority")

        providers = ["test_provider", "high_priority"]

        selected = limiter.select_provider(providers)

        assert selected == "test_provider"

    def test_select_provider_returns_none_all_limited(self, limiter):
        """select_provider should return None when all providers limited."""
        # Rate limit both providers
        for _ in range(10):
            limiter.record_request("test_provider")
        for _ in range(5):
            limiter.record_request("low_priority")

        providers = ["test_provider", "low_priority"]

        selected = limiter.select_provider(providers)

        assert selected is None

    def test_time_until_available_when_available(self, limiter):
        """time_until_available should return 0 when provider is available."""
        wait = limiter.time_until_available("test_provider")
        assert wait == 0.0

    def test_time_until_available_when_rate_limited(self, limiter):
        """time_until_available should return positive value when limited."""
        # Rate limit the provider
        for _ in range(10):
            limiter.record_request("test_provider")

        wait = limiter.time_until_available("test_provider")

        # Should be > 0 but < 60 seconds
        assert wait > 0
        assert wait <= 60

    def test_update_from_headers(self, limiter):
        """update_from_headers should update dynamic limits."""
        headers = {
            "X-RateLimit-Remaining": "50",
            "X-RateLimit-Limit": "100",
        }

        updated = limiter.update_from_headers("test_provider", headers)

        assert updated is True
        assert limiter.limits["test_provider"].dynamic_remaining is not None

    def test_get_status(self, limiter):
        """get_status should return provider status dict."""
        limiter.record_request("test_provider")

        status = limiter.get_status("test_provider")

        assert "available" in status
        assert "enabled" in status
        assert "priority" in status
        assert "requests_minute" in status
        assert "limits" in status
        assert status["requests_minute"] == 1

    def test_disabled_provider_not_enabled(self, limiter):
        """Disabled providers should not be available."""
        limiter.limits["test_provider"].enabled = False

        assert limiter.can_request("test_provider") is False

    def test_dynamic_limits_take_precedence(self, limiter):
        """Dynamic limits should be checked before hardcoded limits."""
        # Set dynamic remaining to 0
        limiter.limits["test_provider"].update_dynamic(remaining=0)

        assert limiter.can_request("test_provider") is False


class TestRateLimiterDailyReset:
    """Tests for daily/monthly reset behavior."""

    def test_daily_count_reset(self):
        """Daily counts should reset when day changes."""
        limiter = RateLimiter(limits={
            "test": ProviderLimits(requests_per_day=5)
        })

        # Record some requests
        for _ in range(3):
            limiter.record_request("test")

        # Simulate day change by modifying the day start
        from datetime import datetime, timedelta, UTC
        limiter._day_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0
        ) - timedelta(days=1)

        # After reset, should be able to request again
        assert limiter.can_request("test") is True


# ============================================================================
# Global Limiter Tests
# ============================================================================

class TestGlobalLimiter:
    """Tests for global limiter singleton."""

    def test_get_limiter_returns_singleton(self):
        """get_limiter should return the same instance."""
        reset_limiter()

        limiter1 = get_limiter()
        limiter2 = get_limiter()

        assert limiter1 is limiter2

    def test_reset_limiter_clears_instance(self):
        """reset_limiter should clear the global instance."""
        limiter1 = get_limiter()
        reset_limiter()
        limiter2 = get_limiter()

        assert limiter1 is not limiter2

    def test_limiter_has_default_providers(self):
        """Default limiter should have common providers configured."""
        reset_limiter()
        limiter = get_limiter()

        assert "alpaca" in limiter.limits
        assert "finnhub" in limiter.limits
        assert "coingecko" in limiter.limits


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_requests(self):
        """Limiter should handle concurrent requests safely."""
        import threading

        limiter = RateLimiter(limits={
            "test": ProviderLimits(requests_per_minute=1000)
        })

        results = []

        def make_requests():
            for _ in range(100):
                limiter.record_request("test")
                results.append(limiter.can_request("test"))

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all requests
        status = limiter.get_status("test")
        assert status["requests_minute"] == 1000
