"""
Rate limiting for API providers.

Supports both:
- Hardcoded defaults (from provider documentation)
- Dynamic updates (from response headers when available)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProviderLimits:
    """Rate limit configuration for a provider."""

    # Hardcoded defaults (from documentation)
    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    requests_per_month: int | None = None
    priority: int = 50  # Higher = preferred (0-100)
    enabled: bool = True

    # Dynamic rate limit tracking (updated from response headers)
    dynamic_remaining: int | None = None
    dynamic_limit: int | None = None
    dynamic_reset_at: float | None = None
    dynamic_window: str | None = None

    # Whether to use dynamic limits when available
    use_dynamic: bool = True

    def update_dynamic(
        self,
        remaining: int | None = None,
        limit: int | None = None,
        reset_at: float | None = None,
        window: str | None = None,
    ):
        """Update dynamic rate limit info from response headers."""
        if remaining is not None:
            self.dynamic_remaining = remaining
        if limit is not None:
            self.dynamic_limit = limit
        if reset_at is not None:
            self.dynamic_reset_at = reset_at
        if window is not None:
            self.dynamic_window = window

    def clear_dynamic(self):
        """Clear dynamic rate limit info."""
        self.dynamic_remaining = None
        self.dynamic_limit = None
        self.dynamic_reset_at = None
        self.dynamic_window = None

    @property
    def has_dynamic_info(self) -> bool:
        """Check if we have dynamic rate limit info."""
        return self.dynamic_remaining is not None or self.dynamic_limit is not None


# Default rate limits for all providers (free tier)
PROVIDER_LIMITS: dict[str, ProviderLimits] = {
    # Market data providers
    "alpaca": ProviderLimits(requests_per_minute=200, priority=80),
    "finnhub": ProviderLimits(requests_per_minute=60, priority=75),
    "alpha_vantage": ProviderLimits(requests_per_minute=5, requests_per_day=25, priority=30),
    "twelve_data": ProviderLimits(requests_per_minute=8, requests_per_day=800, priority=60),
    "fmp": ProviderLimits(requests_per_day=250, priority=65),
    "tiingo": ProviderLimits(requests_per_hour=1000, priority=55),
    "tradier": ProviderLimits(requests_per_minute=120, priority=70),
    "fred": ProviderLimits(requests_per_minute=120, priority=90),
    "polygon": ProviderLimits(requests_per_minute=5, priority=60),
    "eodhd": ProviderLimits(requests_per_day=20, priority=25),
    "marketstack": ProviderLimits(requests_per_month=100, priority=20),
    "yfinance": ProviderLimits(requests_per_minute=60, priority=50),

    # Crypto providers
    "coingecko": ProviderLimits(requests_per_minute=30, requests_per_month=10000, priority=80),
    "coinmarketcap": ProviderLimits(requests_per_minute=30, requests_per_day=333, priority=70),
    "cryptocompare": ProviderLimits(requests_per_minute=50, priority=75),
    "messari": ProviderLimits(requests_per_minute=30, requests_per_day=2000, priority=65),
}


@dataclass
class ParsedRateLimit:
    """Parsed rate limit information from response headers."""
    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None
    window: str | None = None


class RateLimitHeaderParser:
    """Parses rate limit headers from API responses."""

    PROVIDER_PATTERNS: dict[str, dict[str, list[str]]] = {
        "tradier": {
            "limit": ["X-Ratelimit-Allowed", "x-ratelimit-allowed"],
            "remaining": ["X-Ratelimit-Available", "x-ratelimit-available"],
            "reset": ["X-Ratelimit-Expiry", "x-ratelimit-expiry"],
            "used": ["X-Ratelimit-Used", "x-ratelimit-used"],
        },
        "twelve_data": {
            "remaining": ["api-credits-left", "Api-Credits-Left"],
            "used": ["api-credits-used", "Api-Credits-Used"],
        },
        "standard": {
            "limit": [
                "X-RateLimit-Limit", "x-ratelimit-limit",
                "RateLimit-Limit", "ratelimit-limit",
            ],
            "remaining": [
                "X-RateLimit-Remaining", "x-ratelimit-remaining",
                "RateLimit-Remaining", "ratelimit-remaining",
            ],
            "reset": [
                "X-RateLimit-Reset", "x-ratelimit-reset",
                "RateLimit-Reset", "ratelimit-reset",
            ],
        },
    }

    @classmethod
    def parse(
        cls,
        headers: dict[str, str],
        provider: str | None = None,
        status_code: int = 200,
    ) -> ParsedRateLimit | None:
        """Parse rate limit information from response headers."""
        result = ParsedRateLimit()
        found_any = False

        norm_headers = {k.lower(): v for k, v in headers.items()}

        # Try provider-specific patterns first
        if provider and provider in cls.PROVIDER_PATTERNS:
            found_any = cls._parse_with_patterns(
                norm_headers, cls.PROVIDER_PATTERNS[provider], result, provider
            )

        # Fall back to standard patterns
        if not found_any:
            found_any = cls._parse_with_patterns(
                norm_headers, cls.PROVIDER_PATTERNS["standard"], result
            )

        # Check for Retry-After header (especially on 429)
        if status_code == 429:
            retry_after = norm_headers.get("retry-after")
            if retry_after:
                try:
                    if retry_after.isdigit():
                        result.reset_at = time.time() + int(retry_after)
                    else:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(retry_after)
                        result.reset_at = dt.timestamp()
                    result.remaining = 0
                    found_any = True
                except (ValueError, TypeError):
                    pass

        return result if found_any else None

    @classmethod
    def _parse_with_patterns(
        cls,
        headers: dict[str, str],
        patterns: dict[str, list[str]],
        result: ParsedRateLimit,
        provider: str | None = None,
    ) -> bool:
        """Parse headers using specified patterns."""
        found_any = False

        # Parse limit
        for header_name in patterns.get("limit", []):
            value = headers.get(header_name.lower())
            if value:
                try:
                    result.limit = int(value)
                    found_any = True
                    break
                except ValueError:
                    pass

        # Parse remaining
        for header_name in patterns.get("remaining", []):
            value = headers.get(header_name.lower())
            if value:
                try:
                    result.remaining = int(value)
                    found_any = True
                    break
                except ValueError:
                    pass

        # Parse used (calculate remaining if we have limit)
        if result.remaining is None and result.limit is not None:
            for header_name in patterns.get("used", []):
                value = headers.get(header_name.lower())
                if value:
                    try:
                        used = int(value)
                        result.remaining = result.limit - used
                        found_any = True
                        break
                    except ValueError:
                        pass

        # Parse reset time
        for header_name in patterns.get("reset", []):
            value = headers.get(header_name.lower())
            if value:
                try:
                    reset_val = float(value)
                    if provider == "tradier" or reset_val > 1e12:
                        reset_val = reset_val / 1000
                    if reset_val < 1e9:
                        reset_val = time.time() + reset_val
                    result.reset_at = reset_val
                    found_any = True
                    break
                except ValueError:
                    pass

        # Determine window type
        if found_any:
            if provider == "twelve_data":
                result.window = "credits"
            elif result.reset_at:
                seconds_until_reset = result.reset_at - time.time()
                if seconds_until_reset <= 65:
                    result.window = "minute"
                elif seconds_until_reset <= 3700:
                    result.window = "hour"
                else:
                    result.window = "day"

        return found_any


@dataclass
class RequestWindow:
    """Sliding window for tracking requests in a time period."""
    timestamps: deque = field(default_factory=deque)

    def add(self, timestamp: float | None = None):
        """Add a request timestamp."""
        self.timestamps.append(timestamp or time.time())

    def count_in_window(self, window_seconds: float) -> int:
        """Count requests within the time window."""
        cutoff = time.time() - window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        return len(self.timestamps)


class RateLimiter:
    """
    Centralized rate limiting for all API providers.

    Thread-safe implementation supporting both hardcoded limits
    and dynamic updates from response headers.
    """

    def __init__(self, limits: dict[str, ProviderLimits] | None = None):
        self.limits = limits or PROVIDER_LIMITS.copy()
        self._windows: dict[str, RequestWindow] = {}
        self._daily_counts: dict[str, int] = {}
        self._monthly_counts: dict[str, int] = {}
        self._day_start: datetime = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._month_start: datetime = datetime.now(UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        self._lock = threading.RLock()
        self._disabled_until: dict[str, float] = {}

    def _get_window(self, provider: str) -> RequestWindow:
        """Get or create request window for provider."""
        if provider not in self._windows:
            self._windows[provider] = RequestWindow()
        return self._windows[provider]

    def _reset_daily_if_needed(self):
        """Reset daily counts if day has changed."""
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_start > self._day_start:
            self._daily_counts.clear()
            self._day_start = day_start

    def _reset_monthly_if_needed(self):
        """Reset monthly counts if month has changed."""
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_start > self._month_start:
            self._monthly_counts.clear()
            self._month_start = month_start

    def can_request(self, provider: str) -> bool:
        """Check if a request can be made to the provider."""
        with self._lock:
            if provider not in self.limits:
                return True

            limits = self.limits[provider]
            if not limits.enabled:
                return False

            # Check circuit breaker
            if provider in self._disabled_until:
                if time.time() <= self._disabled_until[provider]:
                    return False
                else:
                    del self._disabled_until[provider]

            # Check dynamic limits first
            if limits.use_dynamic and limits.has_dynamic_info:
                if limits.dynamic_reset_at and time.time() > limits.dynamic_reset_at:
                    limits.clear_dynamic()
                else:
                    if limits.has_dynamic_info and limits.dynamic_remaining is not None and limits.dynamic_remaining <= 0:
                        return False

            # Fall back to hardcoded limits
            window = self._get_window(provider)
            self._reset_daily_if_needed()
            self._reset_monthly_if_needed()

            if limits.requests_per_minute:
                if window.count_in_window(60) >= limits.requests_per_minute:
                    return False

            if limits.requests_per_hour:
                if window.count_in_window(3600) >= limits.requests_per_hour:
                    return False

            if limits.requests_per_day:
                daily = self._daily_counts.get(provider, 0)
                if daily >= limits.requests_per_day:
                    return False

            if limits.requests_per_month:
                monthly = self._monthly_counts.get(provider, 0)
                if monthly >= limits.requests_per_month:
                    return False

            return True

    def record_request(self, provider: str):
        """Record a request to a provider."""
        with self._lock:
            window = self._get_window(provider)
            window.add()
            self._daily_counts[provider] = self._daily_counts.get(provider, 0) + 1
            self._monthly_counts[provider] = self._monthly_counts.get(provider, 0) + 1

    def record_failure(self, provider: str, disable_seconds: float = 60):
        """Record a failure and temporarily disable provider."""
        with self._lock:
            self._disabled_until[provider] = time.time() + disable_seconds
            logger.warning(f"Provider {provider} disabled for {disable_seconds}s after failure")

    def update_from_headers(
        self,
        provider: str,
        headers: dict[str, str],
        status_code: int = 200,
    ) -> bool:
        """Update rate limit tracking from response headers."""
        with self._lock:
            parsed = RateLimitHeaderParser.parse(headers, provider, status_code)

            if not parsed:
                return False

            if provider not in self.limits:
                self.limits[provider] = ProviderLimits()

            limits = self.limits[provider]
            limits.update_dynamic(
                remaining=parsed.remaining,
                limit=parsed.limit,
                reset_at=parsed.reset_at,
                window=parsed.window,
            )

            if limits.dynamic_remaining is not None and limits.dynamic_remaining > 0:
                limits.dynamic_remaining -= 1

            if status_code == 429:
                if parsed.reset_at:
                    wait_seconds = max(1, parsed.reset_at - time.time())
                else:
                    wait_seconds = 60
                self._disabled_until[provider] = time.time() + wait_seconds

            return True

    def time_until_available(self, provider: str) -> float:
        """Get seconds until provider is available."""
        with self._lock:
            if provider not in self.limits:
                return 0.0

            limits = self.limits[provider]
            if not limits.enabled:
                return float('inf')

            if provider in self._disabled_until:
                remaining = self._disabled_until[provider] - time.time()
                if remaining >= 0:
                    return max(0, remaining)

            if limits.use_dynamic and limits.has_dynamic_info:
                if limits.dynamic_remaining is not None and limits.dynamic_remaining <= 0:
                    if limits.dynamic_reset_at:
                        wait = limits.dynamic_reset_at - time.time()
                        if wait > 0:
                            return wait
                    return 60.0

            window = self._get_window(provider)

            if limits.requests_per_minute:
                count = window.count_in_window(60)
                if count >= limits.requests_per_minute:
                    if window.timestamps:
                        oldest = window.timestamps[0]
                        wait = (oldest + 60) - time.time()
                        return max(0, wait)

            return 0.0

    def select_provider(
        self,
        providers: list[str],
        prefer: str | None = None,
    ) -> str | None:
        """Select the best available provider."""
        if prefer and prefer in providers and self.can_request(prefer):
            return prefer

        available = [p for p in providers if self.can_request(p)]
        available.sort(
            key=lambda p: self.limits.get(p, ProviderLimits()).priority,
            reverse=True
        )
        return available[0] if available else None

    def get_status(self, provider: str) -> dict[str, Any]:
        """Get status for a provider."""
        with self._lock:
            if provider not in self.limits:
                return {"available": True, "configured": False}

            limits = self.limits[provider]
            window = self._get_window(provider)

            return {
                "available": self.can_request(provider),
                "enabled": limits.enabled,
                "priority": limits.priority,
                "requests_minute": window.count_in_window(60),
                "requests_day": self._daily_counts.get(provider, 0),
                "limits": {
                    "per_minute": limits.requests_per_minute,
                    "per_day": limits.requests_per_day,
                },
                "seconds_until_available": self.time_until_available(provider),
            }


# Global rate limiter instance
_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def reset_limiter():
    """Reset the global rate limiter (for testing)."""
    global _limiter
    _limiter = None
