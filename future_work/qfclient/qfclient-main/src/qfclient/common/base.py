"""
Base classes for qfclient models and result containers.
"""

from datetime import datetime, timezone
from typing import TypeVar, Generic, Iterator, overload
from pydantic import BaseModel, Field

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


T = TypeVar("T", bound=BaseModel)


class ResultList(list, Generic[T]):
    """
    A list container with DataFrame conversion support.

    Usage:
        candles = client.get_ohlcv("AAPL", "2024-01-01", "2024-12-31")
        candles[0]           # Access individual OHLCV model
        candles.to_df()      # Convert to DataFrame for analysis
    """

    def __init__(self, items: list[T] | None = None, provider: str | None = None):
        super().__init__(items or [])
        self.provider = provider
        self.fetched_at = datetime.now(timezone.utc)

    def to_df(self) -> "pd.DataFrame":
        """
        Convert the list of models to a pandas DataFrame.

        Returns:
            DataFrame with model fields as columns.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required for DataFrame conversion. "
                "Install with: pip install pandas"
            )

        if not self:
            return pd.DataFrame()

        return pd.DataFrame([item.model_dump() for item in self])

    def to_dicts(self) -> list[dict]:
        """Convert to a list of dictionaries."""
        return [item.model_dump() for item in self]

    def to_json(self) -> str:
        """Convert to a JSON string."""
        import json
        return json.dumps(self.to_dicts(), default=str)

    def filter(self, predicate) -> "ResultList[T]":
        """Filter items by a predicate function."""
        filtered = ResultList([item for item in self if predicate(item)])
        filtered.provider = self.provider
        filtered.fetched_at = self.fetched_at
        return filtered

    def first(self) -> T | None:
        """Get the first item or None."""
        return self[0] if self else None

    def last(self) -> T | None:
        """Get the last item or None."""
        return self[-1] if self else None


class DataSource(BaseModel):
    """Metadata about the data source."""
    provider: str = Field(..., description="Provider name (e.g., 'alpaca', 'finnhub')")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When data was fetched")
    latency_ms: float | None = Field(default=None, description="API response latency in ms")


class ProviderError(Exception):
    """Exception raised when a provider request fails."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    """Exception raised when rate limited by a provider."""

    def __init__(self, provider: str, retry_after: float | None = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f" (retry after {retry_after:.1f}s)"
        super().__init__(provider, message, status_code=429)


class ConfigurationError(Exception):
    """Exception raised for configuration errors (e.g., missing API key)."""
    pass
