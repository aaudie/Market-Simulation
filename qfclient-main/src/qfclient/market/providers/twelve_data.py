"""
Twelve Data provider.

Rate limits: 8 requests/minute, 800/day (free tier)
Features: OHLCV, Quotes, Technical indicators
"""

import os
from datetime import date, datetime

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV
from .base import BaseProvider


def _interval_to_twelve(interval: Interval) -> str:
    """Convert interval to Twelve Data format."""
    mapping = {
        Interval.MINUTE_1: "1min",
        Interval.MINUTE_5: "5min",
        Interval.MINUTE_15: "15min",
        Interval.MINUTE_30: "30min",
        Interval.HOUR_1: "1h",
        Interval.HOUR_4: "4h",
        Interval.DAY_1: "1day",
        Interval.WEEK_1: "1week",
        Interval.MONTH_1: "1month",
    }
    return mapping.get(interval, "1day")


class TwelveDataProvider(BaseProvider):
    """
    Twelve Data provider.

    Provides:
    - Real-time and historical quotes
    - OHLCV data (up to 5000 data points)
    - Technical indicators
    - Forex and crypto data
    """

    provider_name = "twelve_data"
    base_url = "https://api.twelvedata.com"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add API key to params."""
        params = params or {}
        params["apikey"] = self.api_key
        return super().get(url, params=params, **kwargs)

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        data = self.get("/quote", params={"symbol": symbol})

        if data.get("status") == "error":
            raise ProviderError(self.provider_name, data.get("message", "Unknown error"))

        return Quote(
            symbol=symbol.upper(),
            price=float(data.get("close", 0)),
            open=float(data.get("open")) if data.get("open") else None,
            high=float(data.get("high")) if data.get("high") else None,
            low=float(data.get("low")) if data.get("low") else None,
            volume=int(data.get("volume")) if data.get("volume") else None,
            previous_close=float(data.get("previous_close")) if data.get("previous_close") else None,
            change=float(data.get("change")) if data.get("change") else None,
            change_percent=float(data.get("percent_change")) if data.get("percent_change") else None,
            timestamp=datetime.fromisoformat(data["datetime"]) if data.get("datetime") else None,
        )

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[OHLCV]:
        """Get OHLCV candle data."""
        params = {
            "symbol": symbol,
            "interval": _interval_to_twelve(interval),
            "outputsize": limit,
        }

        if start:
            params["start_date"] = start.isoformat()
        if end:
            params["end_date"] = end.isoformat()

        data = self.get("/time_series", params=params)

        if data.get("status") == "error":
            raise ProviderError(self.provider_name, data.get("message", "Unknown error"))

        values = data.get("values", [])

        candles = ResultList(provider=self.provider_name)
        for item in values:
            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=datetime.fromisoformat(item["datetime"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=int(item.get("volume", 0)),
                interval=interval,
            ))

        return candles
