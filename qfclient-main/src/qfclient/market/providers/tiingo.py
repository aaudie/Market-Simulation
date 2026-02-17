"""
Tiingo provider.

Rate limits: 1000 requests/hour, 500 unique symbols/month (free tier)
Features: Quotes (IEX), OHLCV (daily), Company metadata, News
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV, CompanyProfile
from .base import BaseProvider


class TiingoProvider(BaseProvider):
    """
    Tiingo data provider.

    Provides:
    - Real-time IEX quotes
    - Historical daily OHLCV data (20+ years)
    - Company metadata
    - News feed

    Note: Free tier limited to 500 unique symbols per month.
    Best for focused watchlists rather than universe scanning.
    """

    provider_name = "tiingo"
    base_url = "https://api.tiingo.com"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    @property
    def supports_company_profile(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """
        Get the latest IEX quote for a symbol.

        Note: Returns IEX exchange data only (subset of market volume).
        """
        data = self.get(f"/iex/{symbol}")

        if not data or len(data) == 0:
            raise ProviderError(self.provider_name, f"No IEX data for {symbol}")

        quote_data = data[0] if isinstance(data, list) else data

        # Parse timestamp
        timestamp = None
        if quote_data.get("timestamp"):
            try:
                ts_str = quote_data["timestamp"]
                if "." in ts_str:
                    ts_str = ts_str[:ts_str.index("+")] if "+" in ts_str else ts_str
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", ""))
                else:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Quote(
            symbol=symbol.upper(),
            price=quote_data.get("last") or quote_data.get("tngoLast") or 0,
            open=quote_data.get("open"),
            high=quote_data.get("high"),
            low=quote_data.get("low"),
            volume=quote_data.get("volume"),
            bid=quote_data.get("bidPrice"),
            ask=quote_data.get("askPrice"),
            bid_size=quote_data.get("bidSize"),
            ask_size=quote_data.get("askSize"),
            previous_close=quote_data.get("prevClose"),
            timestamp=timestamp,
        )

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[OHLCV]:
        """
        Get OHLCV bars for a symbol.

        Note: Tiingo free tier only supports daily data.
        Intraday (IEX) data has limited history.

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval (only DAY_1 fully supported on free tier)
            start: Start date
            end: End date
            limit: Maximum number of bars to return

        Returns:
            ResultList of OHLCV candles
        """
        # Tiingo free tier only supports daily data
        if interval not in (Interval.DAY_1, Interval.WEEK_1, Interval.MONTH_1):
            raise ProviderError(
                self.provider_name,
                f"Interval {interval.value} not supported on free tier. Use daily/weekly/monthly."
            )

        params = {}

        if start:
            params["startDate"] = start.isoformat()
        else:
            # Default to 1 year of data
            params["startDate"] = (date.today() - timedelta(days=365)).isoformat()

        if end:
            params["endDate"] = end.isoformat()

        # Tiingo uses resampleFreq for weekly/monthly
        if interval == Interval.WEEK_1:
            params["resampleFreq"] = "weekly"
        elif interval == Interval.MONTH_1:
            params["resampleFreq"] = "monthly"

        data = self.get(f"/tiingo/daily/{symbol}/prices", params=params)

        if not data:
            raise ProviderError(self.provider_name, f"No OHLCV data for {symbol}")

        candles = ResultList(provider=self.provider_name)

        for bar in data[-limit:]:  # Take last N bars
            try:
                # Parse date
                date_str = bar.get("date", "")
                if "T" in date_str:
                    timestamp = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")

                candles.append(OHLCV(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open=bar.get("adjOpen") or bar.get("open") or 0,
                    high=bar.get("adjHigh") or bar.get("high") or 0,
                    low=bar.get("adjLow") or bar.get("low") or 0,
                    close=bar.get("adjClose") or bar.get("close") or 0,
                    volume=int(bar.get("adjVolume") or bar.get("volume") or 0),
                    interval=interval,
                ))
            except (ValueError, KeyError, TypeError):
                continue

        return candles

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company metadata from Tiingo."""
        data = self.get(f"/tiingo/daily/{symbol}")

        if not data:
            raise ProviderError(self.provider_name, f"No profile data for {symbol}")

        # Parse start/end dates
        start_date = None
        if data.get("startDate"):
            try:
                start_date = datetime.strptime(data["startDate"], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                pass

        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            exchange=data.get("exchangeCode"),
            description=data.get("description"),
            ipo_date=start_date,
        )
