"""
Polygon.io provider (formerly Massive).

Rate limits: 5 requests/minute (free tier)
Features: Quotes, OHLCV, Reference data, Options (paid)
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV, CompanyProfile
from .base import BaseProvider


def _interval_to_polygon(interval: Interval) -> tuple[str, str]:
    """
    Convert interval to Polygon multiplier and timespan.

    Returns:
        Tuple of (multiplier, timespan)
    """
    mapping = {
        Interval.MINUTE_1: ("1", "minute"),
        Interval.MINUTE_5: ("5", "minute"),
        Interval.MINUTE_15: ("15", "minute"),
        Interval.MINUTE_30: ("30", "minute"),
        Interval.HOUR_1: ("1", "hour"),
        Interval.HOUR_4: ("4", "hour"),
        Interval.DAY_1: ("1", "day"),
        Interval.WEEK_1: ("1", "week"),
        Interval.MONTH_1: ("1", "month"),
    }
    return mapping.get(interval, ("1", "day"))


class PolygonProvider(BaseProvider):
    """
    Polygon.io data provider (also known as Massive).

    Provides:
    - Real-time quotes (delayed on free tier)
    - Historical OHLCV data
    - Reference data (tickers, exchanges)
    - Options data (paid tiers)

    Note: Free tier is limited to 5 req/min and EOD data only.
    Paid tiers ($29+) unlock real-time and full historical.
    """

    provider_name = "polygon"
    base_url = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        # Support both naming conventions
        self.api_key = api_key or os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add API key to params."""
        params = params or {}
        params["apiKey"] = self.api_key
        return super().get(url, params=params, **kwargs)

    def _check_response(self, data: dict) -> dict:
        """Check API response for errors."""
        if data.get("status") == "ERROR":
            msg = data.get("error", data.get("message", "Unknown error"))
            raise ProviderError(self.provider_name, msg)
        return data

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
        """Get the previous day's OHLCV (free tier doesn't have real-time)."""
        symbol = symbol.upper()
        data = self.get(f"/v2/aggs/ticker/{symbol}/prev")
        self._check_response(data)

        results = data.get("results", [])
        if not results:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        bar = results[0]

        return Quote(
            symbol=symbol,
            price=bar.get("c", 0),  # Close as price
            open=bar.get("o"),
            high=bar.get("h"),
            low=bar.get("l"),
            volume=bar.get("v"),
            vwap=bar.get("vw"),
            timestamp=datetime.fromtimestamp(bar.get("t", 0) / 1000) if bar.get("t") else None,
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

        Note: Free tier only has access to 2 years of EOD data.
        Intraday requires paid subscription.

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval
            start: Start date
            end: End date
            limit: Maximum number of bars

        Returns:
            ResultList of OHLCV candles
        """
        symbol = symbol.upper()
        multiplier, timespan = _interval_to_polygon(interval)

        if not start:
            start = date.today() - timedelta(days=365)
        if not end:
            end = date.today()

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": min(limit, 50000),
        }

        data = self.get(
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.isoformat()}/{end.isoformat()}",
            params=params
        )
        self._check_response(data)

        candles = ResultList(provider=self.provider_name)

        for bar in data.get("results", [])[:limit]:
            try:
                timestamp = datetime.fromtimestamp(bar.get("t", 0) / 1000)

                candles.append(OHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=bar.get("o", 0),
                    high=bar.get("h", 0),
                    low=bar.get("l", 0),
                    close=bar.get("c", 0),
                    volume=int(bar.get("v", 0)),
                    vwap=bar.get("vw"),
                    transactions=bar.get("n"),
                    interval=interval,
                ))
            except (ValueError, TypeError):
                continue

        return candles

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company details from ticker reference."""
        symbol = symbol.upper()
        data = self.get(f"/v3/reference/tickers/{symbol}")
        self._check_response(data)

        result = data.get("results", {})
        if not result:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        # Parse market cap
        market_cap = result.get("market_cap")
        shares = result.get("share_class_shares_outstanding") or result.get("weighted_shares_outstanding")

        return CompanyProfile(
            symbol=symbol,
            name=result.get("name", ""),
            exchange=result.get("primary_exchange"),
            description=result.get("description"),
            market_cap=market_cap,
            shares_outstanding=shares,
            locale=result.get("locale"),
            currency=result.get("currency_name"),
            cik=result.get("cik"),
            composite_figi=result.get("composite_figi"),
            sic_code=result.get("sic_code"),
            sic_description=result.get("sic_description"),
            website=result.get("homepage_url"),
            phone=result.get("phone_number"),
            address=result.get("address", {}).get("address1"),
            city=result.get("address", {}).get("city"),
            state=result.get("address", {}).get("state"),
        )

    def get_tickers(
        self,
        market: str = "stocks",
        active: bool = True,
        limit: int = 100,
    ) -> ResultList[dict]:
        """
        Get list of tickers.

        Args:
            market: Market type (stocks, crypto, fx, otc)
            active: Only active tickers
            limit: Maximum tickers to return

        Returns:
            ResultList of ticker dicts
        """
        params = {
            "market": market,
            "active": str(active).lower(),
            "limit": min(limit, 1000),
        }

        data = self.get("/v3/reference/tickers", params=params)
        self._check_response(data)

        tickers = ResultList(provider=self.provider_name)
        for item in data.get("results", []):
            tickers.append({
                "symbol": item.get("ticker"),
                "name": item.get("name"),
                "type": item.get("type"),
                "exchange": item.get("primary_exchange"),
                "currency": item.get("currency_name"),
                "active": item.get("active"),
            })

        return tickers
