"""
Alpaca Markets provider.

Rate limits: 200 requests/minute
Features: Real-time quotes, OHLCV (daily + intraday), split-adjusted data
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList
from ...common.types import Interval
from ..models import Quote, OHLCV
from .base import BaseProvider


def _interval_to_alpaca(interval: Interval) -> str:
    """Convert interval to Alpaca timeframe."""
    mapping = {
        Interval.MINUTE_1: "1Min",
        Interval.MINUTE_5: "5Min",
        Interval.MINUTE_15: "15Min",
        Interval.MINUTE_30: "30Min",
        Interval.HOUR_1: "1Hour",
        Interval.HOUR_4: "4Hour",
        Interval.DAY_1: "1Day",
        Interval.WEEK_1: "1Week",
        Interval.MONTH_1: "1Month",
    }
    return mapping.get(interval, "1Day")


class AlpacaProvider(BaseProvider):
    """
    Alpaca Markets data provider.

    Provides high-quality market data with:
    - Real-time quotes (IEX or SIP depending on subscription)
    - Historical bars (daily and intraday)
    - Split-adjusted data
    - High rate limits (200/min)
    """

    provider_name = "alpaca"
    base_url = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("ALPACA_API_KEY_ID")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _get_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.api_secret or "",
        }

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        data = self.get(f"/stocks/{symbol}/quotes/latest")

        quote_data = data.get("quote", {})

        # Use ask price if available, otherwise bid price (for after-hours)
        ask = quote_data.get("ap", 0)
        bid = quote_data.get("bp", 0)
        price = ask if ask > 0 else bid

        return Quote(
            symbol=symbol.upper(),
            price=price,
            bid=bid if bid > 0 else None,
            ask=ask if ask > 0 else None,
            bid_size=quote_data.get("bs"),
            ask_size=quote_data.get("as"),
            timestamp=datetime.fromisoformat(
                quote_data.get("t", "").replace("Z", "+00:00")
            ) if quote_data.get("t") else None,
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

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval
            start: Start date (defaults to 1 year ago)
            end: End date (defaults to today)
            limit: Maximum number of bars to return

        Returns:
            ResultList of OHLCV candles
        """
        params = {
            "timeframe": _interval_to_alpaca(interval),
            "limit": limit,
            "adjustment": "split",
        }

        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        data = self.get(f"/stocks/{symbol}/bars", params=params)

        bars = data.get("bars", [])
        candles = ResultList(provider=self.provider_name)

        for bar in bars:
            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=bar["v"],
                vwap=bar.get("vw"),
                trades=bar.get("n"),
                interval=interval,
            ))

        return candles
