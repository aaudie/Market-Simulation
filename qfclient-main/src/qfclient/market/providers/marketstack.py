"""
Marketstack provider.

Rate limits: 100 requests/month (free tier)
Features: Global EOD data, 72+ exchanges, 125,000+ tickers
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV
from .base import BaseProvider


class MarketstackProvider(BaseProvider):
    """
    Marketstack data provider.

    Provides:
    - End-of-day stock prices (72+ exchanges worldwide)
    - Historical data (30+ years for some markets)
    - Exchange and ticker metadata

    Note: Free tier is HTTP only (no HTTPS) and very limited (100 req/month).
    Best for international markets not covered by other providers.
    """

    provider_name = "marketstack"
    # Free tier uses HTTP only
    base_url = "http://api.marketstack.com/v1"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("MARKETSTACK_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add access key to params."""
        params = params or {}
        params["access_key"] = self.api_key
        return super().get(url, params=params, **kwargs)

    def _check_response(self, data: dict) -> dict:
        """Check API response for errors."""
        if "error" in data:
            error = data["error"]
            msg = error.get("message", error.get("info", "Unknown error"))
            raise ProviderError(self.provider_name, msg)
        return data

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """
        Get the latest EOD quote for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL" for US, "AAPL.XNAS" for specific exchange)

        Returns:
            Quote with latest EOD data
        """
        params = {
            "symbols": symbol.upper(),
            "limit": 1,
        }

        data = self.get("/eod/latest", params=params)
        self._check_response(data)

        results = data.get("data", [])
        if not results:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        bar = results[0]

        return Quote(
            symbol=bar.get("symbol", symbol.upper()),
            price=bar.get("close", 0),
            open=bar.get("open"),
            high=bar.get("high"),
            low=bar.get("low"),
            volume=bar.get("volume"),
            adj_close=bar.get("adj_close"),
            adj_high=bar.get("adj_high"),
            adj_low=bar.get("adj_low"),
            adj_open=bar.get("adj_open"),
            exchange=bar.get("exchange"),
            timestamp=datetime.fromisoformat(
                bar.get("date", "").replace("T", " ").split("+")[0]
            ) if bar.get("date") else None,
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

        Note: Marketstack only supports daily data on free tier.

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval (only DAY_1 on free tier)
            start: Start date
            end: End date
            limit: Maximum number of bars

        Returns:
            ResultList of OHLCV candles
        """
        if interval != Interval.DAY_1:
            raise ProviderError(
                self.provider_name,
                f"Interval {interval.value} not supported. Only daily data available."
            )

        params = {
            "symbols": symbol.upper(),
            "limit": min(limit, 1000),
        }

        if start:
            params["date_from"] = start.isoformat()
        if end:
            params["date_to"] = end.isoformat()

        data = self.get("/eod", params=params)
        self._check_response(data)

        candles = ResultList(provider=self.provider_name)

        for bar in data.get("data", []):
            try:
                # Parse timestamp
                date_str = bar.get("date", "")
                if "T" in date_str:
                    timestamp = datetime.fromisoformat(date_str.replace("T", " ").split("+")[0])
                else:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")

                candles.append(OHLCV(
                    symbol=bar.get("symbol", symbol.upper()),
                    timestamp=timestamp,
                    open=bar.get("open", 0),
                    high=bar.get("high", 0),
                    low=bar.get("low", 0),
                    close=bar.get("adj_close") or bar.get("close", 0),
                    volume=int(bar.get("volume", 0)),
                    exchange=bar.get("exchange"),
                    interval=interval,
                ))
            except (ValueError, TypeError, KeyError):
                continue

        # Marketstack returns newest first, reverse to chronological
        candles.reverse()
        return candles

    def get_tickers(
        self,
        exchange: str | None = None,
        search: str | None = None,
        limit: int = 100,
    ) -> ResultList[dict]:
        """
        Get list of available tickers.

        Args:
            exchange: Filter by exchange code (e.g., "XNAS", "XNYS", "XLON")
            search: Search query for ticker or company name
            limit: Maximum tickers to return

        Returns:
            ResultList of ticker dicts
        """
        params = {"limit": min(limit, 1000)}

        if exchange:
            params["exchange"] = exchange
        if search:
            params["search"] = search

        data = self.get("/tickers", params=params)
        self._check_response(data)

        tickers = ResultList(provider=self.provider_name)
        for item in data.get("data", []):
            tickers.append({
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "exchange": item.get("stock_exchange", {}).get("mic"),
                "exchange_name": item.get("stock_exchange", {}).get("name"),
                "country": item.get("stock_exchange", {}).get("country"),
            })

        return tickers

    def get_exchanges(self) -> ResultList[dict]:
        """Get list of supported exchanges."""
        data = self.get("/exchanges")
        self._check_response(data)

        exchanges = ResultList(provider=self.provider_name)
        for item in data.get("data", []):
            exchanges.append({
                "mic": item.get("mic"),
                "name": item.get("name"),
                "acronym": item.get("acronym"),
                "country": item.get("country"),
                "city": item.get("city"),
                "timezone": item.get("timezone", {}).get("timezone"),
            })

        return exchanges
