"""
Tradier provider.

Rate limits: 120 requests/minute
Features: OHLCV, Options chains with Greeks
"""

import os
from datetime import date, datetime

from ...common.base import ResultList
from ...common.types import Interval
from ..models import OHLCV, OptionChain, OptionContract
from .base import BaseProvider


class TradierProvider(BaseProvider):
    """
    Tradier data provider.

    Specializes in:
    - Options data with full Greeks
    - Historical price data
    - Real-time quotes

    Uses sandbox by default (free), set use_production=True for live data.
    """

    provider_name = "tradier"

    def __init__(
        self,
        api_key: str | None = None,
        use_production: bool = False,
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("TRADIER_API_KEY")
        self.use_production = use_production
        self.base_url = (
            "https://api.tradier.com/v1"
            if use_production
            else "https://sandbox.tradier.com/v1"
        )

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    @property
    def supports_ohlcv(self) -> bool:
        return True

    @property
    def supports_options(self) -> bool:
        return True

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[OHLCV]:
        """Get OHLCV candle data."""
        if interval in {Interval.MINUTE_1, Interval.MINUTE_5, Interval.MINUTE_15}:
            return self._fetch_timesales(symbol, interval, start, end, limit)
        else:
            return self._fetch_historical(symbol, interval, start, end, limit)

    def _fetch_historical(
        self,
        symbol: str,
        interval: Interval,
        start: date | None,
        end: date | None,
        limit: int,
    ) -> ResultList[OHLCV]:
        """Fetch daily/weekly/monthly data from history endpoint."""
        interval_map = {
            Interval.DAY_1: "daily",
            Interval.WEEK_1: "weekly",
            Interval.MONTH_1: "monthly",
        }
        tradier_interval = interval_map.get(interval, "daily")

        params = {"symbol": symbol, "interval": tradier_interval}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        data = self.get("/markets/history", params=params)

        history = data.get("history")
        if not history:
            return ResultList(provider=self.provider_name)

        days = history.get("day", [])
        if isinstance(days, dict):
            days = [days]

        candles = ResultList(provider=self.provider_name)
        for item in days[:limit]:
            ts = item.get("date")
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts)
                except ValueError:
                    timestamp = datetime.strptime(ts, "%Y-%m-%d")
            else:
                timestamp = datetime.now()

            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=timestamp,
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                interval=interval,
            ))

        return candles

    def _fetch_timesales(
        self,
        symbol: str,
        interval: Interval,
        start: date | None,
        end: date | None,
        limit: int,
    ) -> ResultList[OHLCV]:
        """Fetch intraday data from timesales endpoint."""
        interval_map = {
            Interval.MINUTE_1: "1min",
            Interval.MINUTE_5: "5min",
            Interval.MINUTE_15: "15min",
        }
        tradier_interval = interval_map.get(interval, "1min")

        params = {"symbol": symbol, "interval": tradier_interval}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()

        data = self.get("/markets/timesales", params=params)

        series = data.get("series")
        if not series:
            return ResultList(provider=self.provider_name)

        bars = series.get("data", [])
        if isinstance(bars, dict):
            bars = [bars]

        candles = ResultList(provider=self.provider_name)
        for item in bars[:limit]:
            ts = item.get("time") or item.get("timestamp")
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            else:
                timestamp = datetime.now()

            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=timestamp,
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close") or item.get("price", 0)),
                volume=int(item.get("volume", 0)),
                vwap=float(item.get("vwap")) if item.get("vwap") else None,
                interval=interval,
            ))

        return candles

    def get_expirations(self, symbol: str) -> list[date]:
        """Get available option expiration dates."""
        data = self.get("/markets/options/expirations", params={"symbol": symbol})

        expirations = data.get("expirations", {}).get("date", [])
        if isinstance(expirations, str):
            expirations = [expirations]

        return [date.fromisoformat(exp) for exp in expirations]

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChain:
        """Get options chain for a symbol."""
        # Get expiration if not provided
        if not expiration:
            expirations = self.get_expirations(symbol)
            if not expirations:
                from ...common.base import ProviderError
                raise ProviderError(self.provider_name, f"No options available for {symbol}")
            expiration = expirations[0]

        # Get underlying price
        quote_data = self.get("/markets/quotes", params={"symbols": symbol})
        quotes = quote_data.get("quotes", {}).get("quote", {})
        if isinstance(quotes, list):
            quotes = quotes[0] if quotes else {}
        underlying_price = quotes.get("last")

        # Get option chain
        params = {
            "symbol": symbol,
            "expiration": expiration.isoformat(),
            "greeks": "true",
        }
        data = self.get("/markets/options/chains", params=params)

        options = data.get("options", {}).get("option", [])
        if isinstance(options, dict):
            options = [options]

        calls = []
        puts = []

        for opt in options:
            contract = self._parse_contract(symbol, opt, expiration)
            if opt.get("option_type") == "call":
                calls.append(contract)
            else:
                puts.append(contract)

        return OptionChain(
            underlying=symbol.upper(),
            expiration=expiration,
            calls=calls,
            puts=puts,
            underlying_price=underlying_price,
        )

    def _parse_contract(
        self,
        underlying: str,
        data: dict,
        expiration: date,
    ) -> OptionContract:
        """Parse Tradier option data to OptionContract."""
        greeks = data.get("greeks", {}) or {}

        return OptionContract(
            symbol=data.get("symbol", ""),
            underlying=underlying.upper(),
            contract_type=data.get("option_type", "call"),
            strike=float(data.get("strike", 0)),
            expiration=expiration,
            bid=data.get("bid"),
            ask=data.get("ask"),
            last=data.get("last"),
            volume=data.get("volume"),
            open_interest=data.get("open_interest"),
            implied_volatility=greeks.get("mid_iv"),
            delta=greeks.get("delta"),
            gamma=greeks.get("gamma"),
            theta=greeks.get("theta"),
            vega=greeks.get("vega"),
        )
