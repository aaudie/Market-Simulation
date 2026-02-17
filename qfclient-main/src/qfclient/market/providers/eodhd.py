"""
EOD Historical Data provider.

Rate limits: 20 requests/day (free tier)
Features: Global EOD data, Fundamentals, Dividends, Splits
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV, CompanyProfile
from .base import BaseProvider


class EODHDProvider(BaseProvider):
    """
    EOD Historical Data provider.

    Provides:
    - Global EOD stock prices (70+ exchanges)
    - Fundamental data
    - Dividends and splits
    - ETF data

    Note: Free tier is very limited (20 req/day).
    Best for international markets not covered by other providers.
    """

    provider_name = "eodhd"
    base_url = "https://eodhd.com/api"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("EODHD_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add API key and format to params."""
        params = params or {}
        params["api_token"] = self.api_key
        params["fmt"] = "json"
        return super().get(url, params=params, **kwargs)

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    @property
    def supports_company_profile(self) -> bool:
        # Note: Fundamentals require paid EODHD subscription
        return False

    def _format_symbol(self, symbol: str, exchange: str = "US") -> str:
        """Format symbol with exchange suffix."""
        if "." in symbol:
            return symbol
        return f"{symbol}.{exchange}"

    def get_quote(self, symbol: str, exchange: str = "US") -> Quote:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Stock ticker symbol
            exchange: Exchange code (US, LSE, TSE, etc.)

        Returns:
            Quote with latest price data
        """
        formatted = self._format_symbol(symbol, exchange)
        data = self.get(f"/real-time/{formatted}")

        if not data or data.get("code") == "NOT_FOUND":
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return Quote(
            symbol=symbol.upper(),
            price=data.get("close", 0),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            volume=data.get("volume"),
            previous_close=data.get("previousClose"),
            change=data.get("change"),
            change_percent=data.get("change_p"),
            timestamp=datetime.fromtimestamp(data.get("timestamp", 0)) if data.get("timestamp") else None,
        )

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
        exchange: str = "US",
    ) -> ResultList[OHLCV]:
        """
        Get OHLCV bars for a symbol.

        Note: EODHD only supports daily data on free tier.

        Args:
            symbol: Stock ticker symbol
            interval: Candle interval (only DAY_1 on free tier)
            start: Start date
            end: End date
            limit: Maximum number of bars
            exchange: Exchange code (US, LSE, TSE, etc.)

        Returns:
            ResultList of OHLCV candles
        """
        if interval not in (Interval.DAY_1, Interval.WEEK_1, Interval.MONTH_1):
            raise ProviderError(
                self.provider_name,
                f"Interval {interval.value} not supported. Use daily/weekly/monthly."
            )

        formatted = self._format_symbol(symbol, exchange)

        params = {}
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()

        # Map interval to period
        if interval == Interval.WEEK_1:
            params["period"] = "w"
        elif interval == Interval.MONTH_1:
            params["period"] = "m"
        else:
            params["period"] = "d"

        data = self.get(f"/eod/{formatted}", params=params)

        if not data or isinstance(data, dict) and data.get("code") == "NOT_FOUND":
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        candles = ResultList(provider=self.provider_name)

        for bar in data[-limit:] if isinstance(data, list) else []:
            try:
                timestamp = datetime.strptime(bar.get("date", ""), "%Y-%m-%d")

                candles.append(OHLCV(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open=bar.get("open", 0),
                    high=bar.get("high", 0),
                    low=bar.get("low", 0),
                    close=bar.get("adjusted_close") or bar.get("close", 0),
                    volume=int(bar.get("volume", 0)),
                    interval=interval,
                ))
            except (ValueError, TypeError, KeyError):
                continue

        return candles

    def get_company_profile(self, symbol: str, exchange: str = "US") -> CompanyProfile:
        """Get company fundamentals."""
        formatted = self._format_symbol(symbol, exchange)
        data = self.get(f"/fundamentals/{formatted}")

        if not data or data.get("code") == "NOT_FOUND":
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        general = data.get("General", {})
        highlights = data.get("Highlights", {})
        valuation = data.get("Valuation", {})
        technicals = data.get("Technicals", {})

        return CompanyProfile(
            symbol=symbol.upper(),
            name=general.get("Name", ""),
            exchange=general.get("Exchange"),
            sector=general.get("Sector"),
            industry=general.get("Industry"),
            country=general.get("CountryName"),
            currency=general.get("CurrencyCode"),
            market_cap=highlights.get("MarketCapitalization"),
            shares_outstanding=data.get("SharesStats", {}).get("SharesOutstanding"),
            pe_ratio=highlights.get("PERatio"),
            forward_pe=valuation.get("ForwardPE"),
            eps=highlights.get("EarningsShare"),
            dividend_yield=highlights.get("DividendYield"),
            beta=technicals.get("Beta"),
            high_52_week=technicals.get("52WeekHigh"),
            low_52_week=technicals.get("52WeekLow"),
            description=general.get("Description"),
            website=general.get("WebURL"),
            ipo_date=datetime.strptime(
                general.get("IPODate", ""), "%Y-%m-%d"
            ).date() if general.get("IPODate") else None,
        )

    def get_exchanges(self) -> ResultList[dict]:
        """Get list of supported exchanges."""
        data = self.get("/exchanges-list")

        exchanges = ResultList(provider=self.provider_name)
        for item in data if isinstance(data, list) else []:
            exchanges.append({
                "code": item.get("Code"),
                "name": item.get("Name"),
                "country": item.get("Country"),
                "currency": item.get("Currency"),
            })

        return exchanges
