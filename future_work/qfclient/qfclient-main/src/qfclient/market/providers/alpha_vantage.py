"""
Alpha Vantage provider.

Rate limits: 5 requests/minute, 25 requests/day (free tier)
Features: Quotes, OHLCV (daily + intraday), Company profiles, Earnings
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import Quote, OHLCV, CompanyProfile, EarningsEvent
from .base import BaseProvider


def _interval_to_alpha_vantage(interval: Interval) -> tuple[str, str]:
    """
    Convert interval to Alpha Vantage function and interval param.

    Returns:
        Tuple of (function_name, interval_param or None)
    """
    intraday_mapping = {
        Interval.MINUTE_1: ("TIME_SERIES_INTRADAY", "1min"),
        Interval.MINUTE_5: ("TIME_SERIES_INTRADAY", "5min"),
        Interval.MINUTE_15: ("TIME_SERIES_INTRADAY", "15min"),
        Interval.MINUTE_30: ("TIME_SERIES_INTRADAY", "30min"),
        Interval.HOUR_1: ("TIME_SERIES_INTRADAY", "60min"),
    }

    if interval in intraday_mapping:
        return intraday_mapping[interval]

    # Daily and above use different functions
    # Note: ADJUSTED endpoints are premium-only, use non-adjusted for free tier
    if interval == Interval.WEEK_1:
        return ("TIME_SERIES_WEEKLY", None)
    elif interval == Interval.MONTH_1:
        return ("TIME_SERIES_MONTHLY", None)
    else:
        return ("TIME_SERIES_DAILY", None)


class AlphaVantageProvider(BaseProvider):
    """
    Alpha Vantage data provider.

    Provides:
    - Real-time and delayed quotes
    - Historical OHLCV (daily, weekly, monthly, intraday)
    - Company fundamentals (profile, financials)
    - Earnings calendar and history

    Note: Very restrictive free tier (5/min, 25/day).
    Supports multi-account scaling with multiple API keys.
    """

    provider_name = "alpha_vantage"
    base_url = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _make_request(self, function: str, **params) -> dict:
        """Make request to Alpha Vantage API."""
        params["function"] = function
        params["apikey"] = self.api_key

        data = self.get("", params=params)

        # Check for API errors
        if "Error Message" in data:
            raise ProviderError(self.provider_name, data["Error Message"])
        if "Note" in data:
            # Rate limit warning
            raise ProviderError(self.provider_name, f"Rate limited: {data['Note']}")
        if "Information" in data:
            raise ProviderError(self.provider_name, data["Information"])

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

    @property
    def supports_earnings(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        data = self._make_request("GLOBAL_QUOTE", symbol=symbol)

        quote_data = data.get("Global Quote", {})
        if not quote_data:
            raise ProviderError(self.provider_name, f"No quote data for {symbol}")

        price = float(quote_data.get("05. price", 0))
        prev_close = float(quote_data.get("08. previous close", 0)) if quote_data.get("08. previous close") else None
        change = float(quote_data.get("09. change", 0)) if quote_data.get("09. change") else None
        change_pct_str = quote_data.get("10. change percent", "0%").rstrip("%")
        change_pct = float(change_pct_str) if change_pct_str else None

        return Quote(
            symbol=symbol.upper(),
            price=price,
            open=float(quote_data.get("02. open", 0)) if quote_data.get("02. open") else None,
            high=float(quote_data.get("03. high", 0)) if quote_data.get("03. high") else None,
            low=float(quote_data.get("04. low", 0)) if quote_data.get("04. low") else None,
            volume=int(quote_data.get("06. volume", 0)) if quote_data.get("06. volume") else None,
            previous_close=prev_close,
            change=change,
            change_percent=change_pct,
            timestamp=datetime.strptime(
                quote_data.get("07. latest trading day", ""),
                "%Y-%m-%d"
            ) if quote_data.get("07. latest trading day") else None,
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
            start: Start date (filters results)
            end: End date (filters results)
            limit: Maximum number of bars to return

        Returns:
            ResultList of OHLCV candles
        """
        function, av_interval = _interval_to_alpha_vantage(interval)

        # Note: outputsize=full is premium-only, compact returns last 100 data points
        params = {"symbol": symbol, "outputsize": "compact"}
        if av_interval:
            params["interval"] = av_interval

        data = self._make_request(function, **params)

        # Find the time series key (varies by function)
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "Weekly" in key or "Monthly" in key:
                time_series_key = key
                break

        if not time_series_key:
            raise ProviderError(self.provider_name, f"No time series data for {symbol}")

        time_series = data[time_series_key]
        candles = ResultList(provider=self.provider_name)

        # Parse and filter candles
        sorted_dates = sorted(time_series.keys(), reverse=True)

        for date_str in sorted_dates[:limit * 2]:  # Fetch extra for date filtering
            try:
                # Parse timestamp based on interval
                if av_interval:  # Intraday
                    ts = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    ts = datetime.strptime(date_str, "%Y-%m-%d")

                # Apply date filters
                if start and ts.date() < start:
                    continue
                if end and ts.date() > end:
                    continue

                bar = time_series[date_str]

                # Handle different key formats (some have numbered prefixes)
                def get_val(keys: list[str]) -> float | None:
                    for k in keys:
                        if k in bar:
                            return float(bar[k])
                    return None

                candles.append(OHLCV(
                    symbol=symbol.upper(),
                    timestamp=ts,
                    open=get_val(["1. open", "open"]) or 0,
                    high=get_val(["2. high", "high"]) or 0,
                    low=get_val(["3. low", "low"]) or 0,
                    close=get_val(["4. close", "close", "5. adjusted close"]) or 0,
                    volume=int(get_val(["5. volume", "6. volume", "volume"]) or 0),
                    interval=interval,
                ))

                if len(candles) >= limit:
                    break

            except (ValueError, KeyError) as e:
                continue

        # Return in chronological order
        candles.reverse()
        return candles

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company fundamentals and profile."""
        data = self._make_request("OVERVIEW", symbol=symbol)

        if not data or data.get("Symbol") is None:
            raise ProviderError(self.provider_name, f"No profile data for {symbol}")

        def safe_float(val: str | None) -> float | None:
            if val is None or val == "None" or val == "-":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_int(val: str | None) -> int | None:
            if val is None or val == "None" or val == "-":
                return None
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("Name", ""),
            exchange=data.get("Exchange"),
            sector=data.get("Sector"),
            industry=data.get("Industry"),
            country=data.get("Country"),
            currency=data.get("Currency"),
            market_cap=safe_float(data.get("MarketCapitalization")),
            shares_outstanding=safe_float(data.get("SharesOutstanding")),
            pe_ratio=safe_float(data.get("PERatio")),
            forward_pe=safe_float(data.get("ForwardPE")),
            eps=safe_float(data.get("EPS")),
            dividend_yield=safe_float(data.get("DividendYield")),
            beta=safe_float(data.get("Beta")),
            high_52_week=safe_float(data.get("52WeekHigh")),
            low_52_week=safe_float(data.get("52WeekLow")),
            avg_volume=safe_int(data.get("AverageVolume")),
            description=data.get("Description"),
            website=None,  # Not provided by Alpha Vantage
        )

    def get_earnings(
        self,
        symbol: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[EarningsEvent]:
        """Get earnings history and estimates for a symbol."""
        if not symbol:
            raise ProviderError(
                self.provider_name,
                "Alpha Vantage requires a symbol for earnings data"
            )

        data = self._make_request("EARNINGS", symbol=symbol)

        earnings = ResultList(provider=self.provider_name)

        # Process quarterly earnings
        for item in data.get("quarterlyEarnings", []):
            try:
                fiscal_date = item.get("fiscalDateEnding")
                if not fiscal_date:
                    continue

                report_date = datetime.strptime(fiscal_date, "%Y-%m-%d").date()

                # Apply date filters
                if start and report_date < start:
                    continue
                if end and report_date > end:
                    continue

                def safe_float(val: str | None) -> float | None:
                    if val is None or val == "None":
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None

                earnings.append(EarningsEvent(
                    symbol=symbol.upper(),
                    report_date=report_date,
                    fiscal_quarter=None,  # Not directly provided
                    fiscal_year=report_date.year,
                    eps_estimate=safe_float(item.get("estimatedEPS")),
                    eps_actual=safe_float(item.get("reportedEPS")),
                    surprise=safe_float(item.get("surprise")),
                    surprise_percent=safe_float(item.get("surprisePercentage")),
                ))
            except (ValueError, KeyError):
                continue

        return earnings
