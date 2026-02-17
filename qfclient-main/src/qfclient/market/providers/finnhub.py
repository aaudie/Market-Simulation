"""
Finnhub provider.

Rate limits: 60 requests/minute (free tier)
Features: Quotes, OHLCV, Company profiles, Earnings, News, Insider transactions, Recommendations
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList
from ...common.types import Interval
from ..models import (
    Quote, OHLCV, CompanyProfile, EarningsEvent,
    NewsArticle, InsiderTransaction, AnalystRecommendation, PriceTarget,
)
from .base import BaseProvider


def _interval_to_finnhub(interval: Interval) -> str:
    """Convert interval to Finnhub resolution."""
    mapping = {
        Interval.MINUTE_1: "1",
        Interval.MINUTE_5: "5",
        Interval.MINUTE_15: "15",
        Interval.MINUTE_30: "30",
        Interval.HOUR_1: "60",
        Interval.DAY_1: "D",
        Interval.WEEK_1: "W",
        Interval.MONTH_1: "M",
    }
    return mapping.get(interval, "D")


class FinnhubProvider(BaseProvider):
    """
    Finnhub data provider.

    Provides comprehensive market data:
    - Real-time quotes
    - Historical candles (25+ years)
    - Company profiles
    - Earnings calendar
    - News and sentiment
    """

    provider_name = "finnhub"
    base_url = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        return {}  # Finnhub uses query param for auth

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add API key to params."""
        params = params or {}
        params["token"] = self.api_key
        return super().get(url, params=params, **kwargs)

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        # Note: OHLCV (candle) data requires paid Finnhub subscription
        return False

    @property
    def supports_company_profile(self) -> bool:
        return True

    @property
    def supports_earnings(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        data = self.get("/quote", params={"symbol": symbol})

        return Quote(
            symbol=symbol.upper(),
            price=data.get("c", 0),  # Current price
            open=data.get("o"),
            high=data.get("h"),
            low=data.get("l"),
            previous_close=data.get("pc"),
            change=data.get("d"),
            change_percent=data.get("dp"),
            timestamp=datetime.fromtimestamp(data.get("t", 0)) if data.get("t") else None,
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
        end_date = end or date.today()
        start_date = start or (end_date - timedelta(days=365))

        params = {
            "symbol": symbol,
            "resolution": _interval_to_finnhub(interval),
            "from": int(datetime.combine(start_date, datetime.min.time()).timestamp()),
            "to": int(datetime.combine(end_date, datetime.max.time()).timestamp()),
        }

        data = self.get("/stock/candle", params=params)

        if data.get("s") != "ok":
            from ...common.base import ProviderError
            raise ProviderError(self.provider_name, f"No data: {data.get('s')}")

        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        candles = ResultList(provider=self.provider_name)
        for i in range(min(len(timestamps), limit)):
            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=datetime.fromtimestamp(timestamps[i]),
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=int(volumes[i]),
                interval=interval,
            ))

        return candles

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company profile information."""
        data = self.get("/stock/profile2", params={"symbol": symbol})

        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            exchange=data.get("exchange"),
            industry=data.get("finnhubIndustry"),
            country=data.get("country"),
            currency=data.get("currency"),
            market_cap=data.get("marketCapitalization"),
            shares_outstanding=data.get("shareOutstanding"),
            website=data.get("weburl"),
            logo_url=data.get("logo"),
            ipo_date=date.fromisoformat(data["ipo"]) if data.get("ipo") else None,
        )

    def get_earnings(
        self,
        symbol: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[EarningsEvent]:
        """Get earnings calendar."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()

        data = self.get("/calendar/earnings", params=params)

        earnings = ResultList(provider=self.provider_name)
        for item in data.get("earningsCalendar", []):
            quarter = item.get("quarter")
            earnings.append(EarningsEvent(
                symbol=item.get("symbol", ""),
                report_date=date.fromisoformat(item["date"]) if item.get("date") else None,
                fiscal_quarter=f"Q{quarter}" if quarter else None,
                fiscal_year=item.get("year"),
                eps_estimate=item.get("epsEstimate"),
                eps_actual=item.get("epsActual"),
                revenue_estimate=item.get("revenueEstimate"),
                revenue_actual=item.get("revenueActual"),
                report_time=item.get("hour"),
            ))

        return earnings

    @property
    def supports_news(self) -> bool:
        return True

    @property
    def supports_insider_transactions(self) -> bool:
        return True

    @property
    def supports_recommendations(self) -> bool:
        return True

    def get_news(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
    ) -> ResultList[NewsArticle]:
        """
        Get company news articles.

        Args:
            symbol: Stock ticker symbol
            start: Start date for news (default: 7 days ago)
            end: End date for news (default: today)
            limit: Maximum articles to return

        Returns:
            ResultList of NewsArticle
        """
        end_date = end or date.today()
        start_date = start or (end_date - timedelta(days=7))

        data = self.get("/company-news", params={
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        })

        news = ResultList(provider=self.provider_name)
        for item in data[:limit] if isinstance(data, list) else []:
            news.append(NewsArticle(
                headline=item.get("headline", ""),
                summary=item.get("summary"),
                source=item.get("source"),
                url=item.get("url"),
                image_url=item.get("image"),
                published_at=datetime.fromtimestamp(item["datetime"]) if item.get("datetime") else None,
                symbols=[symbol.upper()],
                category=item.get("category"),
            ))

        return news

    def get_insider_transactions(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[InsiderTransaction]:
        """
        Get insider transactions (Form 4 filings).

        Args:
            symbol: Stock ticker symbol
            start: Start date (optional)
            end: End date (optional)

        Returns:
            ResultList of InsiderTransaction
        """
        params = {"symbol": symbol}

        data = self.get("/stock/insider-transactions", params=params)

        transactions = ResultList(provider=self.provider_name)
        for item in data.get("data", []):
            tx_date = None
            if item.get("transactionDate"):
                try:
                    tx_date = date.fromisoformat(item["transactionDate"])
                except (ValueError, TypeError):
                    pass

            # Apply date filters
            if tx_date:
                if start and tx_date < start:
                    continue
                if end and tx_date > end:
                    continue

            filing_date = None
            if item.get("filingDate"):
                try:
                    filing_date = date.fromisoformat(item["filingDate"])
                except (ValueError, TypeError):
                    pass

            transactions.append(InsiderTransaction(
                symbol=symbol.upper(),
                filing_date=filing_date,
                transaction_date=tx_date,
                insider_name=item.get("name", "Unknown"),
                insider_title=None,  # Not provided by Finnhub
                transaction_type=item.get("transactionCode"),
                shares=item.get("share"),
                price=item.get("transactionPrice"),
                value=item.get("value"),
                shares_owned_after=None,
            ))

        return transactions

    def get_recommendations(self, symbol: str) -> ResultList[AnalystRecommendation]:
        """
        Get analyst recommendation trends.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ResultList of AnalystRecommendation (by period)
        """
        data = self.get("/stock/recommendation", params={"symbol": symbol})

        recommendations = ResultList(provider=self.provider_name)
        for item in data if isinstance(data, list) else []:
            period = None
            if item.get("period"):
                try:
                    period = date.fromisoformat(item["period"])
                except (ValueError, TypeError):
                    pass

            recommendations.append(AnalystRecommendation(
                symbol=symbol.upper(),
                period=period,
                strong_buy=item.get("strongBuy", 0),
                buy=item.get("buy", 0),
                hold=item.get("hold", 0),
                sell=item.get("sell", 0),
                strong_sell=item.get("strongSell", 0),
            ))

        return recommendations

    def get_price_target(self, symbol: str) -> PriceTarget:
        """
        Get analyst price targets.

        Args:
            symbol: Stock ticker symbol

        Returns:
            PriceTarget with high/low/mean/median targets
        """
        data = self.get("/stock/price-target", params={"symbol": symbol})

        last_updated = None
        if data.get("lastUpdated"):
            try:
                last_updated = date.fromisoformat(data["lastUpdated"])
            except (ValueError, TypeError):
                pass

        return PriceTarget(
            symbol=symbol.upper(),
            target_high=data.get("targetHigh"),
            target_low=data.get("targetLow"),
            target_mean=data.get("targetMean"),
            target_median=data.get("targetMedian"),
            num_analysts=data.get("numberOfAnalysts"),
            last_updated=last_updated,
        )
