"""
Yahoo Finance provider using yfinance library.

Rate limits: ~60 requests/minute (IP-based)
Features: Quotes, OHLCV, Company profiles, Options, Dividends, Splits, Recommendations, News
No API key required.
"""

from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import (
    Quote, OHLCV, CompanyProfile, OptionChain, OptionContract,
    Dividend, StockSplit, AnalystRecommendation, NewsArticle,
)
from .base import BaseProvider


def _interval_to_yfinance(interval: Interval) -> str:
    """Convert interval to yfinance format."""
    mapping = {
        Interval.MINUTE_1: "1m",
        Interval.MINUTE_5: "5m",
        Interval.MINUTE_15: "15m",
        Interval.MINUTE_30: "30m",
        Interval.HOUR_1: "1h",
        Interval.DAY_1: "1d",
        Interval.WEEK_1: "1wk",
        Interval.MONTH_1: "1mo",
    }
    return mapping.get(interval, "1d")


class YFinanceProvider(BaseProvider):
    """
    Yahoo Finance provider using the yfinance library.

    Advantages:
    - Free, no API key needed
    - Good historical coverage
    - Includes options data

    Limitations:
    - Rate limited by IP
    - Intraday data limited to recent periods
    - Can be unreliable during high traffic
    """

    provider_name = "yfinance"
    base_url = ""  # Uses yfinance library, not direct HTTP

    def __init__(self):
        super().__init__()
        self._yf = None

    def is_configured(self) -> bool:
        """Check if yfinance is installed."""
        try:
            import yfinance
            self._yf = yfinance
            return True
        except ImportError:
            return False

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
    def supports_options(self) -> bool:
        return True

    def _get_ticker(self, symbol: str):
        """Get a yfinance Ticker object."""
        if self._yf is None:
            import yfinance
            self._yf = yfinance
        return self._yf.Ticker(symbol)

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        ticker = self._get_ticker(symbol)
        info = ticker.info

        return Quote(
            symbol=symbol.upper(),
            price=info.get("currentPrice") or info.get("regularMarketPrice", 0),
            bid=info.get("bid"),
            ask=info.get("ask"),
            bid_size=info.get("bidSize"),
            ask_size=info.get("askSize"),
            volume=info.get("volume") or info.get("regularMarketVolume"),
            open=info.get("open") or info.get("regularMarketOpen"),
            high=info.get("dayHigh") or info.get("regularMarketDayHigh"),
            low=info.get("dayLow") or info.get("regularMarketDayLow"),
            previous_close=info.get("previousClose") or info.get("regularMarketPreviousClose"),
            change=info.get("regularMarketChange"),
            change_percent=info.get("regularMarketChangePercent"),
            market_cap=info.get("marketCap"),
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
        yf_interval = _interval_to_yfinance(interval)

        # Set date range
        end_date = end or date.today()
        if start:
            start_date = start
        else:
            # Default: 1 year for daily, 7 days for intraday
            if interval in {Interval.MINUTE_1, Interval.MINUTE_5, Interval.MINUTE_15,
                           Interval.MINUTE_30, Interval.HOUR_1}:
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=365)

        ticker = self._get_ticker(symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval=yf_interval,
        )

        if df.empty:
            return ResultList(provider=self.provider_name)

        candles = ResultList(provider=self.provider_name)
        for idx, row in df.iterrows():
            candles.append(OHLCV(
                symbol=symbol.upper(),
                timestamp=idx.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
                interval=interval,
            ))

        return ResultList(candles[:limit], provider=self.provider_name)

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company profile information."""
        ticker = self._get_ticker(symbol)
        info = ticker.info

        return CompanyProfile(
            symbol=symbol.upper(),
            name=info.get("longName") or info.get("shortName", ""),
            description=info.get("longBusinessSummary"),
            exchange=info.get("exchange"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            currency=info.get("currency"),
            market_cap=info.get("marketCap"),
            shares_outstanding=info.get("sharesOutstanding"),
            employees=info.get("fullTimeEmployees"),
            website=info.get("website"),
            pe_ratio=info.get("trailingPE"),
            pb_ratio=info.get("priceToBook"),
            dividend_yield=info.get("dividendYield"),
            beta=info.get("beta"),
            revenue=info.get("totalRevenue"),
            net_income=info.get("netIncomeToCommon"),
            eps=info.get("trailingEps"),
        )

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> OptionChain:
        """Get options chain for a symbol."""
        ticker = self._get_ticker(symbol)

        # Get available expirations
        expirations = ticker.options
        if not expirations:
            raise ProviderError(self.provider_name, f"No options available for {symbol}")

        # Use specified expiration or first available
        if expiration:
            exp_str = expiration.isoformat()
            if exp_str not in expirations:
                # Find closest expiration
                exp_str = min(expirations, key=lambda x: abs(
                    date.fromisoformat(x) - expiration
                ))
        else:
            exp_str = expirations[0]

        # Fetch option chain
        opt = ticker.option_chain(exp_str)

        calls = []
        for _, row in opt.calls.iterrows():
            calls.append(self._parse_option_contract(symbol, row, "call", exp_str))

        puts = []
        for _, row in opt.puts.iterrows():
            puts.append(self._parse_option_contract(symbol, row, "put", exp_str))

        # Get underlying price
        info = ticker.info
        underlying_price = info.get("currentPrice") or info.get("regularMarketPrice")

        return OptionChain(
            underlying=symbol.upper(),
            expiration=date.fromisoformat(exp_str),
            calls=calls,
            puts=puts,
            underlying_price=underlying_price,
        )

    def _parse_option_contract(
        self,
        underlying: str,
        row,
        contract_type: str,
        expiration: str,
    ) -> OptionContract:
        """Parse a yfinance option row to OptionContract."""
        return OptionContract(
            symbol=row.get("contractSymbol", ""),
            underlying=underlying.upper(),
            contract_type=contract_type,
            strike=float(row.get("strike", 0)),
            expiration=date.fromisoformat(expiration),
            bid=row.get("bid"),
            ask=row.get("ask"),
            last=row.get("lastPrice"),
            volume=int(row.get("volume", 0)) if row.get("volume") else None,
            open_interest=int(row.get("openInterest", 0)) if row.get("openInterest") else None,
            implied_volatility=row.get("impliedVolatility"),
        )

    def get_expirations(self, symbol: str) -> list[date]:
        """Get available option expiration dates."""
        ticker = self._get_ticker(symbol)
        return [date.fromisoformat(exp) for exp in ticker.options]

    @property
    def supports_dividends(self) -> bool:
        return True

    @property
    def supports_recommendations(self) -> bool:
        return True

    @property
    def supports_news(self) -> bool:
        return True

    def get_dividends(self, symbol: str) -> ResultList[Dividend]:
        """
        Get dividend history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ResultList of Dividend
        """
        ticker = self._get_ticker(symbol)
        divs = ticker.dividends

        dividends = ResultList(provider=self.provider_name)
        if divs.empty:
            return dividends

        for idx, amount in divs.items():
            dividends.append(Dividend(
                symbol=symbol.upper(),
                ex_date=idx.date(),
                amount=float(amount),
            ))

        return dividends

    def get_stock_splits(self, symbol: str) -> ResultList[StockSplit]:
        """
        Get stock split history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ResultList of StockSplit
        """
        ticker = self._get_ticker(symbol)
        splits_data = ticker.splits

        splits = ResultList(provider=self.provider_name)
        if splits_data.empty:
            return splits

        for idx, ratio in splits_data.items():
            splits.append(StockSplit(
                symbol=symbol.upper(),
                split_date=idx.date(),
                ratio=float(ratio),
            ))

        return splits

    def get_recommendations(self, symbol: str) -> ResultList[AnalystRecommendation]:
        """
        Get analyst recommendations.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ResultList of AnalystRecommendation
        """
        ticker = self._get_ticker(symbol)

        try:
            recs = ticker.recommendations
            if recs is None or recs.empty:
                return ResultList(provider=self.provider_name)
        except Exception:
            return ResultList(provider=self.provider_name)

        recommendations = ResultList(provider=self.provider_name)

        # Group by period and aggregate
        # yfinance returns individual analyst grades, we need to aggregate
        for idx, row in recs.iterrows():
            grade = str(row.get("To Grade", "")).lower()

            rec = AnalystRecommendation(
                symbol=symbol.upper(),
                period=idx.date() if hasattr(idx, 'date') else None,
                strong_buy=1 if grade in ["strong buy", "strongbuy"] else 0,
                buy=1 if grade in ["buy", "outperform", "overweight"] else 0,
                hold=1 if grade in ["hold", "neutral", "equal-weight", "market perform"] else 0,
                sell=1 if grade in ["sell", "underperform", "underweight"] else 0,
                strong_sell=1 if grade in ["strong sell", "strongsell"] else 0,
            )
            recommendations.append(rec)

        return recommendations

    def get_news(
        self,
        symbol: str,
        limit: int = 20,
    ) -> ResultList[NewsArticle]:
        """
        Get recent news for a symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum articles to return

        Returns:
            ResultList of NewsArticle
        """
        ticker = self._get_ticker(symbol)

        try:
            news_data = ticker.news
            if not news_data:
                return ResultList(provider=self.provider_name)
        except Exception:
            return ResultList(provider=self.provider_name)

        news = ResultList(provider=self.provider_name)
        for item in news_data[:limit]:
            published = None
            if item.get("providerPublishTime"):
                try:
                    published = datetime.fromtimestamp(item["providerPublishTime"])
                except (ValueError, TypeError):
                    pass

            news.append(NewsArticle(
                headline=item.get("title", ""),
                source=item.get("publisher"),
                url=item.get("link"),
                image_url=item.get("thumbnail", {}).get("resolutions", [{}])[-1].get("url") if item.get("thumbnail") else None,
                published_at=published,
                symbols=[symbol.upper()],
            ))

        return news
