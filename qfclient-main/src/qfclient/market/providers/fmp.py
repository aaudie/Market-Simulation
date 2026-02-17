"""
Financial Modeling Prep (FMP) provider.

Rate limits: 250 requests/day (free tier)
Features: OHLCV, Company profiles, Earnings, Financial statements, Dividends, Insider transactions
"""

import os
from datetime import date, datetime

from ...common.base import ResultList
from ...common.types import Interval
from ..models import (
    Quote, OHLCV, CompanyProfile, EarningsEvent,
    FinancialStatement, Dividend, StockSplit, InsiderTransaction,
)
from .base import BaseProvider


class FMPProvider(BaseProvider):
    """
    Financial Modeling Prep data provider.

    Provides comprehensive fundamental data:
    - Historical price data (30+ years)
    - Company profiles
    - Financial statements
    - Earnings calendar
    """

    provider_name = "fmp"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FMP_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict | list:
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

    @property
    def supports_company_profile(self) -> bool:
        return True

    @property
    def supports_earnings(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Get the latest quote for a symbol."""
        # FMP stable API uses query params
        data = self.get("/quote", params={"symbol": symbol})

        if isinstance(data, list) and data:
            data = data[0]
        elif not data:
            from ...common.base import ProviderError
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return Quote(
            symbol=symbol.upper(),
            price=data.get("price", 0),
            open=data.get("open"),
            high=data.get("dayHigh"),
            low=data.get("dayLow"),
            volume=data.get("volume"),
            previous_close=data.get("previousClose"),
            change=data.get("change"),
            change_percent=data.get("changePercentage"),
            market_cap=data.get("marketCap"),
            timestamp=datetime.fromtimestamp(data["timestamp"]) if data.get("timestamp") else None,
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
        # FMP uses different endpoints for different intervals
        if interval in {Interval.MINUTE_1, Interval.MINUTE_5, Interval.MINUTE_15,
                        Interval.MINUTE_30, Interval.HOUR_1, Interval.HOUR_4}:
            interval_str = {
                Interval.MINUTE_1: "1min",
                Interval.MINUTE_5: "5min",
                Interval.MINUTE_15: "15min",
                Interval.MINUTE_30: "30min",
                Interval.HOUR_1: "1hour",
                Interval.HOUR_4: "4hour",
            }[interval]
            url = f"/historical-chart/{interval_str}"
        else:
            url = "/historical-price-eod/full"

        params = {"symbol": symbol}
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()

        data = self.get(url, params=params)

        # Handle different response formats
        if isinstance(data, list):
            items = data[:limit]
        else:
            items = data.get("historical", data.get("results", []))[:limit]

        candles = ResultList(provider=self.provider_name)
        for item in items:
            ts = item.get("date") or item.get("datetime")
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
                open=item.get("open", 0),
                high=item.get("high", 0),
                low=item.get("low", 0),
                close=item.get("close", 0),
                volume=int(item.get("volume", 0)),
                vwap=item.get("vwap"),
                interval=interval,
            ))

        return candles

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company profile information."""
        # FMP stable API uses query params
        data = self.get("/profile", params={"symbol": symbol})

        if isinstance(data, list) and data:
            data = data[0]
        elif not data:
            from ...common.base import ProviderError
            raise ProviderError(self.provider_name, f"No profile data for {symbol}")

        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("companyName", ""),
            description=data.get("description"),
            exchange=data.get("exchange") or data.get("exchangeShortName"),
            sector=data.get("sector"),
            industry=data.get("industry"),
            country=data.get("country"),
            currency=data.get("currency"),
            market_cap=data.get("marketCap") or data.get("mktCap"),
            employees=data.get("fullTimeEmployees"),
            website=data.get("website"),
            logo_url=data.get("image"),
            ceo=data.get("ceo"),
            ipo_date=date.fromisoformat(data["ipoDate"]) if data.get("ipoDate") else None,
            pe_ratio=data.get("pe"),
            beta=data.get("beta"),
            dividend_yield=data.get("lastDividend") or data.get("lastDiv"),
        )

    def get_earnings(
        self,
        symbol: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ResultList[EarningsEvent]:
        """Get earnings calendar."""
        params = {}
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()

        if symbol:
            data = self.get(f"/historical/earning_calendar/{symbol}", params=params)
        else:
            data = self.get("/earning_calendar", params=params)

        earnings = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            earnings.append(EarningsEvent(
                symbol=item.get("symbol", ""),
                report_date=date.fromisoformat(item["date"]) if item.get("date") else None,
                fiscal_quarter=item.get("fiscalDateEnding"),
                eps_estimate=item.get("epsEstimated"),
                eps_actual=item.get("eps"),
                revenue_estimate=item.get("revenueEstimated"),
                revenue_actual=item.get("revenue"),
                report_time=item.get("time"),
                updated_from_date=date.fromisoformat(item["updatedFromDate"]) if item.get("updatedFromDate") else None,
            ))

        return earnings

    @property
    def supports_financials(self) -> bool:
        return True

    @property
    def supports_dividends(self) -> bool:
        return True

    @property
    def supports_insider_transactions(self) -> bool:
        return True

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return

        Returns:
            ResultList of FinancialStatement
        """
        endpoint = "income-statement" if period == "annual" else "income-statement"
        data = self.get(f"/{endpoint}/{symbol}", params={"period": period, "limit": limit})

        statements = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            fiscal_date = None
            if item.get("date"):
                try:
                    fiscal_date = date.fromisoformat(item["date"])
                except ValueError:
                    pass

            statements.append(FinancialStatement(
                symbol=symbol.upper(),
                statement_type="income",
                period=period,
                fiscal_date=fiscal_date or date.today(),
                currency=item.get("reportedCurrency", "USD"),
                data={
                    "revenue": item.get("revenue"),
                    "cost_of_revenue": item.get("costOfRevenue"),
                    "gross_profit": item.get("grossProfit"),
                    "gross_profit_ratio": item.get("grossProfitRatio"),
                    "operating_expenses": item.get("operatingExpenses"),
                    "operating_income": item.get("operatingIncome"),
                    "operating_income_ratio": item.get("operatingIncomeRatio"),
                    "ebitda": item.get("ebitda"),
                    "ebitda_ratio": item.get("ebitdaratio"),
                    "net_income": item.get("netIncome"),
                    "net_income_ratio": item.get("netIncomeRatio"),
                    "eps": item.get("eps"),
                    "eps_diluted": item.get("epsdiluted"),
                    "weighted_avg_shares": item.get("weightedAverageShsOut"),
                },
            ))

        return statements

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return

        Returns:
            ResultList of FinancialStatement
        """
        data = self.get(f"/balance-sheet-statement/{symbol}", params={"period": period, "limit": limit})

        statements = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            fiscal_date = None
            if item.get("date"):
                try:
                    fiscal_date = date.fromisoformat(item["date"])
                except ValueError:
                    pass

            statements.append(FinancialStatement(
                symbol=symbol.upper(),
                statement_type="balance_sheet",
                period=period,
                fiscal_date=fiscal_date or date.today(),
                currency=item.get("reportedCurrency", "USD"),
                data={
                    "total_assets": item.get("totalAssets"),
                    "total_current_assets": item.get("totalCurrentAssets"),
                    "cash_and_equivalents": item.get("cashAndCashEquivalents"),
                    "short_term_investments": item.get("shortTermInvestments"),
                    "inventory": item.get("inventory"),
                    "total_liabilities": item.get("totalLiabilities"),
                    "total_current_liabilities": item.get("totalCurrentLiabilities"),
                    "long_term_debt": item.get("longTermDebt"),
                    "total_equity": item.get("totalStockholdersEquity"),
                    "retained_earnings": item.get("retainedEarnings"),
                    "total_debt": item.get("totalDebt"),
                    "net_debt": item.get("netDebt"),
                },
            ))

        return statements

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> ResultList[FinancialStatement]:
        """
        Get cash flow statement data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarterly"
            limit: Number of periods to return

        Returns:
            ResultList of FinancialStatement
        """
        data = self.get(f"/cash-flow-statement/{symbol}", params={"period": period, "limit": limit})

        statements = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            fiscal_date = None
            if item.get("date"):
                try:
                    fiscal_date = date.fromisoformat(item["date"])
                except ValueError:
                    pass

            statements.append(FinancialStatement(
                symbol=symbol.upper(),
                statement_type="cash_flow",
                period=period,
                fiscal_date=fiscal_date or date.today(),
                currency=item.get("reportedCurrency", "USD"),
                data={
                    "operating_cash_flow": item.get("operatingCashFlow"),
                    "investing_cash_flow": item.get("netCashUsedForInvestingActivites"),
                    "financing_cash_flow": item.get("netCashUsedProvidedByFinancingActivities"),
                    "free_cash_flow": item.get("freeCashFlow"),
                    "capital_expenditure": item.get("capitalExpenditure"),
                    "dividends_paid": item.get("dividendsPaid"),
                    "stock_repurchased": item.get("commonStockRepurchased"),
                    "net_change_in_cash": item.get("netChangeInCash"),
                },
            ))

        return statements

    def get_dividends(self, symbol: str) -> ResultList[Dividend]:
        """
        Get dividend history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ResultList of Dividend
        """
        data = self.get(f"/historical-price-eod/full/stock_dividend/{symbol}")

        if isinstance(data, dict):
            data = data.get("historical", [])

        dividends = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            ex_date = None
            if item.get("date"):
                try:
                    ex_date = date.fromisoformat(item["date"])
                except ValueError:
                    continue

            if not ex_date:
                continue

            payment_date = None
            if item.get("paymentDate"):
                try:
                    payment_date = date.fromisoformat(item["paymentDate"])
                except ValueError:
                    pass

            record_date = None
            if item.get("recordDate"):
                try:
                    record_date = date.fromisoformat(item["recordDate"])
                except ValueError:
                    pass

            declaration_date = None
            if item.get("declarationDate"):
                try:
                    declaration_date = date.fromisoformat(item["declarationDate"])
                except ValueError:
                    pass

            dividends.append(Dividend(
                symbol=symbol.upper(),
                ex_date=ex_date,
                payment_date=payment_date,
                record_date=record_date,
                declaration_date=declaration_date,
                amount=item.get("adjDividend") or item.get("dividend", 0),
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
        data = self.get(f"/historical-price-eod/full/stock_split/{symbol}")

        if isinstance(data, dict):
            data = data.get("historical", [])

        splits = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            split_date = None
            if item.get("date"):
                try:
                    split_date = date.fromisoformat(item["date"])
                except ValueError:
                    continue

            if not split_date:
                continue

            # Parse split ratio (e.g., "4:1" means 4.0)
            ratio = 1.0
            numerator = item.get("numerator", 1)
            denominator = item.get("denominator", 1)
            if denominator and numerator:
                ratio = numerator / denominator

            splits.append(StockSplit(
                symbol=symbol.upper(),
                split_date=split_date,
                ratio=ratio,
                description=item.get("label"),
            ))

        return splits

    def get_insider_transactions(
        self,
        symbol: str,
        limit: int = 100,
    ) -> ResultList[InsiderTransaction]:
        """
        Get insider transactions.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum transactions to return

        Returns:
            ResultList of InsiderTransaction
        """
        data = self.get(f"/insider-trading", params={"symbol": symbol, "limit": limit})

        transactions = ResultList(provider=self.provider_name)
        for item in (data if isinstance(data, list) else []):
            tx_date = None
            if item.get("transactionDate"):
                try:
                    tx_date = date.fromisoformat(item["transactionDate"])
                except ValueError:
                    pass

            filing_date = None
            if item.get("filingDate"):
                try:
                    filing_date = date.fromisoformat(item["filingDate"])
                except ValueError:
                    pass

            transactions.append(InsiderTransaction(
                symbol=symbol.upper(),
                filing_date=filing_date,
                transaction_date=tx_date,
                insider_name=item.get("reportingName", "Unknown"),
                insider_title=item.get("typeOfOwner"),
                transaction_type=item.get("transactionType"),
                shares=item.get("securitiesTransacted"),
                price=item.get("price"),
                value=item.get("securitiesTransacted", 0) * item.get("price", 0) if item.get("price") else None,
                shares_owned_after=item.get("securitiesOwned"),
            ))

        return transactions
