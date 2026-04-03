"""
Company profile and earnings models.
"""

from datetime import date
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CompanyProfile(BaseModel):
    """
    Company profile and fundamental information.

    Normalized from:
    - FMP: get_profile
    - Finnhub: get_company_profile
    - Alpha Vantage: get_company_overview
    - Tiingo: get_meta
    - Yahoo Finance: info
    """
    symbol: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Company name")
    description: Optional[str] = Field(default=None, description="Business description")
    exchange: Optional[str] = Field(default=None, description="Primary exchange")
    sector: Optional[str] = Field(default=None, description="Business sector")
    industry: Optional[str] = Field(default=None, description="Industry classification")
    country: Optional[str] = Field(default=None, description="Country of incorporation")
    currency: Optional[str] = Field(default=None, description="Trading currency")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization")
    shares_outstanding: Optional[float] = Field(default=None, description="Shares outstanding")
    employees: Optional[int] = Field(default=None, description="Number of employees")
    website: Optional[str] = Field(default=None, description="Company website")
    logo_url: Optional[str] = Field(default=None, description="Logo URL")
    ceo: Optional[str] = Field(default=None, description="CEO name")
    ipo_date: Optional[date] = Field(default=None, description="IPO date")

    # Valuation metrics
    pe_ratio: Optional[float] = Field(default=None, description="Price to earnings ratio")
    pb_ratio: Optional[float] = Field(default=None, description="Price to book ratio")
    ps_ratio: Optional[float] = Field(default=None, description="Price to sales ratio")
    dividend_yield: Optional[float] = Field(default=None, description="Dividend yield")
    beta: Optional[float] = Field(default=None, description="Beta coefficient")

    # Financial metrics
    revenue: Optional[float] = Field(default=None, description="Annual revenue")
    net_income: Optional[float] = Field(default=None, description="Net income")
    eps: Optional[float] = Field(default=None, description="Earnings per share")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 2800000000000,
            "pe_ratio": 28.5
        }
    })


class FinancialStatement(BaseModel):
    """Financial statement data (income statement, balance sheet, cash flow)."""
    symbol: str = Field(..., description="Ticker symbol")
    statement_type: str = Field(..., description="Type: income, balance_sheet, cash_flow")
    period: str = Field(..., description="Period: annual, quarterly")
    fiscal_date: date = Field(..., description="Fiscal period end date")
    reported_date: Optional[date] = Field(default=None, description="Report filing date")
    currency: str = Field(default="USD", description="Currency")
    data: Dict[str, Any] = Field(default_factory=dict, description="Statement line items")


class EarningsEvent(BaseModel):
    """
    Earnings announcement event.

    Normalized from:
    - FMP: get_earnings_calendar
    - Finnhub: get_earnings_calendar
    - Alpha Vantage: get_earnings
    """
    symbol: str = Field(..., description="Ticker symbol")
    company_name: Optional[str] = Field(default=None, description="Company name")
    report_date: Optional[date] = Field(default=None, description="Expected report date")
    fiscal_quarter: Optional[str] = Field(default=None, description="Fiscal quarter (e.g., Q1 2024)")
    fiscal_year: Optional[int] = Field(default=None, description="Fiscal year")
    eps_estimate: Optional[float] = Field(default=None, description="EPS estimate")
    eps_actual: Optional[float] = Field(default=None, description="Actual EPS (after report)")
    revenue_estimate: Optional[float] = Field(default=None, description="Revenue estimate")
    revenue_actual: Optional[float] = Field(default=None, description="Actual revenue")
    report_time: Optional[str] = Field(default=None, description="Report time: bmo, amc, dmh")
    surprise: Optional[float] = Field(default=None, description="EPS surprise amount")
    surprise_percent: Optional[float] = Field(default=None, description="EPS surprise percentage")
    updated_from_date: Optional[date] = Field(default=None, description="Date estimate was updated")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "report_date": "2024-02-01",
            "fiscal_quarter": "Q1 2024",
            "fiscal_year": 2024,
            "eps_estimate": 2.10,
            "report_time": "amc"
        }
    })
