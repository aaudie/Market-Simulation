"""
News, insider transactions, and analyst data models.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field


class NewsArticle(BaseModel):
    """
    News article from market data providers.

    Normalized from:
    - Finnhub: company_news
    - Yahoo Finance: news
    - Tiingo: news
    - Alpaca: news
    """
    headline: str = Field(..., description="Article headline")
    summary: Optional[str] = Field(default=None, description="Article summary/snippet")
    source: Optional[str] = Field(default=None, description="News source name")
    url: Optional[str] = Field(default=None, description="Article URL")
    image_url: Optional[str] = Field(default=None, description="Article image URL")
    published_at: Optional[datetime] = Field(default=None, description="Publication timestamp")
    symbols: list[str] = Field(default_factory=list, description="Related ticker symbols")
    category: Optional[str] = Field(default=None, description="News category")
    sentiment: Optional[float] = Field(default=None, description="Sentiment score (-1 to 1)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "headline": "Apple Reports Record Q4 Earnings",
            "source": "Reuters",
            "published_at": "2024-01-30T14:30:00Z",
            "symbols": ["AAPL"],
            "sentiment": 0.8
        }
    })


class InsiderTransaction(BaseModel):
    """
    Insider transaction (Form 4 filings).

    Normalized from:
    - Finnhub: insider_transactions
    - Yahoo Finance: insider_transactions
    - SEC EDGAR: Form 4
    """
    symbol: str = Field(..., description="Ticker symbol")
    filing_date: Optional[date] = Field(default=None, description="SEC filing date")
    transaction_date: Optional[date] = Field(default=None, description="Transaction date")
    insider_name: str = Field(..., description="Insider name")
    insider_title: Optional[str] = Field(default=None, description="Insider title/role")
    transaction_type: Optional[str] = Field(default=None, description="Buy, Sell, Gift, etc.")
    shares: Optional[float] = Field(default=None, description="Number of shares")
    price: Optional[float] = Field(default=None, description="Price per share")
    value: Optional[float] = Field(default=None, description="Total transaction value")
    shares_owned_after: Optional[float] = Field(default=None, description="Shares owned after transaction")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL",
            "insider_name": "Tim Cook",
            "insider_title": "CEO",
            "transaction_type": "Sell",
            "shares": 50000,
            "price": 185.50,
            "value": 9275000
        }
    })


class AnalystRecommendation(BaseModel):
    """
    Analyst recommendation/rating.

    Normalized from:
    - Finnhub: recommendation_trends
    - Yahoo Finance: recommendations
    """
    symbol: str = Field(..., description="Ticker symbol")
    period: Optional[date] = Field(default=None, description="Recommendation period")
    strong_buy: int = Field(default=0, description="Number of strong buy ratings")
    buy: int = Field(default=0, description="Number of buy ratings")
    hold: int = Field(default=0, description="Number of hold ratings")
    sell: int = Field(default=0, description="Number of sell ratings")
    strong_sell: int = Field(default=0, description="Number of strong sell ratings")

    def total_analysts(self) -> int:
        """Total number of analysts."""
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    def consensus(self) -> str:
        """Get consensus recommendation."""
        if self.total_analysts() == 0:
            return "N/A"
        # Weighted score: strong_buy=5, buy=4, hold=3, sell=2, strong_sell=1
        score = (self.strong_buy * 5 + self.buy * 4 + self.hold * 3 +
                 self.sell * 2 + self.strong_sell * 1) / self.total_analysts()
        if score >= 4.5:
            return "Strong Buy"
        elif score >= 3.5:
            return "Buy"
        elif score >= 2.5:
            return "Hold"
        elif score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"


class PriceTarget(BaseModel):
    """
    Analyst price target.

    Normalized from:
    - Finnhub: price_target
    - Yahoo Finance: analyst_price_target
    """
    symbol: str = Field(..., description="Ticker symbol")
    target_high: Optional[float] = Field(default=None, description="Highest price target")
    target_low: Optional[float] = Field(default=None, description="Lowest price target")
    target_mean: Optional[float] = Field(default=None, description="Mean price target")
    target_median: Optional[float] = Field(default=None, description="Median price target")
    num_analysts: Optional[int] = Field(default=None, description="Number of analysts")
    last_updated: Optional[date] = Field(default=None, description="Last update date")


class Dividend(BaseModel):
    """
    Dividend payment information.

    Normalized from:
    - Yahoo Finance: dividends
    - FMP: dividends
    - Polygon: dividends
    """
    symbol: str = Field(..., description="Ticker symbol")
    ex_date: date = Field(..., description="Ex-dividend date")
    payment_date: Optional[date] = Field(default=None, description="Payment date")
    record_date: Optional[date] = Field(default=None, description="Record date")
    declaration_date: Optional[date] = Field(default=None, description="Declaration date")
    amount: float = Field(..., description="Dividend amount per share")
    currency: str = Field(default="USD", description="Currency")
    frequency: Optional[str] = Field(default=None, description="Frequency: quarterly, annual, etc.")


class StockSplit(BaseModel):
    """
    Stock split information.

    Normalized from:
    - Yahoo Finance: splits
    - FMP: splits
    - Polygon: splits
    """
    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., description="Ticker symbol")
    split_date: date = Field(..., description="Split date", alias="date")
    ratio: float = Field(..., description="Split ratio (e.g., 4.0 for 4:1 split)")
    description: Optional[str] = Field(default=None, description="Split description")
