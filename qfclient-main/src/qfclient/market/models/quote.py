"""
Quote and OHLCV models for market data.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from ...common.types import Interval


class Quote(BaseModel):
    """
    Real-time or delayed quote for a symbol.

    Normalized from various providers:
    - Alpaca: get_latest_quote
    - Finnhub: get_quote
    - Alpha Vantage: get_quote
    - Twelve Data: get_quote
    - Yahoo Finance: info
    """
    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., description="Current/last price")
    bid: Optional[float] = Field(default=None, description="Best bid price")
    ask: Optional[float] = Field(default=None, description="Best ask price")
    bid_size: Optional[int] = Field(default=None, description="Bid size")
    ask_size: Optional[int] = Field(default=None, description="Ask size")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    open: Optional[float] = Field(default=None, description="Day open price")
    high: Optional[float] = Field(default=None, description="Day high price")
    low: Optional[float] = Field(default=None, description="Day low price")
    previous_close: Optional[float] = Field(default=None, description="Previous day close")
    change: Optional[float] = Field(default=None, description="Price change from prev close")
    change_percent: Optional[float] = Field(default=None, description="Percent change")
    timestamp: Optional[datetime] = Field(default=None, description="Quote timestamp")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL",
            "price": 175.50,
            "bid": 175.48,
            "ask": 175.52,
            "volume": 45000000,
            "change": 2.30,
            "change_percent": 1.33
        }
    })


class OHLCV(BaseModel):
    """
    OHLCV (Open, High, Low, Close, Volume) candlestick data.

    Normalized from various providers:
    - Alpaca: get_bars
    - Polygon: get_aggregates
    - Twelve Data: get_time_series
    - Yahoo Finance: history
    - FMP: get_historical_price
    """
    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(..., description="Candle timestamp (start of period)")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    vwap: Optional[float] = Field(default=None, description="Volume-weighted average price")
    trades: Optional[int] = Field(default=None, description="Number of trades")
    interval: Optional[Interval] = Field(default=None, description="Candle interval")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL",
            "timestamp": "2024-01-15T09:30:00Z",
            "open": 175.00,
            "high": 176.50,
            "low": 174.80,
            "close": 175.50,
            "volume": 5000000,
            "interval": "1d"
        }
    })
