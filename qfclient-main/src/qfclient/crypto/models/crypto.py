"""
Cryptocurrency data models.
"""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from ...common.types import Interval


class CryptoQuote(BaseModel):
    """
    Real-time or delayed quote for a cryptocurrency.

    Normalized from:
    - CoinGecko: get_price
    - CoinMarketCap: get_quotes_latest
    - Messari: get_asset_metrics
    """
    symbol: str = Field(..., description="Crypto symbol (e.g., BTC, ETH)")
    name: Optional[str] = Field(default=None, description="Full name (e.g., Bitcoin)")
    price_usd: float = Field(..., description="Price in USD")
    price_btc: Optional[float] = Field(default=None, description="Price in BTC")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization USD")
    volume_24h: Optional[float] = Field(default=None, description="24h trading volume USD")
    change_1h: Optional[float] = Field(default=None, description="1 hour price change %")
    change_24h: Optional[float] = Field(default=None, description="24 hour price change %")
    change_7d: Optional[float] = Field(default=None, description="7 day price change %")
    change_30d: Optional[float] = Field(default=None, description="30 day price change %")
    circulating_supply: Optional[float] = Field(default=None, description="Circulating supply")
    total_supply: Optional[float] = Field(default=None, description="Total supply")
    max_supply: Optional[float] = Field(default=None, description="Maximum supply")
    market_cap_rank: Optional[int] = Field(default=None, description="Market cap rank")
    timestamp: Optional[datetime] = Field(default=None, description="Quote timestamp")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "BTC",
            "name": "Bitcoin",
            "price_usd": 95000.00,
            "market_cap": 1850000000000,
            "volume_24h": 45000000000,
            "change_24h": 2.5,
            "market_cap_rank": 1
        }
    })


class CryptoOHLCV(BaseModel):
    """
    OHLCV candlestick data for cryptocurrencies.

    Normalized from:
    - CoinGecko: get_coin_market_chart
    - CoinMarketCap: get_ohlcv
    """
    symbol: str = Field(..., description="Crypto symbol")
    timestamp: datetime = Field(..., description="Candle timestamp")
    open: float = Field(..., description="Opening price USD")
    high: float = Field(..., description="High price USD")
    low: float = Field(..., description="Low price USD")
    close: float = Field(..., description="Closing price USD")
    volume: float = Field(..., description="Trading volume USD")
    market_cap: Optional[float] = Field(default=None, description="Market cap at close")
    interval: Optional[Interval] = Field(default=None, description="Candle interval")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "BTC",
            "timestamp": "2024-01-15T00:00:00Z",
            "open": 94000.00,
            "high": 96000.00,
            "low": 93500.00,
            "close": 95000.00,
            "volume": 45000000000,
            "interval": "1d"
        }
    })


class CryptoAsset(BaseModel):
    """
    Crypto asset profile and metadata.

    Normalized from:
    - CoinGecko: get_coin
    - CoinMarketCap: get_metadata
    - Messari: get_asset
    """
    symbol: str = Field(..., description="Crypto symbol")
    name: str = Field(..., description="Full name")
    slug: Optional[str] = Field(default=None, description="URL-friendly slug")
    description: Optional[str] = Field(default=None, description="Asset description")
    category: Optional[str] = Field(default=None, description="Category (e.g., Currency, DeFi)")
    sector: Optional[str] = Field(default=None, description="Sector classification")
    tags: List[str] = Field(default_factory=list, description="Tags/categories")

    # Links
    website: Optional[str] = Field(default=None, description="Official website")
    whitepaper: Optional[str] = Field(default=None, description="Whitepaper URL")
    github: Optional[str] = Field(default=None, description="GitHub repository")
    twitter: Optional[str] = Field(default=None, description="Twitter handle")
    reddit: Optional[str] = Field(default=None, description="Reddit community")

    # Technical
    blockchain: Optional[str] = Field(default=None, description="Blockchain platform")
    consensus: Optional[str] = Field(default=None, description="Consensus mechanism")
    genesis_date: Optional[date] = Field(default=None, description="Launch date")

    # Supply
    circulating_supply: Optional[float] = Field(default=None)
    total_supply: Optional[float] = Field(default=None)
    max_supply: Optional[float] = Field(default=None)

    # Market
    market_cap_rank: Optional[int] = Field(default=None)
    coingecko_rank: Optional[int] = Field(default=None)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "BTC",
            "name": "Bitcoin",
            "slug": "bitcoin",
            "category": "Cryptocurrency",
            "blockchain": "Bitcoin",
            "consensus": "Proof-of-Work",
            "max_supply": 21000000,
            "market_cap_rank": 1
        }
    })


class CryptoMarketData(BaseModel):
    """
    Comprehensive market data for a crypto asset.

    Combines quote data with additional metrics.
    """
    symbol: str = Field(..., description="Crypto symbol")
    name: Optional[str] = Field(default=None)

    # Price data
    price_usd: float = Field(...)
    price_btc: Optional[float] = Field(default=None)

    # Market metrics
    market_cap: Optional[float] = Field(default=None)
    fully_diluted_valuation: Optional[float] = Field(default=None)
    volume_24h: Optional[float] = Field(default=None)
    market_cap_rank: Optional[int] = Field(default=None)

    # Price changes
    change_1h: Optional[float] = Field(default=None)
    change_24h: Optional[float] = Field(default=None)
    change_7d: Optional[float] = Field(default=None)
    change_30d: Optional[float] = Field(default=None)
    change_1y: Optional[float] = Field(default=None)

    # Price extremes
    ath: Optional[float] = Field(default=None, description="All-time high USD")
    ath_date: Optional[datetime] = Field(default=None)
    ath_change_percent: Optional[float] = Field(default=None)
    atl: Optional[float] = Field(default=None, description="All-time low USD")
    atl_date: Optional[datetime] = Field(default=None)
    atl_change_percent: Optional[float] = Field(default=None)

    # Supply
    circulating_supply: Optional[float] = Field(default=None)
    total_supply: Optional[float] = Field(default=None)
    max_supply: Optional[float] = Field(default=None)

    # Timestamp
    last_updated: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "BTC",
            "name": "Bitcoin",
            "price_usd": 95000.00,
            "market_cap": 1850000000000,
            "volume_24h": 45000000000,
            "change_24h": 2.5,
            "market_cap_rank": 1,
            "ath": 108000.00,
            "max_supply": 21000000
        }
    })
