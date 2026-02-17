"""
Options contract and chain models.
"""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class OptionContract(BaseModel):
    """Single option contract."""
    symbol: str = Field(..., description="Option symbol/OCC code")
    underlying: str = Field(..., description="Underlying symbol")
    contract_type: str = Field(..., description="call or put")
    strike: float = Field(..., description="Strike price")
    expiration: date = Field(..., description="Expiration date")
    bid: Optional[float] = Field(default=None, description="Bid price")
    ask: Optional[float] = Field(default=None, description="Ask price")
    last: Optional[float] = Field(default=None, description="Last trade price")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    open_interest: Optional[int] = Field(default=None, description="Open interest")
    implied_volatility: Optional[float] = Field(default=None, description="Implied volatility")
    delta: Optional[float] = Field(default=None, description="Delta greek")
    gamma: Optional[float] = Field(default=None, description="Gamma greek")
    theta: Optional[float] = Field(default=None, description="Theta greek")
    vega: Optional[float] = Field(default=None, description="Vega greek")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "AAPL240315C00175000",
            "underlying": "AAPL",
            "contract_type": "call",
            "strike": 175.0,
            "expiration": "2024-03-15",
            "bid": 5.50,
            "ask": 5.70,
            "implied_volatility": 0.25,
            "delta": 0.55
        }
    })


class OptionChain(BaseModel):
    """
    Option chain for an underlying symbol.

    Normalized from:
    - Tradier: get_option_chains
    - Yahoo Finance: option_chain
    """
    underlying: str = Field(..., description="Underlying symbol")
    expiration: date = Field(..., description="Expiration date")
    calls: List[OptionContract] = Field(default_factory=list, description="Call options")
    puts: List[OptionContract] = Field(default_factory=list, description="Put options")
    underlying_price: Optional[float] = Field(default=None, description="Current underlying price")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "underlying": "AAPL",
            "expiration": "2024-03-15",
            "underlying_price": 175.50,
            "calls": [],
            "puts": []
        }
    })
