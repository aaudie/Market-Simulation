"""
Type definitions and data classes for market simulation.

Defines core data structures used throughout the simulation.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class HistoricalPoint:
    """
    Single historical price observation.
    
    Attributes:
        date: Observation timestamp
        price: Closing price
    """
    date: datetime
    price: float


@dataclass
class CalibratedParams:
    """
    Calibrated drift and volatility parameters.
    
    Attributes:
        mu_monthly: Monthly drift (log return)
        sigma_monthly: Monthly volatility
        mu_annual: Annualized drift
        sigma_annual: Annualized volatility
    """
    mu_monthly: float
    sigma_monthly: float
    mu_annual: float
    sigma_annual: float


@dataclass
class ScenarioParams:
    """
    Simulation scenario configuration.
    
    Defines drift, volatility, and microstructure parameters for a simulation run.
    
    Attributes:
        name: Scenario name/description
        mu_monthly: Monthly drift parameter
        sigma_monthly: Monthly volatility parameter
        tick_size: Minimum price increment
        default_spread_ticks: Default bid-ask spread in ticks
        default_depth_levels: Number of price levels in order book
        default_level_size: Quantity at each price level
        trader_activity_rate: Probability of trader activity per tick
        adoption_curve: Tokenization adoption curve type ('linear' or 'logistic')
        anchor_weight: Weight on fundamental anchor (0=micro only, 1=fully anchored)
    """
    name: str
    mu_monthly: float
    sigma_monthly: float

    # Microstructure parameters
    tick_size: int = 1
    default_spread_ticks: int = 1
    default_depth_levels: int = 5
    default_level_size: int = 10

    # Agent behavior
    trader_activity_rate: float = 0.2

    # Tokenization parameters
    adoption_curve: str = "linear"
    anchor_weight: float = 0.6
