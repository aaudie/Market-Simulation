#dataclasses
# sim/types.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HistoricalPoint:
    date: datetime
    price: float

@dataclass
class CalibratedParams:
    mu_monthly: float
    sigma_monthly: float
    mu_annual: float
    sigma_annual: float

@dataclass
class ScenarioParams:
    # name: str
    # mu_monthly: float
    # sigma_monthly: float
    """
    main.py can keep constructing this with just (name, mu_monthly, sigma_monthly).
    The rest have defaults so the simulator/agents don't hit missing attributes.
    """
    name: str
    mu_monthly: float
    sigma_monthly: float

    # --- microstructure defaults ---
    tick_size: int = 1
    default_spread_ticks: int = 1
    default_depth_levels: int = 5
    default_level_size: int = 10

    # --- agent behavior ---
    trader_activity_rate: float = 0.2

    # --- tokenization/adoption knobs ---
    adoption_curve: str = "linear"   # 'linear' | 'logistic'
    anchor_weight: float = 0.6       # 0..1
