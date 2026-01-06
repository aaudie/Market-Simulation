# sim/regimes.py
from __future__ import annotations

from typing import List, Tuple, Optional
import random

STATES = ["calm", "neutral", "volatile", "panic"]

# --- Markov chain ---
class RegimeMarkovChain:
    def __init__(
        self,
        states: List[str],
        P: List[List[float]],
        start_state: str = "neutral",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}
        self.P = P
        self.state = start_state
        self.rng = rng or random.Random()

    def step(self) -> str:
        i = self.index[self.state]
        probs = self.P[i]
        self.state = self.rng.choices(self.states, weights=probs)[0]
        return self.state

    def set_transition_matrix(self, P_new: List[List[float]]) -> None:
        self.P = P_new


# --- Regime mapping helpers ---
def sigma_multiplier(regime: str) -> float:
    return {
        "calm": 0.8,
        "neutral": 1.0,
        "volatile": 1.5,
        "panic": 2.2,
    }[regime]

def regime_severity(regime: str) -> int:
    # higher = worse
    return {
        "calm": 0,
        "neutral": 1,
        "volatile": 2,
        "panic": 3,
    }[regime]

def label_regime_from_realized_vol(realized_sigma: float, base_sigma: float) -> str:
    """
    Your existing thresholding logic, expressed as a pure function.
    """
    b = base_sigma
    if realized_sigma < 0.7 * b:
        return "calm"
    elif realized_sigma < 1.2 * b:
        return "neutral"
    elif realized_sigma < 2.0 * b:
        return "volatile"
    else:
        return "panic"


# --- Hybrid combining rule ---
def combine_markov_and_realized(
    markov_regime: str,
    realized_regime: str,
) -> str:
    """
    Conservative hybrid:
    - Markov gives baseline macro regime.
    - Realized volatility can *upgrade* severity if it's worse.
    """
    return realized_regime if regime_severity(realized_regime) > regime_severity(markov_regime) else markov_regime


def sigma_from_regime(base_sigma: float, regime: str) -> float:
    return base_sigma * sigma_multiplier(regime)
