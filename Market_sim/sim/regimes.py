"""
Market regime dynamics and Markov chain transitions.

Implements regime classification and state transition logic for market volatility.
"""

from __future__ import annotations

from typing import List, Optional
import random

# Market regime states (ordered by severity)
STATES = ["calm", "neutral", "volatile", "panic"]


class RegimeMarkovChain:
    """
    Markov chain for regime transitions.
    
    Simulates transitions between market regimes using a transition matrix.
    Each state has probabilities of transitioning to other states.
    """
    
    def __init__(
        self,
        states: List[str],
        P: List[List[float]],
        start_state: str = "neutral",
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize Markov chain.
        
        Args:
            states: List of state names
            P: Transition matrix (row i = probabilities from state i)
            start_state: Initial state
            rng: Random number generator (creates new one if None)
        """
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}
        self.P = P
        self.state = start_state
        self.rng = rng or random.Random()

    def step(self) -> str:
        """
        Execute one transition step.
        
        Returns:
            New state after transition
        """
        i = self.index[self.state]
        probs = self.P[i]
        self.state = self.rng.choices(self.states, weights=probs)[0]
        return self.state

    def set_transition_matrix(self, P_new: List[List[float]]) -> None:
        """
        Update transition matrix.
        
        Args:
            P_new: New transition matrix
        """
        self.P = P_new



# =============================================================================
# Regime Classification and Mapping
# =============================================================================

def sigma_multiplier(regime: str) -> float:
    """
    Get volatility multiplier for a given regime.
    
    Args:
        regime: Regime name ('calm', 'neutral', 'volatile', or 'panic')
        
    Returns:
        Multiplier to apply to base volatility
    """
    return {
        "calm": 0.8,
        "neutral": 1.0,
        "volatile": 1.5,
        "panic": 2.2,
    }[regime]


def regime_severity(regime: str) -> int:
    """
    Get numeric severity level for a regime.
    
    Higher values indicate more severe/volatile regimes.
    
    Args:
        regime: Regime name
        
    Returns:
        Severity level (0=calm, 3=panic)
    """
    return {
        "calm": 0,
        "neutral": 1,
        "volatile": 2,
        "panic": 3,
    }[regime]


def label_regime_from_realized_vol(realized_sigma: float, base_sigma: float) -> str:
    """
    Classify regime based on realized volatility.
    
    Uses threshold multiples of base volatility to classify:
    - < 0.7x base: calm
    - 0.7x - 1.2x: neutral
    - 1.2x - 2.0x: volatile
    - > 2.0x: panic
    
    Args:
        realized_sigma: Observed volatility
        base_sigma: Baseline/calibrated volatility
        
    Returns:
        Regime label
    """
    if realized_sigma < 0.7 * base_sigma:
        return "calm"
    elif realized_sigma < 1.2 * base_sigma:
        return "neutral"
    elif realized_sigma < 2.0 * base_sigma:
        return "volatile"
    else:
        return "panic"


def combine_markov_and_realized(
    markov_regime: str,
    realized_regime: str,
) -> str:
    """
    Combine Markov and realized-volatility regimes.
    
    Conservative hybrid approach:
    - Markov chain provides baseline macro regime
    - Realized volatility can upgrade to more severe regime if observed
    - Never downgrades severity based on Markov alone
    
    Args:
        markov_regime: Regime from Markov chain
        realized_regime: Regime inferred from realized volatility
        
    Returns:
        Combined regime (more severe of the two)
    """
    if regime_severity(realized_regime) > regime_severity(markov_regime):
        return realized_regime
    return markov_regime


def sigma_from_regime(base_sigma: float, regime: str) -> float:
    """
    Calculate volatility for a given regime.
    
    Args:
        base_sigma: Base volatility parameter
        regime: Current regime
        
    Returns:
        Regime-adjusted volatility
    """
    return base_sigma * sigma_multiplier(regime)
