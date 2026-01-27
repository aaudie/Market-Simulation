"""
Portfolio optimization utilities.

Provides functions for calculating optimal portfolio allocations.
"""


def merton_optimal_weight(
    mu_annual: float, 
    r_annual: float, 
    gamma: float, 
    sigma_annual: float
) -> float:
    """
    Calculate Merton's optimal portfolio weight for a risky asset.
    
    Formula: w* = (μ - r) / (γ * σ²)
    
    Where:
        μ = expected return of risky asset
        r = risk-free rate
        γ = risk aversion coefficient
        σ = volatility of risky asset
    
    Args:
        mu_annual: Expected annual return of risky asset
        r_annual: Risk-free annual rate
        gamma: Risk aversion coefficient (higher = more risk averse)
        sigma_annual: Annual volatility of risky asset
        
    Returns:
        Optimal weight in risky asset (can be > 1 or < 0 for leverage/short)
        Returns 0 if denominator would be zero
    """
    excess = mu_annual - r_annual
    denom = gamma * (sigma_annual ** 2)
    return excess / denom if denom != 0 else 0.0
