#Merton's helper
def merton_optimal_weight(mu_annual: float, r_annual: float, gamma: float, sigma_annual: float) -> float:
    """
    w* = (mu - r) / (gamma * sigma^2)
    """
    excess = mu_annual - r_annual
    denom = gamma * (sigma_annual ** 2)
    return excess / denom if denom != 0 else 0.0
