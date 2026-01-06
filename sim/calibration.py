#history calibration
import math
from typing import List

from sim.types import HistoricalPoint, CalibratedParams

def calibrate_from_history(history: List[HistoricalPoint]) -> CalibratedParams:
    """
    Calibrate monthly mu and sigma from log returns of history.
    """
    if len(history) < 2:
        return CalibratedParams(0.0, 0.0, 0.0, 0.0)

    prices = [p.price for p in history]
    rets = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0 and prices[i] > 0:
            rets.append(math.log(prices[i] / prices[i-1]))

    if len(rets) < 2:
        return CalibratedParams(0.0, 0.0, 0.0, 0.0)

    mu_m = sum(rets) / len(rets)
    var_m = sum((x - mu_m) ** 2 for x in rets) / (len(rets) - 1)
    sig_m = math.sqrt(var_m)

    mu_a = mu_m * 12.0
    sig_a = sig_m * math.sqrt(12.0)

    return CalibratedParams(
        mu_monthly=mu_m,
        sigma_monthly=sig_m,
        mu_annual=mu_a,
        sigma_annual=sig_a,
    )