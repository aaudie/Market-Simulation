# market_sim.py
# Market Simulator with historical CRE input + tokenized scenarios

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import random
import time
import math
import csv
from datetime import datetime

class MathX:
    @staticmethod
    def round_to_int(x: float) -> int:
        return int(round(x))

    @staticmethod
    def sign(x: float) -> int:
        if x > 0: return 1
        if x < 0: return -1
        return 0

    @staticmethod
    def min_int(a: int, b: int) -> int:
        return a if a <= b else b

# ---------- Historical CRE + calibration + Merton helpers ----------

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
    name: str
    mu_monthly: float
    sigma_monthly: float

def load_cre_csv(path: str) -> List[HistoricalPoint]:
    """
    CSV format: date,price
    date can be YYYY-MM or YYYY-MM-DD.
    """
    points: List[HistoricalPoint] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            date_str, price_str = row[0].strip(), row[1].strip()
            # try YYYY-MM, then YYYY-MM-DD
            try:
                dt = datetime.strptime(date_str, "%Y-%m")
            except ValueError:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            price = float(price_str)
            points.append(HistoricalPoint(date=dt, price=price))
    points.sort(key=lambda p: p.date)
    return points

def calibrate_from_history(history: List[HistoricalPoint]) -> CalibratedParams:
    """
    Compute monthly log-return mean and std, then annualize.
    """
    if len(history) < 2:
        raise ValueError("Need at least 2 points to calibrate.")

    rets: List[float] = []
    for i in range(1, len(history)):
        p_t = history[i].price
        p_prev = history[i - 1].price
        r = math.log(p_t / p_prev)
        rets.append(r)

    n = len(rets)
    mu_monthly = sum(rets) / n
    var = sum((r - mu_monthly) ** 2 for r in rets) / (n - 1)
    sigma_monthly = math.sqrt(var)

    mu_annual = 12.0 * mu_monthly
    sigma_annual = math.sqrt(12.0) * sigma_monthly

    return CalibratedParams(
        mu_monthly=mu_monthly,
        sigma_monthly=sigma_monthly,
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
    )

def merton_optimal_weight(mu_annual: float, r_annual: float, gamma: float,
    sigma_annual: float) -> float:
    """
    w* = (mu - r) / (gamma * sigma^2)
    """
    excess = mu_annual - r_annual
    denom = gamma * (sigma_annual ** 2)
    return excess / denom if denom != 0 else 0.0

# ------------------------------------------------------------------------

@dataclass
class Candlestick:
    open: int = 0
    close: int = 0
    high: int = 0
    low: int = 0
    volume: int = 0

    def reset(self, seed_price: int) -> None:
        self.open = seed_price
        self.close = seed_price
        self.high = seed_price
        self.low = seed_price
        self.volume = 0

    def update_with_trade(self, trade_price: int, qty: int) -> None:
        if self.volume == 0:
            self.open = trade_price
            self.high = trade_price
            self.low = trade_price
        self.close = trade_price
        if trade_price > self.high:
            self.high = trade_price
        if trade_price < self.low:
            self.low = trade_price
        self.volume += qty

class Chart:
    def update_live_candle(self, *args, **kwargs) -> None:
        pass

class OrderBook:
    def __init__(self, market: "MarketSimulator") -> None:
        self.market = market
        self.buy_book: Dict[int, int] = defaultdict(int)
        self.sell_book: Dict[int, int] = defaultdict(int)
        self.best_bid: int = 0
        self.best_ask: int = 2**31 - 1
        self.total_buy_volume: int = 0
        self.total_sell_volume: int = 0
        self.imbalance: float = 0.0
        self.last_price: int = 100
        self.last_traded_price: int = 100
        self.trader_activity_rate: float = market.trader_activity_rate

    def _refresh_best_prices(self) -> None:
        self.best_bid = max(self.buy_book.keys()) if self.buy_book else 0
        self.best_ask = min(self.sell_book.keys()) if self.sell_book else 2**31 - 1

    def place_order(self, price: int, quantity: int, is_buy: bool) -> None:
        if price <= 0 or quantity <= 0:
            return
        self._refresh_best_prices()
        if is_buy:
            if self.sell_book and price >= self.best_ask:
                self._take_order(price, quantity, True)
            else:
                self._make_order(price, quantity, True)
        else:
            if self.buy_book and price <= self.best_bid:
                self._take_order(price, quantity, False)
            else:
                self._make_order(price, quantity, False)

    def _make_order(self, price: int, quantity: int, is_buy: bool) -> None:
        book = self.buy_book if is_buy else self.sell_book
        book[price] += quantity
        if is_buy:
            self.total_buy_volume += quantity
        else:
            self.total_sell_volume += quantity
        self._refresh_best_prices()

    def _take_order(self, limit_price: int, qty: int, is_buy: bool) -> None:
        remaining = qty
        self._refresh_best_prices()
        while remaining > 0:
            if is_buy:
                if not self.sell_book:
                    self._refresh_best_prices()
                    break
                best_ask = self.best_ask
                if limit_price < best_ask:
                    break
                avail = self.sell_book[best_ask]
                trade_q = MathX.min_int(avail, remaining)
                self.sell_book[best_ask] -= trade_q
                if self.sell_book[best_ask] <= 0:
                    del self.sell_book[best_ask]
                self.total_buy_volume += trade_q
                self.last_traded_price = best_ask
                self.market.live_candlestick.update_with_trade(best_ask, trade_q)
                self.market.chart.update_live_candle(
                    limit_price, self.market.live_candlestick.volume,
                    self.market.live_candlestick.high, self.market.live_candlestick.low
                )
                remaining -= trade_q
            else:
                if not self.buy_book:
                    self._refresh_best_prices()
                    break
                best_bid = self.best_bid
                if limit_price > best_bid:
                    break
                avail = self.buy_book[best_bid]
                trade_q = MathX.min_int(avail, remaining)
                self.buy_book[best_bid] -= trade_q
                if self.buy_book[best_bid] <= 0:
                    del self.buy_book[best_bid]
                self.total_sell_volume += trade_q
                self.last_traded_price = best_bid
                self.market.live_candlestick.update_with_trade(best_bid, trade_q)
                self.market.chart.update_live_candle(
                    limit_price, self.market.live_candlestick.volume,
                    self.market.live_candlestick.high, self.market.live_candlestick.low
                )
                remaining -= trade_q
            self._refresh_best_prices()
        if self.last_traded_price > 0:
            self.last_price = self.last_traded_price
        tv = self.total_buy_volume + self.total_sell_volume
        self.imbalance = (self.total_buy_volume - self.total_sell_volume) / tv if tv > 0 else 0.0
        if remaining > 0:
            self._make_order(limit_price, remaining, is_buy)

class Trader:
    def __init__(self, order_book: OrderBook, rng: random.Random) -> None:
        self.order_book = order_book
        self.rng = rng

    def try_place_orders(self) -> None:
        if self.rng.random() < self.order_book.trader_activity_rate:
            self._place_orders()

    def _place_orders(self) -> None:
        p, q, is_buy = self._determine_order()
        self.order_book.place_order(p, q, is_buy)

    def _determine_order(self) -> Tuple[int, int, bool]:
        # Bias orders around market.target_price if set
        target = self.order_book.market.target_price
        base_price = target if target is not None else self.order_book.last_price

        # --- VOLATILITY-SCALED ORDER-BOOK WIDTH ---
        sigma = self.order_book.market.current_sigma_monthly or 0.02  # fallback
        base_spread = 0.02     # 2% minimum band
        k = 1.5                # how sensitive spreads are to volatility
        spread = base_spread + k * sigma
        spread = min(max(spread, 0.01), 0.20)  # clamp between 1% and 20%

        rand_factor = 1.0 + self.rng.uniform(-spread, spread)
        rand_price = MathX.round_to_int(base_price * rand_factor)

        is_buy = rand_price > self.order_book.last_price
        qty = self.rng.randint(1, 20)
        return rand_price, qty, is_buy

class BatchManager:
    def evaluate_batch_step(self) -> None:
        pass

class MarketSimulator:
    def __init__(self) -> None:
        # ---------- volatility regime state ----------
        self.base_sigma_monthly
        self.current_sigma_monthly
        self.regime
        self.base_sigma_monthly: float = 0.03       # default; replaced when scenario attached
        self.current_sigma_monthly: float = self.base_sigma_monthly
        self.regime: str = "neutral"
        self._ret_window: List[float] = []          # rolling window of log returns
        self._regime_window: int = 6                # lookback in "months"
        self._last_fundamental_price: Optional[float] = None
        
        self.active: bool = True
        self.market_speed: float = 10.0
        self.trader_count: int = 1000
        self.trader_activity_rate: float = 0.0005
        self.chart = Chart()
        self.live_candlestick = Candlestick()
        self.order_book = OrderBook(self)
        self.rng = random.Random()
        self.traders: List[Trader] = [
            Trader(self.order_book, self.rng) for _ in range(self.trader_count)
        ]
        if self.order_book.last_price <= 0:
            self.order_book.last_price = 100
        self.live_candlestick.reset(self.order_book.last_price)

        # ---------- fundamental price path / CRE integration ----------
        self.target_price: Optional[int] = None      # "fundamental" price
        self._fundamental_price: Optional[float] = None
        self._scenario: Optional[ScenarioParams] = None
        self._history_prices: List[float] = []
        self._history_index: int = 0

    # ---------- Volatility regime updater ----------

    def _update_vol_regime(self, new_fundamental_price: float) -> None:
        """
        Update rolling realized volatility and choose a regime.
        Adjust current_sigma_monthly as a multiple of base_sigma_monthly.
        """
        if self._last_fundamental_price is None:
            self._last_fundamental_price = new_fundamental_price
            return

        # log return from previous month
        r = math.log(new_fundamental_price / self._last_fundamental_price)
        self._last_fundamental_price = new_fundamental_price

        self._ret_window.append(r)
        if len(self._ret_window) > self._regime_window:
            self._ret_window.pop(0)

        if len(self._ret_window) < 2:
            return  # not enough data yet

        mu = sum(self._ret_window) / len(self._ret_window)
        var = sum((x - mu) ** 2 for x in self._ret_window) / (len(self._ret_window) - 1)
        realized_sigma = math.sqrt(var)

        # Compare realized volatility to base_sigma to choose regime
        b = self.base_sigma_monthly
        if realized_sigma < 0.7 * b:
            regime = "calm"
            mult = 0.8
        elif realized_sigma < 1.2 * b:
            regime = "neutral"
            mult = 1.0
        elif realized_sigma < 2.0 * b:
            regime = "volatile"
            mult = 1.5
        else:
            regime = "panic"
            mult = 2.0

        self.regime = regime
        self.current_sigma_monthly = b * mult
        # (optional) debug
        # print(f"Regime={regime}, realized_sigma={realized_sigma:.4f}, current_sigma={self.current_sigma_monthly:.4f}")


    def attach_history_and_scenario(self,
                                    history_points: List[HistoricalPoint],
                                    scenario: ScenarioParams) -> None:
        """
        History: list of CRE prices (monthly).
        Scenario: mu/sigma for tokenized regime.
        """
        self._history_prices = [p.price for p in history_points]
        self._history_index = 0
        self._scenario = scenario

        # set base sigma from scenario (neutral tokenized CRE volatility)
        self.base_sigma_monthly = scenario.sigma_monthly
        self.current_sigma_monthly = self.base_sigma_monthly
        self.regime = "neutral"
        self._ret_window = []
        self._last_fundamental_price = None

        if self._history_prices:
            self._fundamental_price = self._history_prices[0]
        else:
            self._fundamental_price = float(self.order_book.last_price)
        # initialize target + order book price from fundamental
        seed = MathX.round_to_int(self._fundamental_price)
        self.target_price = seed
        self.order_book.last_price = seed
        self.live_candlestick.reset(seed)


    def _step_fundamental_month(self) -> None:
        """
        One "month" step of the fundamental price:
        - replay history while there is history
        - after that, simulate GBM with (regime-adjusted) sigma
        """
        if self._fundamental_price is None:
            self._fundamental_price = float(self.order_book.last_price)

        # still in historical window: just follow it
        if self._history_index < len(self._history_prices):
            new_price = self._history_prices[self._history_index]
            self._history_index += 1
        else:
            # tokenized / projected regime
            if self._scenario is None:
                new_price = self._fundamental_price
            else:
                mu = self._scenario.mu_monthly
                sigma = self.current_sigma_monthly   # regime-adjusted sigma
                dt = 1.0  # 1 month
                z = self._sample_standard_normal()
                log_ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
                new_price = self._fundamental_price * math.exp(log_ret)

        self._fundamental_price = new_price
        self._update_vol_regime(new_price)
        self.target_price = MathX.round_to_int(self._fundamental_price)
        

    def _sample_standard_normal(self) -> float:
        u1 = 1.0 - self.rng.random()
        u2 = 1.0 - self.rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    # ---------- Existing tick/candle processing ----------

    def process_market_tick(self) -> None:
        if not self.active:
            return
        for t in self.traders:
            t.try_place_orders()

    def roll_candle(self) -> None:
        """
        At candle roll, we've finished a "month" in CRE terms.
        Step the fundamental and reset the candle.

        IMPORTANT: we NO LONGER force last_price = target_price here.
        That lets the micro price (last_price) drift away from the
        fundamental (target_price), while traders still anchor their
        orders around target_price via Trader._determine_order.
        """
        if self._scenario is not None or self._history_prices:
            self._step_fundamental_month()
            # DO NOT reset order_book.last_price to target_price here
        self.live_candlestick.reset(self.order_book.last_price)

# ---------- Original quick runtime demo (still works) ----------

def run_simulation_runtime(seconds: float = 5.0, ticks_per_candle: int = 10) -> None:
    sim = MarketSimulator()
    tick_interval = 1.0 / sim.market_speed
    next_tick = time.perf_counter()
    end_time = next_tick + seconds
    ticks_in_candle = 0
    while time.perf_counter() < end_time:
        now = time.perf_counter()
        if now >= next_tick:
            sim.process_market_tick()
            ticks_in_candle += 1
            next_tick += tick_interval
            if ticks_in_candle >= ticks_per_candle:
                ticks_in_candle = 0
                print(f"Candle: O={sim.live_candlestick.open} H={sim.live_candlestick.high} "
                      f"L={sim.live_candlestick.low} C={sim.live_candlestick.close} "
                      f"Vol={sim.live_candlestick.volume}")
                sim.roll_candle()
        else:
            time.sleep(0.001)

# ---------- Example CRE + tokenized demo ----------

def demo_cre_with_tokenization(path_to_csv: str,
                               months_ahead: int = 24,
                               ticks_per_candle: int = 50) -> None:
    """
    Simple demo:
      - load CRE history
      - calibrate μ, σ
      - build a bullish tokenized scenario
      - replay history + simulate 'months_ahead' tokenized months
      - print fundamental vs micro price candles
    """
    history = load_cre_csv(path_to_csv)
    calib = calibrate_from_history(history)

    # Example parameters (AI agent can tweak these):
    r_annual = 0.02
    gamma = 3.0
    w_star = merton_optimal_weight(calib.mu_annual, r_annual, gamma, calib.sigma_annual)
    print(f"Calibrated: mu_annual={calib.mu_annual:.4f}, sigma_annual={calib.sigma_annual:.4f}, "
          f"Merton w*={w_star:.3f}")

    # Baseline and bullish tokenized scenario
    uplift_annual = 0.02  # +2% annual return for tokenization
    mu_tok_annual = calib.mu_annual + uplift_annual
    mu_tok_monthly = mu_tok_annual / 12.0
    tok_sigma_monthly = calib.sigma_monthly * 1.2  # more active trading

    scenario = ScenarioParams(
        name="Tokenized CRE - Bullish",
        mu_monthly=mu_tok_monthly,
        sigma_monthly=tok_sigma_monthly,
    )

    sim = MarketSimulator()
    sim.attach_history_and_scenario(history, scenario)

    total_months = len(history) + months_ahead
    tick_interval = 1.0 / sim.market_speed

    for month_idx in range(total_months):
        ticks_in_candle = 0
        while ticks_in_candle < ticks_per_candle:
            sim.process_market_tick()
            ticks_in_candle += 1
            time.sleep(tick_interval * 0.1)  # slower than real-time for demo

        # For plotting:
        if month_idx < len(history):
            traditional = history[month_idx].price
        else:
            traditional = history[-1].price  # hold flat after history, or change if you want

        tokenized_micro = sim.order_book.last_price
        tokenized_fund = sim.target_price

        print(f"Month {month_idx+1} | Trad={traditional:.2f} | "
              f"Tok_micro={tokenized_micro} | Tok_fund={tokenized_fund} | "
              f"Candle: O={sim.live_candlestick.open} H={sim.live_candlestick.high} "
              f"L={sim.live_candlestick.low} C={sim.live_candlestick.close} "
              f"Vol={sim.live_candlestick.volume}")

        sim.roll_candle()

if __name__ == "__main__":
    # Original micro demo:
    # run_simulation_runtime(5.0, 10)

    # Example CRE + tokenization demo (point this at your CSV):
    # demo_cre_with_tokenization("cre_monthly.csv")
    run_simulation_runtime(5.0, 10)
