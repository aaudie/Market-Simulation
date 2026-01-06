#engine + world state + steppin
from __future__ import annotations

from typing import List, Optional
import random
import math

from sim.microstructure import OrderBook, Candlestick, Chart
from sim.types import HistoricalPoint, ScenarioParams
from sim.agents.rule_based import Trader
from sim.regimes import RegimeMarkovChain, STATES, label_regime_from_realized_vol, combine_markov_and_realized, sigma_from_regime


class BatchManager:
    def evaluate_batch_step(self) -> None:
        pass

class MarketSimulator:
    def __init__(self) -> None:
        # ---------- Markov regime process ----------
        self.use_markov_regimes: bool = False
        self.regime_chain: Optional[RegimeMarkovChain] = None
        self._markov_regime: str = "neutral"   # last Markov state (macro)

        # ---------- volatility regime state ----------
        self.base_sigma_monthly: float = 0.03       # default; replaced when scenario attached
        self.current_sigma_monthly: float = self.base_sigma_monthly
        self.regime: str = "neutral"

        self._ret_window: List[float] = []          # rolling window of log returns
        self._regime_window: int = 6                # lookback in "months"
        self._last_fundamental_price: Optional[float] = None

        # ---------- microstructure + agents ----------
        self.active: bool = True
        self.market_speed: float = 10.0
        self.trader_count: int = 1000
        self.trader_activity_rate: float = 0.05  # 5% activity per tick

        self.chart = Chart()
        self.live_candlestick = Candlestick()
        self.order_book: OrderBook = OrderBook(self)

        self.rng = random.Random()
        self.traders = [Trader(self.order_book, self.rng) for _ in range(self.trader_count)]
        self.batch_manager = BatchManager()

        # ---------- fundamental CRE + scenarios ----------
        self.target_price: Optional[int] = None
        self._fundamental_price: Optional[float] = None
        self._scenario: Optional[ScenarioParams] = None

        self._history_prices: List[float] = []
        self._history_index: int = 0

        # -------------------------------------------------
        self.month_idx = 0
        self.adoption = 0.0
        self.anchor_weight = 1.0

        # knobs
        self.adoption_speed = 0.15      # how fast adoption rises
        self.adoption_midpoint = 24     # month where adoption ~ 0.5
        self.anchor_floor = 0.05        # never fully zero (optional)
        self._micro_scaled_once: bool = False  # scale micro to fundamental only at month 0

        self._micro_initialized: bool = False   #flag

    def _sigmoid(self, x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def _update_adoption(self) -> None:
        # adoption in [0,1]
        x = self.adoption_speed * (self.month_idx - self.adoption_midpoint)
        self.adoption = self._sigmoid(x)

        # anchor_weight falls as adoption rises
        self.anchor_weight = max(self.anchor_floor, 1.0 - self.adoption)

        # expose to traders (wherever they can see it)
        self.order_book.market.anchor_weight = self.anchor_weight
        self.order_book.market.adoption = self.adoption
        self.order_book.market.month_idx = self.month_idx


    # ---------- Volatility regime updater ----------
    def _update_vol_regime(self, new_fundamental_price: float) -> None:
        """
        Realized-vol regime + optional Markov regime (hybrid).
        """
        if self._last_fundamental_price is None:
            self._last_fundamental_price = new_fundamental_price
            return

        r = math.log(new_fundamental_price / self._last_fundamental_price)
        self._last_fundamental_price = new_fundamental_price

        self._ret_window.append(r)
        if len(self._ret_window) > self._regime_window:
            self._ret_window.pop(0)

        if len(self._ret_window) < 2:
            return

        mu = sum(self._ret_window) / len(self._ret_window)
        var = sum((x - mu) ** 2 for x in self._ret_window) / (len(self._ret_window) - 1)
        realized_sigma = math.sqrt(var)

        # Realized-vol inferred regime
        realized_regime = label_regime_from_realized_vol(realized_sigma, self.base_sigma_monthly)

        # Markov step (macro) happens once per month; we do it in _step_fundamental_month
        # Here we only combine states if Markov is enabled.
        if self.use_markov_regimes and self.regime_chain is not None:
            final_regime = combine_markov_and_realized(self._markov_regime, realized_regime)
        else:
            final_regime = realized_regime

        self.regime = final_regime
        self.current_sigma_monthly = sigma_from_regime(self.base_sigma_monthly, final_regime)


    # ---------- Attach history + scenario ----------
    def attach_history_and_scenario(self, history_points: List[HistoricalPoint], scenario: ScenarioParams) -> None:
        self._history_prices = [p.price for p in history_points]
        self._history_index = 0
        self._scenario = scenario

        self.base_sigma_monthly = scenario.sigma_monthly
        self.current_sigma_monthly = self.base_sigma_monthly
        self.regime = "neutral"
        self._ret_window = []
        self._last_fundamental_price = None

        if self._history_prices:
            self._fundamental_price = self._history_prices[0]
            self.target_price = int(round(self._fundamental_price))
            if self._history_prices:
                self._fundamental_price = self._history_prices[0]
                self.target_price = int(round(self._fundamental_price))

                # 1) Rebase micro to the SAME initial scale as fundamental
                self.order_book.last_price = self.target_price
                self.order_book.last_traded_price = self.target_price

                # 2) Clear any stale liquidity seeded at the old default scale (~100)
                self.order_book.buy_book.clear()
                self.order_book.sell_book.clear()

                # 3) Reseed the book around the new last_price so micro doesn't snap back
                levels = 10          # depth on each side
                qty_per_level = 5    # liquidity per level
                for i in range(1, levels + 1):
                    self.order_book.buy_book[self.target_price - i] = qty_per_level
                    self.order_book.sell_book[self.target_price + i] = qty_per_level

                # Ensure best prices are set correctly
                self.order_book._refresh_best_prices()

                # 4) Start the candle at the rebased micro price
                self.live_candlestick.reset(self.order_book.last_price)


    def enable_markov_regimes(self, P: List[List[float]], start_state: str = "neutral") -> None:
        """
        Turn on Markov regime dynamics. Call this once after constructing the simulator.
        """
        self.use_markov_regimes = True
        self.regime_chain = RegimeMarkovChain(STATES, P, start_state=start_state, rng=self.rng)
        self._markov_regime = start_state

    #Optional toggle off
    def disable_markov_regimes(self) -> None:
        self.use_markov_regimes = False
        self.regime_chain = None
        self._markov_regime = "neutral"


    # ---------- Fundamental step (monthly) ----------
    def _sample_standard_normal(self) -> float:
        # Box–Muller
        u1 = 1.0 - self.rng.random()
        u2 = 1.0 - self.rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


    def _scale_micro_to_fundamental_once(self) -> None:
        """Put micro on same price level as the fundamental *only once* (month 0).
        After this, micro is free to drift based on order flow; no ongoing anchoring."""
        if self._micro_scaled_once:
            return
        if self._fundamental_price is None:
            return
        if self.target_price is None:
            # keep target_price consistent with the current fundamental if missing
            self.target_price = int(round(self._fundamental_price))
            # -------------------------------------------------
            # ONE-TIME micro price level alignment (Month 0 ONLY)
            # -------------------------------------------------
            if not self._micro_initialized and self.month_idx == 0:
                p0 = int(self.target_price)

                # move micro to same price level as fund ONCE
                self.order_book.reset_with_price(p0)

                # reset live candle so it uses micro prices, not fund
                self.live_candlestick.reset(self.order_book.last_price)

                self._micro_initialized = True

        # set micro last price to the fundamental level
        self.order_book.last_price = int(self.target_price)
        # wipe any pre-seeded liquidity at the old scale and reseed around the new scale
        self.order_book.buy_book.clear()
        self.order_book.sell_book.clear()
        self.order_book.ensure_minimum_liquidity()
        # reset the live candle to the new micro level so charts don't mix scales
        self.live_candlestick.reset(self.order_book.last_price)
        self._micro_scaled_once = True
    def _step_fundamental_month(self) -> None:
        if self._fundamental_price is None:
            self._fundamental_price = float(self.order_book.last_price)
        if self.month_idx == 0 and not self._micro_scaled_once:
            self._scale_micro_to_fundamental_once()
        

        # Replay history first
        if self._history_index < len(self._history_prices):
            new_price = self._history_prices[self._history_index]
            self._history_index += 1
        else:
            # Then GBM projection
            if self._scenario is None:
                new_price = self._fundamental_price
            else:
                mu = self._scenario.mu_monthly
                sigma = self.current_sigma_monthly
                dt = 1.0
                z = self._sample_standard_normal()
                log_ret = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
                new_price = self._fundamental_price * math.exp(log_ret)

        self._fundamental_price = float(new_price)

        # Advance Markov macro regime once per month (exogenous persistence/shocks)
        if self.use_markov_regimes and self.regime_chain is not None:
            self._markov_regime = self.regime_chain.step()

        # update regime from realized vol
        self._update_vol_regime(self._fundamental_price)

        # update fundamental anchor
        self.target_price = int(round(self._fundamental_price))
        if (not self._micro_initialized) and (self.month_idx == 0) and (self.target_price is not None):
            p0 = int(self.target_price)

            # ONE-TIME: put micro on the same initial level as fundamental
            self.order_book.reset_with_price(p0)

            # also reset the live candle to micro (so candle uses micro series)
            self.live_candlestick.reset(int(self.order_book.last_price))

            self._micro_initialized = True

    # ---------- Microstructure stepping ----------
    def run_micro_ticks(self, ticks: int) -> None:
        # for _ in range(ticks):
        #     if not self.active:
        #         return
        #     for t in self.traders:
        #         t.try_place_orders()
        #     self.batch_manager.evaluate_batch_step()
        for _ in range(ticks):
            self.order_book.ensure_minimum_liquidity(self.order_book.last_price)
            for t in self.traders:
                t.try_place_orders()

    def roll_candle(self) -> None:
        """
        At candle roll, we've finished a "month" in CRE terms.
        Step the fundamental and reset the candle.

        IMPORTANT: we do NOT force last_price = target_price here.
        Micro price can drift away from fundamental; traders anchor around target_price.
        """
        self.month_idx += 1
        self._update_adoption()
        
        if self._scenario is not None or self._history_prices:
            self._step_fundamental_month()
        self.live_candlestick.reset(self.order_book.last_price)

    def place_market_order(self, quantity: int, is_buy: bool) -> None:
        """
        Market order = sweep the opposite book until quantity is filled or book empty.
        Implemented by calling place_order with an extreme limit_price so it can cross
        multiple levels.
        """
        if quantity <= 0:
            return

        if is_buy:
            if not self.order_book.sell_book:
                return
            # allow sweeping all asks
            self.order_book.place_order(float("inf"), quantity, True)
        else:
            if not self.order_book.buy_book:
                return
            # allow sweeping all bids
            self.order_book.place_order(0.0, quantity, False)