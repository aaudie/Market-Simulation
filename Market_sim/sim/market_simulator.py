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
        self._markov_traditional_P: Optional[List[List[float]]] = None
        self._markov_tokenized_P: Optional[List[List[float]]] = None

        # ---------- volatility regime state ----------
        self.base_sigma_monthly: float = 0.03       # default; replaced when scenario attached
        self.current_sigma_monthly: float = self.base_sigma_monthly
        self.regime: str = "neutral"

        self._ret_window: List[float] = []          # rolling window of log returns
        self._regime_window: int = 6                # lookback in "months"
        self._last_fundamental_price: Optional[float] = None
        self._last_micro_price: Optional[float] = None

        # ---------- microstructure + agents ----------
        self.active: bool = True
        self.market_speed: float = 10.0
        self.trader_count: int = 1000
        self.trader_activity_rate: float = 0.05  # 5% activity per tick
        self.capital_multiple: float = 40.0
        self.trader_base_qty: int = 2

        self.chart = Chart()
        self.live_candlestick = Candlestick()
        self.order_book: OrderBook = OrderBook(self)

        self.rng = random.Random()
        self.traders: List[Trader] = []
        self._build_traders(initial_price=self.order_book.last_price)
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
        self.use_micro_feedback: bool = False
        self.regime_micro_weight: float = 0.0
        self.fundamental_micro_feedback: float = 0.0

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

        # If both endpoint matrices are configured, interpolate by adoption.
        if (
            self.use_markov_regimes
            and self.regime_chain is not None
            and self._markov_traditional_P is not None
            and self._markov_tokenized_P is not None
        ):
            self.regime_chain.set_transition_matrix(self._interpolate_transition_matrix(self.adoption))

    def _interpolate_transition_matrix(self, alpha: float) -> List[List[float]]:
        """
        Interpolate transition matrix between traditional and tokenized endpoints.
        """
        if self._markov_traditional_P is None or self._markov_tokenized_P is None:
            raise ValueError("Transition matrix endpoints are not configured.")

        a = max(0.0, min(1.0, alpha))
        rows = len(self._markov_traditional_P)
        cols = len(self._markov_traditional_P[0]) if rows > 0 else 0
        out: List[List[float]] = []
        for i in range(rows):
            row = []
            for j in range(cols):
                p_trad = self._markov_traditional_P[i][j]
                p_tok = self._markov_tokenized_P[i][j]
                row.append((1.0 - a) * p_trad + a * p_tok)
            out.append(row)
        return out

    def enable_microstructure_feedback(
        self,
        regime_micro_weight: float = 0.25,
        fundamental_micro_feedback: float = 0.10,
    ) -> None:
        """
        Enable microstructure-to-macro coupling.

        regime_micro_weight: weight in [0, 1] for micro return in realized-vol signal
        fundamental_micro_feedback: strength for nudging projected fundamental by micro return
        """
        self.use_micro_feedback = True
        self.regime_micro_weight = max(0.0, min(1.0, float(regime_micro_weight)))
        self.fundamental_micro_feedback = max(0.0, min(1.0, float(fundamental_micro_feedback)))

    def disable_microstructure_feedback(self) -> None:
        """Disable microstructure-to-macro coupling."""
        self.use_micro_feedback = False
        self.regime_micro_weight = 0.0
        self.fundamental_micro_feedback = 0.0


    # ---------- Volatility regime updater ----------
    def _update_vol_regime(self, new_fundamental_price: float) -> None:
        """
        Realized-vol regime + optional Markov regime (hybrid).
        """
        if self._last_fundamental_price is None:
            self._last_fundamental_price = new_fundamental_price
            return

        r_fund = math.log(new_fundamental_price / self._last_fundamental_price)
        self._last_fundamental_price = new_fundamental_price

        r_used = r_fund
        micro_now = float(self.order_book.last_price) if self.order_book.last_price is not None else None
        if micro_now is not None:
            if self._last_micro_price is None:
                self._last_micro_price = micro_now
            else:
                if self.use_micro_feedback and self.regime_micro_weight > 0.0 and self._last_micro_price > 0:
                    r_micro = math.log(micro_now / self._last_micro_price)
                    w = self.regime_micro_weight
                    r_used = (1.0 - w) * r_fund + w * r_micro
                self._last_micro_price = micro_now

        self._ret_window.append(r_used)
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
        self._last_micro_price = None

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
                self.order_book._clear_books()

                # 3) Reseed the book around the new last_price so micro doesn't snap back
                levels = 10          # depth on each side
                qty_per_level = 5    # liquidity per level
                for i in range(1, levels + 1):
                    self.order_book._make_order(
                        self.target_price - i, qty_per_level, True
                    )
                    self.order_book._make_order(
                        self.target_price + i, qty_per_level, False
                    )

                # 4) Start the candle at the rebased micro price
                self.live_candlestick.reset(self.order_book.last_price)


    def enable_markov_regimes(self, P: List[List[float]], start_state: str = "neutral") -> None:
        """
        Turn on Markov regime dynamics. Call this once after constructing the simulator.
        """
        self.use_markov_regimes = True
        self.regime_chain = RegimeMarkovChain(STATES, P, start_state=start_state, rng=self.rng)
        self._markov_regime = start_state
        self._markov_traditional_P = None
        self._markov_tokenized_P = None

    def enable_adoption_markov_regimes(
        self,
        P_traditional: List[List[float]],
        P_tokenized: List[List[float]],
        start_state: str = "neutral",
    ) -> None:
        """
        Enable Markov regimes using adoption-weighted matrix interpolation.
        P(alpha) = (1 - alpha) * P_traditional + alpha * P_tokenized
        """
        self._markov_traditional_P = [row[:] for row in P_traditional]
        self._markov_tokenized_P = [row[:] for row in P_tokenized]

        P0 = self._interpolate_transition_matrix(self.adoption)
        self.use_markov_regimes = True
        self.regime_chain = RegimeMarkovChain(STATES, P0, start_state=start_state, rng=self.rng)
        self._markov_regime = start_state

    #Optional toggle off
    def disable_markov_regimes(self) -> None:
        self.use_markov_regimes = False
        self.regime_chain = None
        self._markov_regime = "neutral"
        self._markov_traditional_P = None
        self._markov_tokenized_P = None


    # ---------- Fundamental step (monthly) ----------
    def _sample_standard_normal(self) -> float:
        # Box–Muller
        u1 = 1.0 - self.rng.random()
        u2 = 1.0 - self.rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def _build_traders(self, initial_price: int) -> None:
        """
        Build traders with wallet heterogeneity.

        Capital pool:
            C_total = capital_multiple * N * P0 * base_qty
        Then allocated by tiers:
            - retail:       70% agents, 20% capital
            - professional: 25% agents, 35% capital
            - whales:        5% agents, 45% capital
        Each trader's wealth in a tier is sampled lognormally and normalized to tier pool.
        Wealth is split 50/50 cash vs inventory value at initialization.
        """
        n = max(1, int(self.trader_count))
        p0 = max(1, int(initial_price))
        base_qty = max(1, int(self.trader_base_qty))
        c_total = float(self.capital_multiple) * n * p0 * base_qty

        tier_agent_fracs = [0.70, 0.25, 0.05]
        tier_capital_fracs = [0.20, 0.35, 0.45]

        tier_counts = [int(n * f) for f in tier_agent_fracs]
        tier_counts[-1] = n - sum(tier_counts[:-1])

        traders: List[Trader] = []
        trader_id = 0
        lognormal_sigma = 0.50

        for tier_count, cap_frac in zip(tier_counts, tier_capital_fracs):
            if tier_count <= 0:
                continue
            tier_pool = c_total * cap_frac
            raw = [self.rng.lognormvariate(0.0, lognormal_sigma) for _ in range(tier_count)]
            raw_sum = sum(raw) if raw else 1.0
            for w in raw:
                wealth = tier_pool * (w / raw_sum)
                cash = 0.50 * wealth
                inv_units = (0.50 * wealth) / p0
                traders.append(
                    Trader(
                        self.order_book,
                        self.rng,
                        trader_id=trader_id,
                        cash=cash,
                        inventory=inv_units,
                        base_qty=base_qty,
                    )
                )
                trader_id += 1

        self.traders = traders


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
        self.order_book._clear_books()
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
                if self.use_micro_feedback and self.fundamental_micro_feedback > 0.0 and self._last_micro_price and self._last_micro_price > 0:
                    r_micro = math.log(float(self.order_book.last_price) / self._last_micro_price)
                    # Conservative coupling to avoid destabilizing the macro process.
                    log_ret += self.fundamental_micro_feedback * r_micro
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