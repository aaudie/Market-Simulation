#Trader strategies
from __future__ import annotations
from typing import Tuple
import random
from sim.microstructure import OrderBook
from sim.utils import MathX


class Trader:
    def __init__(self, order_book: OrderBook, rng: random.Random) -> None:
        self.order_book = order_book
        self.rng = rng

    # def try_place_orders(self) -> None:
        # # --- liquidity taker ---
        # if self.rng.random() < 0.001:
        #     qty = self.rng.randint(1, 2)
        #     is_buy = self.rng.random() < 0.5
        #     self.order_book.market.place_market_order(qty, is_buy)
        #     return


        # # --- existing limit-order logic ---
        # if self.rng.random() < self.order_book.trader_activity_rate:
        #     self._place_orders()
    def try_place_orders(self) -> None:
        ob = self.order_book

        # Activity gate (prevents overposting)
        activity = getattr(ob, "trader_activity_rate", 1.0)
        if self.rng.random() >= activity:
            return

        # Reference price (pure micro)
        ref = ob.last_price
        if ref is None:
            return

        # Inventory (optional, defaults to 0)
        inv = float(getattr(self, "inventory", 0.0))

        # Parameters
        spread_bps   = float(getattr(self, "spread_bps", 10.0))   # total spread
        inv_skew_bps = float(getattr(self, "inv_skew_bps", 5.0))  # inventory pressure
        max_abs_inv  = float(getattr(self, "max_inventory", 10.0))
        base_qty     = int(getattr(self, "base_qty", 2))

        # Normalize inventory
        if max_abs_inv > 0:
            inv = max(-max_abs_inv, min(max_abs_inv, inv))
            inv_norm = inv / max_abs_inv
        else:
            inv_norm = 0.0

        # Inventory-skewed mid
        mid = ref * (1.0 - (inv_skew_bps * 1e-4) * inv_norm)

        # Bid / Ask
        half_spread = ref * (spread_bps * 1e-4) * 0.5
        bid = max(1, int(mid - half_spread))
        ask = max(bid + 1, int(mid + half_spread))

        # Size tilts with inventory
        urgency = 1.0 + 0.5 * abs(inv_norm)
        qty = max(1, int(base_qty * urgency))

        # ---- PLACE ORDERS (THIS IS THE KEY FIX) ----
        ob.place_order(bid, qty, True)    # bid
        ob.place_order(ask, qty, False)   # ask

        # ---- Optional: taker flow to generate volume ----
        if self.rng.random() < 0.10:  # 10% chance of market order
            is_buy = self.rng.random() < 0.5
            ob.market.place_market_order(self.rng.randint(1, 5), is_buy)  # 1-5 units


    def _place_orders(self) -> None:
        p, q, is_buy = self._determine_order()
        self.order_book.place_order(p, q, is_buy)

    def _determine_order(self) -> Tuple[int, int, bool]:

        target = self.order_book.market.target_price          # fund / NAV anchor
        micro_reference = self.order_book.last_price          # micro memory (can upgrade to EMA later)

        w = getattr(self.order_book.market, "anchor_weight", 1.0)  # default fully anchored if missing
        w = max(0.0, min(1.0, w))                                  # clamp just in case

        if target is None:
            base_price = micro_reference
        else:
            base_price = w * target + (1.0 - w) * micro_reference

        # --- VOLATILITY-SCALED ORDER-BOOK WIDTH ---
        sigma = self.order_book.market.current_sigma_monthly or 0.02  # fallback
        base_spread = 0.02  # 2% minimum band
        k = 1.5             # sensitivity to volatility
        spread = base_spread + k * sigma
        spread = min(max(spread, 0.01), 0.20)  # clamp [1%, 20%]

        rand_factor = 1.0 + self.rng.uniform(-spread, spread)
        rand_price = MathX.round_to_int(base_price * rand_factor)

        is_buy = rand_price > self.order_book.last_price
        qty = self.rng.randint(1, 20)
        return rand_price, qty, is_buy
