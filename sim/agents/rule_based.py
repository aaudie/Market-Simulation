"""
Rule-based trading agents for market simulation.

Implements automated traders that provide liquidity and respond to
market conditions with simple behavioral rules.
"""

from __future__ import annotations
import random
from sim.microstructure import OrderBook


class Trader:
    """
    Market-making trader with inventory management.
    
    Places two-sided quotes (bid and ask) around a reference price,
    adjusting for inventory position and market conditions.
    """
    
    def __init__(self, order_book: OrderBook, rng: random.Random) -> None:
        """
        Initialize trader.
        
        Args:
            order_book: Order book to trade in
            rng: Random number generator
        """
        self.order_book = order_book
        self.rng = rng

    def try_place_orders(self) -> None:
        """
        Attempt to place orders (subject to activity rate).
        
        Uses inventory-aware market making strategy with fundamental anchoring.
        Blends fundamental target price with micro price based on anchor weight.
        """
        ob = self.order_book

        # Activity gate (prevents overposting)
        activity = getattr(ob, "trader_activity_rate", 1.0)
        if self.rng.random() >= activity:
            return

        # Get reference price: blend fundamental and micro
        micro_ref = ob.last_price
        if micro_ref is None:
            return
        
        target = ob.market.target_price
        w = getattr(ob.market, "anchor_weight", 1.0)  # 1.0 = fully anchored
        w = max(0.0, min(1.0, w))
        
        if target is None:
            ref = micro_ref
        else:
            # High anchor_weight pulls towards fundamental, low stays near micro
            ref = w * target + (1.0 - w) * micro_ref

        # Get trader parameters
        inv = float(getattr(self, "inventory", 0.0))
        spread_bps = float(getattr(self, "spread_bps", 10.0))
        inv_skew_bps = float(getattr(self, "inv_skew_bps", 5.0))
        max_abs_inv = float(getattr(self, "max_inventory", 10.0))
        base_qty = int(getattr(self, "base_qty", 2))

        # Normalize inventory
        if max_abs_inv > 0:
            inv = max(-max_abs_inv, min(max_abs_inv, inv))
            inv_norm = inv / max_abs_inv
        else:
            inv_norm = 0.0

        # Inventory-skewed mid (short inventory pushes quotes up, long pushes down)
        mid = ref * (1.0 - (inv_skew_bps * 1e-4) * inv_norm)

        # Calculate bid and ask
        half_spread = ref * (spread_bps * 1e-4) * 0.5
        bid = max(1, int(mid - half_spread))
        ask = max(bid + 1, int(mid + half_spread))

        # Size increases with inventory urgency
        urgency = 1.0 + 0.5 * abs(inv_norm)
        qty = max(1, int(base_qty * urgency))

        # Place two-sided quotes
        ob.place_order(bid, qty, True)    # bid
        ob.place_order(ask, qty, False)   # ask

        # Occasionally generate market orders for volume
        if self.rng.random() < 0.10:  # 10% chance
            is_buy = self.rng.random() < 0.5
            mkt_qty = self.rng.randint(1, 5)
            ob.market.place_market_order(mkt_qty, is_buy)
