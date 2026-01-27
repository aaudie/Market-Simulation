"""
Market microstructure: order book, candlesticks, and trade execution.

Implements order matching, price formation, and market data structures.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from sim.market_simulator import MarketSimulator


class Candlestick:
    """
    OHLCV candlestick for aggregating trades.
    
    Tracks open, high, low, close prices and volume for a time period.
    """
    
    def __init__(self) -> None:
        """Initialize empty candlestick."""
        self.open: int = 0
        self.high: int = 0
        self.low: int = 0
        self.close: int = 0
        self.volume: int = 0

    def reset(self, last_price: int) -> None:
        """
        Reset candlestick to start a new period.
        
        Args:
            last_price: Opening price for new candle
        """
        self.open = last_price
        self.high = last_price
        self.low = last_price
        self.close = last_price
        self.volume = 0

    def update_with_trade(self, trade_price: int, qty: int) -> None:
        """
        Update candlestick with a new trade.
        
        Args:
            trade_price: Trade execution price
            qty: Trade quantity
        """
        self.close = trade_price
        if trade_price > self.high:
            self.high = trade_price
        if trade_price < self.low:
            self.low = trade_price
        self.volume += qty


class Chart:
    """
    Chart data aggregator (placeholder for future expansion).
    
    Can be extended to record full trade history and charts.
    """
    
    def update_live_candle(self, *args, **kwargs) -> None:
        """Update live candle data (placeholder for future implementation)."""
        pass

class OrderBook:
    """
    Limit order book with price-time priority matching.
    
    Maintains buy and sell order books, executes trades by matching orders,
    and tracks market statistics like volume and imbalance.
    """
    
    def __init__(self, market: "MarketSimulator") -> None:
        """
        Initialize order book.
        
        Args:
            market: Reference to parent market simulator
        """
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
        """Update best bid and ask prices from order books."""
        self.best_bid = max(self.buy_book.keys()) if self.buy_book else 0
        self.best_ask = min(self.sell_book.keys()) if self.sell_book else 2**31 - 1

    def _make_order(self, price: int, qty: int, is_buy: bool) -> None:
        """
        Add resting limit order to book.
        
        Args:
            price: Limit price
            qty: Order quantity
            is_buy: True for buy order, False for sell order
        """
        if qty <= 0:
            return
        if is_buy:
            self.buy_book[price] += qty
        else:
            self.sell_book[price] += qty
        self._refresh_best_prices()

    def place_order(self, limit_price: float, qty: int, is_buy: bool) -> None:
        """
        Place limit order with immediate matching.
        
        Uses price-time priority. Marketable orders execute immediately against
        opposite book. Unfilled quantity rests in the book as limit order.
        
        Args:
            limit_price: Maximum price to pay (buy) or minimum to accept (sell)
            qty: Order quantity
            is_buy: True for buy order, False for sell order
        """
        if qty <= 0:
            return

        remaining: int = qty

        # Match buy order against sell book
        if is_buy:
            while remaining > 0 and self.sell_book and self.best_ask <= limit_price:
                trade_price = self.best_ask
                avail = self.sell_book[trade_price]
                trade_q = min(remaining, avail)

                self.sell_book[trade_price] -= trade_q
                if self.sell_book[trade_price] <= 0:
                    del self.sell_book[trade_price]

                self.total_buy_volume += trade_q
                self.last_traded_price = trade_price

                self.market.live_candlestick.update_with_trade(trade_price, trade_q)
                self.market.chart.update_live_candle(
                    trade_price,
                    self.market.live_candlestick.volume,
                    self.market.live_candlestick.high,
                    self.market.live_candlestick.low,
                )

                remaining -= trade_q
                self._refresh_best_prices()

        # Match sell order against buy book
        else:
            while remaining > 0 and self.buy_book and self.best_bid >= limit_price:
                trade_price = self.best_bid
                avail = self.buy_book[trade_price]
                trade_q = min(remaining, avail)

                self.buy_book[trade_price] -= trade_q
                if self.buy_book[trade_price] <= 0:
                    del self.buy_book[trade_price]

                self.total_sell_volume += trade_q
                self.last_traded_price = trade_price

                self.market.live_candlestick.update_with_trade(trade_price, trade_q)
                self.market.chart.update_live_candle(
                    trade_price,
                    self.market.live_candlestick.volume,
                    self.market.live_candlestick.high,
                    self.market.live_candlestick.low,
                )

                remaining -= trade_q
                self._refresh_best_prices()

        # Update last price and imbalance
        if self.last_traded_price > 0:
            self.last_price = int(self.last_traded_price)

        tv = self.total_buy_volume + self.total_sell_volume
        self.imbalance = (
            (self.total_buy_volume - self.total_sell_volume) / tv 
            if tv > 0 else 0.0
        )

        # Unfilled quantity rests in book (unless market-style order)
        is_market_style = (
            (is_buy and limit_price == float("inf")) or 
            (not is_buy and limit_price == 0.0)
        )
        if remaining > 0 and not is_market_style:
            self._make_order(int(limit_price), remaining, is_buy)


    def ensure_minimum_liquidity(self, last_price: int | None = None) -> None:
        """
        Add liquidity to order book if it's empty or thin.
        
        Prevents book from becoming empty by seeding liquidity around
        the current price across multiple levels.
        
        Args:
            last_price: Reference price for liquidity seeding (uses last_price if None)
        """
        if last_price is None:
            last_price = self.last_price
        
        min_depth = 20  # Minimum total units per side
        levels = 5  # Number of price levels to seed
        qty_per_level = min_depth // levels
        
        # Seed buy side if thin
        if not self.buy_book or sum(self.buy_book.values()) < min_depth:
            for i in range(1, levels + 1):
                self._make_order(max(1, last_price - i), qty_per_level, True)
        
        # Seed sell side if thin
        if not self.sell_book or sum(self.sell_book.values()) < min_depth:
            for i in range(1, levels + 1):
                self._make_order(last_price + i, qty_per_level, False)

    def reset_with_price(self, last_price: int) -> None:
        """
        Hard reset order book around a new price level.
        
        Clears all orders and statistics, then reseeds liquidity.
        Used when attaching historical data or changing price scale.
        
        Args:
            last_price: New reference price
        """
        self.buy_book.clear()
        self.sell_book.clear()

        self.best_bid = 0
        self.best_ask = 2**31 - 1

        self.total_buy_volume = 0
        self.total_sell_volume = 0
        self.imbalance = 0.0

        self.last_price = int(last_price)
        self.last_traded_price = int(last_price)

        # Seed liquidity so trading can start immediately
        self.ensure_minimum_liquidity(self.last_price)

