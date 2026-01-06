#Candlestick, Chart, OrderBook
from __future__ import annotations

from collections import defaultdict
from typing import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.market_simulator import MarketSimulator
    from sim.microstructure import OrderBook


class Candlestick:
    def __init__(self) -> None:
        self.open: int = 0
        self.high: int = 0
        self.low: int = 0
        self.close: int = 0
        self.volume: int = 0

    def reset(self, last_price: int) -> None:
        self.open = last_price
        self.high = last_price
        self.low = last_price
        self.close = last_price
        self.volume = 0

    def update_with_trade(self, trade_price: int, qty: int) -> None:
        self.close = trade_price
        if trade_price > self.high:
            self.high = trade_price
        if trade_price < self.low:
            self.low = trade_price
        self.volume += qty

class Chart:
    def update_live_candle(self, *args, **kwargs) -> None:
        # placeholder; you can replace with real chart recording later
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

    def _make_order(self, price: int, qty: int, is_buy: bool) -> None:
        if qty <= 0:
            return
        if is_buy:
            self.buy_book[price] += qty
        else:
            self.sell_book[price] += qty
        self._refresh_best_prices()

    def place_order(self, limit_price: float, qty: int, is_buy: bool) -> None:
        if qty <= 0:
            return

        remaining: int = qty

        # Match against opposite book
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
                    trade_price,  # <-- executed price
                    self.market.live_candlestick.volume,
                    self.market.live_candlestick.high,
                    self.market.live_candlestick.low,
                )

                remaining -= trade_q
                self._refresh_best_prices()

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
                    trade_price,  # <-- executed price (NOT limit_price)
                    self.market.live_candlestick.volume,
                    self.market.live_candlestick.high,
                    self.market.live_candlestick.low,
                )

                remaining -= trade_q
                self._refresh_best_prices()

        # update last price / imbalance
        if self.last_traded_price > 0:
            self.last_price = int(self.last_traded_price)

        tv = self.total_buy_volume + self.total_sell_volume
        self.imbalance = (self.total_buy_volume - self.total_sell_volume) / tv if tv > 0 else 0.0

        # any remainder becomes a resting order (but NOT for market-style sentinels)
        is_market_style = (is_buy and limit_price == float("inf")) or ((not is_buy) and limit_price == 0.0)
        if remaining > 0 and not is_market_style:
            self._make_order(int(limit_price), remaining, is_buy)


    def ensure_minimum_liquidity(self, last_price: int | None = None) -> None:
        # Add liquidity if one side is empty or thin
        if last_price is None:
            last_price = self.last_price
        min_depth = 20  # minimum units per side
        if not self.buy_book or sum(self.buy_book.values()) < min_depth:
            for i in range(1, 6):  # 5 price levels
                self._make_order(max(1, last_price - i), min_depth // 5, True)
        if not self.sell_book or sum(self.sell_book.values()) < min_depth:
            for i in range(1, 6):  # 5 price levels
                self._make_order(last_price + i, min_depth // 5, False)

    def reset_with_price(self, last_price: int) -> None:
        """Hard reset the micro book around a new price level."""
        self.buy_book.clear()
        self.sell_book.clear()

        self.best_bid = 0
        self.best_ask = 2**31 - 1

        self.total_buy_volume = 0
        self.total_sell_volume = 0
        self.imbalance = 0.0

        self.last_price = int(last_price)
        self.last_traded_price = int(last_price)

        # seed liquidity so trading can start immediately
        self.ensure_minimum_liquidity(self.last_price)

