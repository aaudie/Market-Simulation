"""Tests for qfclient models."""

from datetime import datetime

import pytest

from qfclient import Quote, OHLCV, Interval
from qfclient.common.base import ResultList


class TestQuoteModel:
    """Tests for Quote model."""

    def test_quote_creation(self):
        """Quote should be created with required fields."""
        quote = Quote(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
        )
        assert quote.symbol == "AAPL"
        assert quote.price == 150.0

    def test_quote_optional_fields(self):
        """Quote should handle optional fields."""
        quote = Quote(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05,
            volume=1000000,
        )
        assert quote.bid == 149.95
        assert quote.ask == 150.05
        assert quote.volume == 1000000


class TestOHLCVModel:
    """Tests for OHLCV model."""

    def test_ohlcv_creation(self):
        """OHLCV should be created with required fields."""
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
        )
        assert ohlcv.symbol == "AAPL"
        assert ohlcv.open == 150.0
        assert ohlcv.close == 154.0

    def test_ohlcv_with_interval(self):
        """OHLCV should accept interval."""
        ohlcv = OHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            interval=Interval.DAY_1,
        )
        assert ohlcv.interval == Interval.DAY_1


class TestResultList:
    """Tests for ResultList wrapper."""

    def test_result_list_is_list(self):
        """ResultList should behave like a list."""
        quotes = ResultList([
            Quote(symbol="AAPL", price=150.0, timestamp=datetime.now()),
            Quote(symbol="MSFT", price=300.0, timestamp=datetime.now()),
        ])
        assert len(quotes) == 2
        assert quotes[0].symbol == "AAPL"

    def test_result_list_to_df(self):
        """ResultList should convert to DataFrame."""
        pytest.importorskip("pandas")

        quotes = ResultList([
            Quote(symbol="AAPL", price=150.0, timestamp=datetime.now()),
            Quote(symbol="MSFT", price=300.0, timestamp=datetime.now()),
        ])
        df = quotes.to_df()

        assert len(df) == 2
        assert "symbol" in df.columns
        assert "price" in df.columns


class TestInterval:
    """Tests for Interval enum."""

    def test_interval_values(self):
        """Interval should have expected values."""
        assert Interval.DAY_1.value == "1d"
        assert Interval.HOUR_1.value == "1h"
        assert Interval.MINUTE_1.value == "1m"

    def test_interval_from_string(self):
        """Interval should be created from string."""
        interval = Interval("1d")
        assert interval == Interval.DAY_1
