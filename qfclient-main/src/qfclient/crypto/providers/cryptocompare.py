"""
CryptoCompare provider (formerly CoinDesk Data API).

Rate limits: 50 requests/minute, 100,000/month (free tier)
Features: Quotes, full OHLCV (1m to 1d intervals), Asset profiles
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import CryptoQuote, CryptoOHLCV, CryptoAsset, CryptoMarketData
from .base import BaseCryptoProvider


def _interval_to_cryptocompare(interval: Interval) -> tuple[str, int]:
    """
    Convert interval to CryptoCompare endpoint and aggregate param.

    Returns:
        Tuple of (endpoint_path, aggregate_value)
    """
    mapping = {
        Interval.MINUTE_1: ("/data/v2/histominute", 1),
        Interval.MINUTE_5: ("/data/v2/histominute", 5),
        Interval.MINUTE_15: ("/data/v2/histominute", 15),
        Interval.MINUTE_30: ("/data/v2/histominute", 30),
        Interval.HOUR_1: ("/data/v2/histohour", 1),
        Interval.HOUR_4: ("/data/v2/histohour", 4),
        Interval.DAY_1: ("/data/v2/histoday", 1),
        Interval.WEEK_1: ("/data/v2/histoday", 7),
        Interval.MONTH_1: ("/data/v2/histoday", 30),
    }
    return mapping.get(interval, ("/data/v2/histoday", 1))


class CryptoCompareProvider(BaseCryptoProvider):
    """
    CryptoCompare data provider.

    Provides:
    - Real-time cryptocurrency prices
    - Historical OHLCV data (minute, hour, day intervals)
    - Coin metadata and profiles
    - Trading pair information

    Superior to CoinGecko for intraday OHLCV data.
    """

    provider_name = "cryptocompare"
    base_url = "https://min-api.cryptocompare.com"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        # Support both env var naming conventions
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY") or os.getenv("COIN_DESK_API_KEY")

    def is_configured(self) -> bool:
        # CryptoCompare works without API key but with lower limits
        return True

    def _get_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Apikey {self.api_key}"
        return headers

    def _check_response(self, data: dict) -> dict:
        """Check API response for errors."""
        if data.get("Response") == "Error":
            msg = data.get("Message", "Unknown error")
            raise ProviderError(self.provider_name, msg)
        return data

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_ohlcv(self) -> bool:
        return True

    @property
    def supports_asset(self) -> bool:
        return True

    @property
    def supports_market_data(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> CryptoQuote:
        """Get the latest quote for a cryptocurrency."""
        symbol = symbol.upper()

        data = self.get("/data/pricemultifull", params={
            "fsyms": symbol,
            "tsyms": "USD,BTC",
        })

        self._check_response(data)

        raw_data = data.get("RAW", {}).get(symbol, {})
        usd_data = raw_data.get("USD", {})
        btc_data = raw_data.get("BTC", {})

        if not usd_data:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return CryptoQuote(
            symbol=symbol,
            price_usd=usd_data.get("PRICE", 0),
            price_btc=btc_data.get("PRICE"),
            market_cap=usd_data.get("MKTCAP"),
            volume_24h=usd_data.get("TOTALVOLUME24HTO"),
            change_24h=usd_data.get("CHANGEPCT24HOUR"),
            high_24h=usd_data.get("HIGH24HOUR"),
            low_24h=usd_data.get("LOW24HOUR"),
            open_24h=usd_data.get("OPEN24HOUR"),
            circulating_supply=usd_data.get("CIRCULATINGSUPPLY"),
            total_supply=usd_data.get("SUPPLY"),
            timestamp=datetime.fromtimestamp(usd_data.get("LASTUPDATE", 0)) if usd_data.get("LASTUPDATE") else None,
        )

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[CryptoOHLCV]:
        """
        Get OHLCV candle data for a cryptocurrency.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            interval: Candle interval (1m to 1d supported)
            start: Start date (used to calculate limit if provided)
            end: End date
            limit: Maximum number of bars to return (max 2000)

        Returns:
            ResultList of CryptoOHLCV candles
        """
        symbol = symbol.upper()
        endpoint, aggregate = _interval_to_cryptocompare(interval)

        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": min(limit, 2000),
            "aggregate": aggregate,
        }

        # CryptoCompare uses toTs for end time
        if end:
            params["toTs"] = int(datetime.combine(end, datetime.max.time()).timestamp())

        data = self.get(endpoint, params=params)
        self._check_response(data)

        candles = ResultList(provider=self.provider_name)
        bars = data.get("Data", {}).get("Data", [])

        for bar in bars:
            try:
                timestamp = datetime.fromtimestamp(bar.get("time", 0))

                # Apply start date filter
                if start and timestamp.date() < start:
                    continue

                candles.append(CryptoOHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=bar.get("open", 0),
                    high=bar.get("high", 0),
                    low=bar.get("low", 0),
                    close=bar.get("close", 0),
                    volume=bar.get("volumeto", 0),  # Volume in quote currency (USD)
                    volume_from=bar.get("volumefrom", 0),  # Volume in base currency
                    interval=interval,
                ))
            except (ValueError, TypeError, KeyError):
                continue

        return candles

    def get_asset(self, symbol: str) -> CryptoAsset:
        """Get asset profile and metadata."""
        symbol = symbol.upper()

        # Get coin info
        data = self.get("/data/coin/generalinfo", params={
            "fsyms": symbol,
            "tsym": "USD",
        })

        self._check_response(data)

        coins = data.get("Data", [])
        if not coins:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        coin_data = coins[0].get("CoinInfo", {})

        return CryptoAsset(
            symbol=symbol,
            name=coin_data.get("FullName", ""),
            slug=coin_data.get("Name", "").lower(),
            description=coin_data.get("Description"),
            algorithm=coin_data.get("Algorithm"),
            proof_type=coin_data.get("ProofType"),
            website=coin_data.get("Url"),
        )

    def get_market_data(self, symbol: str) -> CryptoMarketData:
        """Get comprehensive market data for a cryptocurrency."""
        symbol = symbol.upper()

        data = self.get("/data/pricemultifull", params={
            "fsyms": symbol,
            "tsyms": "USD,BTC",
        })

        self._check_response(data)

        raw_data = data.get("RAW", {}).get(symbol, {})
        usd_data = raw_data.get("USD", {})
        display_data = data.get("DISPLAY", {}).get(symbol, {}).get("USD", {})

        if not usd_data:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return CryptoMarketData(
            symbol=symbol,
            name=usd_data.get("FROMSYMBOL"),
            price_usd=usd_data.get("PRICE", 0),
            price_btc=raw_data.get("BTC", {}).get("PRICE"),
            market_cap=usd_data.get("MKTCAP"),
            volume_24h=usd_data.get("TOTALVOLUME24HTO"),
            change_1h=usd_data.get("CHANGEPCTHOUR"),
            change_24h=usd_data.get("CHANGEPCT24HOUR"),
            circulating_supply=usd_data.get("CIRCULATINGSUPPLY"),
            total_supply=usd_data.get("SUPPLY"),
            max_supply=usd_data.get("MAXSUPPLY") if usd_data.get("MAXSUPPLY") != 0 else None,
            last_updated=datetime.fromtimestamp(usd_data.get("LASTUPDATE", 0)) if usd_data.get("LASTUPDATE") else None,
        )

    def get_top_coins(self, limit: int = 100) -> ResultList[CryptoMarketData]:
        """Get top cryptocurrencies by market cap."""
        data = self.get("/data/top/mktcapfull", params={
            "limit": min(limit, 100),
            "tsym": "USD",
        })

        self._check_response(data)

        coins = ResultList(provider=self.provider_name)

        for item in data.get("Data", []):
            coin_info = item.get("CoinInfo", {})
            raw_data = item.get("RAW", {}).get("USD", {})

            if not raw_data:
                continue

            coins.append(CryptoMarketData(
                symbol=coin_info.get("Name", "").upper(),
                name=coin_info.get("FullName"),
                price_usd=raw_data.get("PRICE", 0),
                market_cap=raw_data.get("MKTCAP"),
                volume_24h=raw_data.get("TOTALVOLUME24HTO"),
                market_cap_rank=item.get("CoinInfo", {}).get("SortOrder"),
                change_1h=raw_data.get("CHANGEPCTHOUR"),
                change_24h=raw_data.get("CHANGEPCT24HOUR"),
                circulating_supply=raw_data.get("CIRCULATINGSUPPLY"),
                total_supply=raw_data.get("SUPPLY"),
                last_updated=datetime.fromtimestamp(raw_data.get("LASTUPDATE", 0)) if raw_data.get("LASTUPDATE") else None,
            ))

        return coins
