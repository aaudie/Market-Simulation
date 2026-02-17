"""
CoinGecko provider.

Rate limits: 30 requests/minute, 10,000/month (free tier)
Features: Quotes, OHLCV, Asset profiles, Market data
"""

import os
from datetime import date, datetime, timedelta, timezone

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import CryptoQuote, CryptoOHLCV, CryptoAsset, CryptoMarketData
from .base import BaseCryptoProvider


# Common symbol to CoinGecko ID mapping
SYMBOL_TO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "USDC": "usd-coin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "LTC": "litecoin",
}


class CoinGeckoProvider(BaseCryptoProvider):
    """
    CoinGecko data provider.

    Provides comprehensive cryptocurrency data:
    - Real-time prices and quotes
    - Historical OHLCV data
    - Asset profiles and metadata
    - Market rankings
    """

    provider_name = "coingecko"
    base_url = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        # CoinGecko free tier doesn't require API key
        # Demo tier uses x-cg-demo-api-key header with api.coingecko.com
        # Pro tier uses x-cg-pro-api-key header with pro-api.coingecko.com
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY") or os.getenv("COIN_GECKO_API_KEY")

    def is_configured(self) -> bool:
        # Free tier works without API key
        return True

    def _get_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        # Demo API keys start with "CG-" and use x-cg-demo-api-key
        if self.api_key:
            if self.api_key.startswith("CG-"):
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                headers["x-cg-pro-api-key"] = self.api_key
        return headers

    def _symbol_to_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID."""
        symbol = symbol.upper()
        if symbol in SYMBOL_TO_ID:
            return SYMBOL_TO_ID[symbol]
        # Fallback: lowercase symbol
        return symbol.lower()

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
        coin_id = self._symbol_to_id(symbol)

        data = self.get("/simple/price", params={
            "ids": coin_id,
            "vs_currencies": "usd,btc",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        })

        if coin_id not in data:
            raise ProviderError(self.provider_name, f"Coin not found: {symbol}")

        coin_data = data[coin_id]

        return CryptoQuote(
            symbol=symbol.upper(),
            price_usd=coin_data.get("usd", 0),
            price_btc=coin_data.get("btc"),
            market_cap=coin_data.get("usd_market_cap"),
            volume_24h=coin_data.get("usd_24h_vol"),
            change_24h=coin_data.get("usd_24h_change"),
            timestamp=datetime.now(timezone.utc),
        )

    def get_ohlcv(
        self,
        symbol: str,
        interval: Interval = Interval.DAY_1,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> ResultList[CryptoOHLCV]:
        """Get OHLCV candle data."""
        coin_id = self._symbol_to_id(symbol)

        # CoinGecko OHLC endpoint only accepts specific day values: 1, 7, 14, 30, 90, 180, 365, max
        # Map the requested range to the closest valid value
        end_date = end or date.today()
        if start:
            requested_days = (end_date - start).days
        else:
            requested_days = min(limit, 365)

        valid_days = [1, 7, 14, 30, 90, 180, 365]
        # Find the smallest valid value that covers the requested range
        days = next((d for d in valid_days if d >= requested_days), 365)

        data = self.get(f"/coins/{coin_id}/ohlc", params={
            "vs_currency": "usd",
            "days": days,
        })

        candles = ResultList(provider=self.provider_name)

        for item in data[:limit]:
            # OHLC format: [timestamp_ms, open, high, low, close]
            timestamp = datetime.fromtimestamp(item[0] / 1000)

            candles.append(CryptoOHLCV(
                symbol=symbol.upper(),
                timestamp=timestamp,
                open=item[1],
                high=item[2],
                low=item[3],
                close=item[4],
                volume=0,  # OHLC endpoint doesn't include volume
                interval=interval,
            ))

        return candles

    def get_asset(self, symbol: str) -> CryptoAsset:
        """Get asset profile information."""
        coin_id = self._symbol_to_id(symbol)

        data = self.get(f"/coins/{coin_id}", params={
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
        })

        links = data.get("links", {})

        return CryptoAsset(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            slug=data.get("id"),
            description=data.get("description", {}).get("en"),
            category=data.get("categories", [None])[0] if data.get("categories") else None,
            tags=data.get("categories", []),
            website=links.get("homepage", [None])[0] if links.get("homepage") else None,
            github=links.get("repos_url", {}).get("github", [None])[0] if links.get("repos_url") else None,
            twitter=links.get("twitter_screen_name"),
            reddit=links.get("subreddit_url"),
            blockchain=data.get("asset_platform_id"),
            genesis_date=date.fromisoformat(data["genesis_date"]) if data.get("genesis_date") else None,
            market_cap_rank=data.get("market_cap_rank"),
            coingecko_rank=data.get("coingecko_rank"),
        )

    def get_market_data(self, symbol: str) -> CryptoMarketData:
        """Get comprehensive market data for an asset."""
        coin_id = self._symbol_to_id(symbol)

        data = self.get(f"/coins/{coin_id}", params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
        })

        market_data = data.get("market_data", {})

        return CryptoMarketData(
            symbol=symbol.upper(),
            name=data.get("name"),
            price_usd=market_data.get("current_price", {}).get("usd", 0),
            price_btc=market_data.get("current_price", {}).get("btc"),
            market_cap=market_data.get("market_cap", {}).get("usd"),
            fully_diluted_valuation=market_data.get("fully_diluted_valuation", {}).get("usd"),
            volume_24h=market_data.get("total_volume", {}).get("usd"),
            market_cap_rank=market_data.get("market_cap_rank"),
            change_1h=market_data.get("price_change_percentage_1h_in_currency", {}).get("usd"),
            change_24h=market_data.get("price_change_percentage_24h"),
            change_7d=market_data.get("price_change_percentage_7d"),
            change_30d=market_data.get("price_change_percentage_30d"),
            change_1y=market_data.get("price_change_percentage_1y"),
            ath=market_data.get("ath", {}).get("usd"),
            ath_date=datetime.fromisoformat(
                market_data.get("ath_date", {}).get("usd", "").replace("Z", "+00:00")
            ) if market_data.get("ath_date", {}).get("usd") else None,
            ath_change_percent=market_data.get("ath_change_percentage", {}).get("usd"),
            atl=market_data.get("atl", {}).get("usd"),
            atl_date=datetime.fromisoformat(
                market_data.get("atl_date", {}).get("usd", "").replace("Z", "+00:00")
            ) if market_data.get("atl_date", {}).get("usd") else None,
            atl_change_percent=market_data.get("atl_change_percentage", {}).get("usd"),
            circulating_supply=market_data.get("circulating_supply"),
            total_supply=market_data.get("total_supply"),
            max_supply=market_data.get("max_supply"),
            last_updated=datetime.fromisoformat(
                market_data.get("last_updated", "").replace("Z", "+00:00")
            ) if market_data.get("last_updated") else None,
        )

    def get_top_coins(self, limit: int = 100) -> ResultList[CryptoMarketData]:
        """Get top coins by market cap."""
        data = self.get("/coins/markets", params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d,30d",
        })

        coins = ResultList(provider=self.provider_name)
        for item in data:
            coins.append(CryptoMarketData(
                symbol=item.get("symbol", "").upper(),
                name=item.get("name"),
                price_usd=item.get("current_price", 0),
                market_cap=item.get("market_cap"),
                volume_24h=item.get("total_volume"),
                market_cap_rank=item.get("market_cap_rank"),
                change_1h=item.get("price_change_percentage_1h_in_currency"),
                change_24h=item.get("price_change_percentage_24h"),
                change_7d=item.get("price_change_percentage_7d_in_currency"),
                change_30d=item.get("price_change_percentage_30d_in_currency"),
                ath=item.get("ath"),
                ath_date=datetime.fromisoformat(
                    item.get("ath_date", "").replace("Z", "+00:00")
                ) if item.get("ath_date") else None,
                ath_change_percent=item.get("ath_change_percentage"),
                circulating_supply=item.get("circulating_supply"),
                total_supply=item.get("total_supply"),
                max_supply=item.get("max_supply"),
                last_updated=datetime.fromisoformat(
                    item.get("last_updated", "").replace("Z", "+00:00")
                ) if item.get("last_updated") else None,
            ))

        return coins

    def get_trending(self) -> list[dict]:
        """
        Get trending coins (most searched in last 24 hours).

        Returns:
            List of trending coins with name, symbol, market_cap_rank, and score
        """
        data = self.get("/search/trending")

        trending = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            trending.append({
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "market_cap_rank": coin.get("market_cap_rank"),
                "score": coin.get("score"),
                "coin_id": coin.get("id"),
                "price_btc": coin.get("price_btc"),
            })

        return trending

    def get_global(self) -> dict:
        """
        Get global cryptocurrency market data.

        Returns:
            Dict with total market cap, volume, BTC dominance, active coins, etc.
        """
        data = self.get("/global")
        global_data = data.get("data", {})

        return {
            "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd"),
            "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
            "markets": global_data.get("markets"),
            "market_cap_change_24h_pct": global_data.get("market_cap_change_percentage_24h_usd"),
            "updated_at": datetime.fromtimestamp(global_data.get("updated_at", 0)) if global_data.get("updated_at") else None,
        }
