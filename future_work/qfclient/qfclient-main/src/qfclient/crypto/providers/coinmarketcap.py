"""
CoinMarketCap provider.

Rate limits: 30 requests/minute, 333/day (free tier)
Features: Quotes, Asset metadata, Market rankings
"""

import os
from datetime import datetime

from ...common.base import ResultList, ProviderError
from ..models import CryptoQuote, CryptoAsset, CryptoMarketData
from .base import BaseCryptoProvider


class CoinMarketCapProvider(BaseCryptoProvider):
    """
    CoinMarketCap data provider.

    Provides:
    - Real-time cryptocurrency prices
    - Asset metadata and descriptions
    - Market rankings and metrics
    - Global market statistics
    """

    provider_name = "coinmarketcap"
    base_url = "https://pro-api.coinmarketcap.com/v1"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("COINMARKETCAP_API_KEY") or os.getenv("COIN_MARKET_CAP_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_headers(self) -> dict[str, str]:
        return {
            "X-CMC_PRO_API_KEY": self.api_key or "",
            "Accept": "application/json",
        }

    @property
    def supports_quotes(self) -> bool:
        return True

    @property
    def supports_asset(self) -> bool:
        return True

    @property
    def supports_market_data(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> CryptoQuote:
        """Get the latest quote for a cryptocurrency."""
        data = self.get("/cryptocurrency/quotes/latest", params={
            "symbol": symbol.upper(),
        })

        crypto_data = data.get("data", {}).get(symbol.upper())
        if not crypto_data:
            raise ProviderError(self.provider_name, f"Coin not found: {symbol}")

        quote = crypto_data.get("quote", {}).get("USD", {})

        return CryptoQuote(
            symbol=symbol.upper(),
            name=crypto_data.get("name"),
            price_usd=quote.get("price", 0),
            market_cap=quote.get("market_cap"),
            volume_24h=quote.get("volume_24h"),
            change_1h=quote.get("percent_change_1h"),
            change_24h=quote.get("percent_change_24h"),
            change_7d=quote.get("percent_change_7d"),
            change_30d=quote.get("percent_change_30d"),
            circulating_supply=crypto_data.get("circulating_supply"),
            total_supply=crypto_data.get("total_supply"),
            max_supply=crypto_data.get("max_supply"),
            market_cap_rank=crypto_data.get("cmc_rank"),
            timestamp=datetime.fromisoformat(
                quote.get("last_updated", "").replace("Z", "+00:00")
            ) if quote.get("last_updated") else None,
        )

    def get_asset(self, symbol: str) -> CryptoAsset:
        """Get asset metadata and profile."""
        data = self.get("/cryptocurrency/info", params={
            "symbol": symbol.upper(),
        })

        crypto_data = data.get("data", {}).get(symbol.upper())
        if not crypto_data:
            raise ProviderError(self.provider_name, f"Coin not found: {symbol}")

        urls = crypto_data.get("urls", {})

        return CryptoAsset(
            symbol=symbol.upper(),
            name=crypto_data.get("name", ""),
            slug=crypto_data.get("slug"),
            description=crypto_data.get("description"),
            category=crypto_data.get("category"),
            tags=[tag.get("name") for tag in crypto_data.get("tags", []) if isinstance(tag, dict)],
            website=urls.get("website", [None])[0] if urls.get("website") else None,
            whitepaper=urls.get("technical_doc", [None])[0] if urls.get("technical_doc") else None,
            github=urls.get("source_code", [None])[0] if urls.get("source_code") else None,
            twitter=urls.get("twitter", [None])[0] if urls.get("twitter") else None,
            reddit=urls.get("reddit", [None])[0] if urls.get("reddit") else None,
            blockchain=crypto_data.get("platform", {}).get("name") if crypto_data.get("platform") else None,
        )

    def get_market_data(self, symbol: str) -> CryptoMarketData:
        """Get comprehensive market data."""
        data = self.get("/cryptocurrency/quotes/latest", params={
            "symbol": symbol.upper(),
        })

        crypto_data = data.get("data", {}).get(symbol.upper())
        if not crypto_data:
            raise ProviderError(self.provider_name, f"Coin not found: {symbol}")

        quote = crypto_data.get("quote", {}).get("USD", {})

        return CryptoMarketData(
            symbol=symbol.upper(),
            name=crypto_data.get("name"),
            price_usd=quote.get("price", 0),
            market_cap=quote.get("market_cap"),
            fully_diluted_valuation=quote.get("fully_diluted_market_cap"),
            volume_24h=quote.get("volume_24h"),
            market_cap_rank=crypto_data.get("cmc_rank"),
            change_1h=quote.get("percent_change_1h"),
            change_24h=quote.get("percent_change_24h"),
            change_7d=quote.get("percent_change_7d"),
            change_30d=quote.get("percent_change_30d"),
            circulating_supply=crypto_data.get("circulating_supply"),
            total_supply=crypto_data.get("total_supply"),
            max_supply=crypto_data.get("max_supply"),
            last_updated=datetime.fromisoformat(
                quote.get("last_updated", "").replace("Z", "+00:00")
            ) if quote.get("last_updated") else None,
        )

    def get_top_coins(self, limit: int = 100) -> ResultList[CryptoMarketData]:
        """Get top coins by market cap."""
        data = self.get("/cryptocurrency/listings/latest", params={
            "limit": limit,
            "sort": "market_cap",
            "sort_dir": "desc",
        })

        coins = ResultList(provider=self.provider_name)
        for item in data.get("data", []):
            quote = item.get("quote", {}).get("USD", {})

            coins.append(CryptoMarketData(
                symbol=item.get("symbol", "").upper(),
                name=item.get("name"),
                price_usd=quote.get("price", 0),
                market_cap=quote.get("market_cap"),
                fully_diluted_valuation=quote.get("fully_diluted_market_cap"),
                volume_24h=quote.get("volume_24h"),
                market_cap_rank=item.get("cmc_rank"),
                change_1h=quote.get("percent_change_1h"),
                change_24h=quote.get("percent_change_24h"),
                change_7d=quote.get("percent_change_7d"),
                change_30d=quote.get("percent_change_30d"),
                circulating_supply=item.get("circulating_supply"),
                total_supply=item.get("total_supply"),
                max_supply=item.get("max_supply"),
                last_updated=datetime.fromisoformat(
                    quote.get("last_updated", "").replace("Z", "+00:00")
                ) if quote.get("last_updated") else None,
            ))

        return coins

    def get_global_metrics(self) -> dict:
        """Get global cryptocurrency market metrics."""
        data = self.get("/global-metrics/quotes/latest")

        metrics = data.get("data", {})
        quote = metrics.get("quote", {}).get("USD", {})

        return {
            "total_market_cap": quote.get("total_market_cap"),
            "total_volume_24h": quote.get("total_volume_24h"),
            "btc_dominance": metrics.get("btc_dominance"),
            "eth_dominance": metrics.get("eth_dominance"),
            "active_cryptocurrencies": metrics.get("active_cryptocurrencies"),
            "active_exchanges": metrics.get("active_exchanges"),
            "last_updated": metrics.get("last_updated"),
        }
