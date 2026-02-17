"""
Messari provider.

Rate limits: 30 requests/minute, 2000 requests/day (free tier)
Features: Asset profiles, comprehensive metrics, market data
"""

import os
from datetime import date, datetime, timedelta

from ...common.base import ResultList, ProviderError
from ...common.types import Interval
from ..models import CryptoQuote, CryptoOHLCV, CryptoAsset, CryptoMarketData
from .base import BaseCryptoProvider


class MessariProvider(BaseCryptoProvider):
    """
    Messari data provider.

    Provides:
    - Comprehensive crypto asset profiles
    - Market metrics and fundamentals
    - Time series data (daily OHLCV)
    - On-chain metrics

    Known for high-quality fundamental data and research.
    """

    provider_name = "messari"
    base_url = "https://data.messari.io/api"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("MESSARI_API_KEY")

    def is_configured(self) -> bool:
        # The old data.messari.io API has been disabled.
        # New API at api.messari.io has different endpoints - needs migration.
        # See: https://docs.messari.io
        return False

    def _get_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-messari-api-key"] = self.api_key
        return headers

    def _check_response(self, data: dict) -> dict:
        """Check API response for errors."""
        if data.get("status", {}).get("error_code"):
            msg = data.get("status", {}).get("error_message", "Unknown error")
            raise ProviderError(self.provider_name, msg)
        return data

    def _symbol_to_slug(self, symbol: str) -> str:
        """Convert symbol to Messari slug (lowercase)."""
        # Common mappings
        slug_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "USDT": "tether",
            "BNB": "binance-coin",
            "SOL": "solana",
            "XRP": "xrp",
            "USDC": "usd-coin",
            "ADA": "cardano",
            "AVAX": "avalanche",
            "DOGE": "dogecoin",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "MATIC": "polygon",
            "UNI": "uniswap",
            "LTC": "litecoin",
        }
        symbol = symbol.upper()
        return slug_map.get(symbol, symbol.lower())

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
        slug = self._symbol_to_slug(symbol)

        data = self.get(f"/v1/assets/{slug}/metrics")
        self._check_response(data)

        asset_data = data.get("data", {})
        market_data = asset_data.get("market_data", {})

        if not market_data:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return CryptoQuote(
            symbol=symbol.upper(),
            name=asset_data.get("name"),
            price_usd=market_data.get("price_usd", 0),
            price_btc=market_data.get("price_btc"),
            market_cap=market_data.get("market_cap"),
            volume_24h=market_data.get("volume_last_24_hours"),
            change_1h=market_data.get("percent_change_usd_last_1_hour"),
            change_24h=market_data.get("percent_change_usd_last_24_hours"),
            high_24h=market_data.get("ohlcv_last_24_hour", {}).get("high"),
            low_24h=market_data.get("ohlcv_last_24_hour", {}).get("low"),
            open_24h=market_data.get("ohlcv_last_24_hour", {}).get("open"),
            circulating_supply=asset_data.get("supply", {}).get("circulating"),
            total_supply=asset_data.get("supply", {}).get("total"),
            max_supply=asset_data.get("supply", {}).get("max"),
            timestamp=datetime.fromisoformat(
                market_data.get("last_trade_at", "").replace("Z", "+00:00")
            ) if market_data.get("last_trade_at") else None,
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

        Note: Messari free tier only supports daily data.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            interval: Candle interval (only DAY_1 supported on free tier)
            start: Start date
            end: End date
            limit: Maximum number of bars to return

        Returns:
            ResultList of CryptoOHLCV candles
        """
        slug = self._symbol_to_slug(symbol)

        # Messari free tier only supports daily data via timeseries
        if interval not in (Interval.DAY_1, Interval.WEEK_1, Interval.MONTH_1):
            raise ProviderError(
                self.provider_name,
                f"Interval {interval.value} not supported. Use daily/weekly/monthly."
            )

        params = {}

        if start:
            params["start"] = start.isoformat()
        else:
            params["start"] = (date.today() - timedelta(days=limit)).isoformat()

        if end:
            params["end"] = end.isoformat()

        # Map interval to Messari format
        if interval == Interval.WEEK_1:
            params["interval"] = "1w"
        elif interval == Interval.MONTH_1:
            params["interval"] = "1m"
        else:
            params["interval"] = "1d"

        data = self.get(f"/v1/assets/{slug}/metrics/price/time-series", params=params)
        self._check_response(data)

        values = data.get("data", {}).get("values", [])
        candles = ResultList(provider=self.provider_name)

        for item in values[-limit:]:
            try:
                # Format: [timestamp, open, high, low, close, volume]
                timestamp = datetime.fromtimestamp(item[0] / 1000) if item[0] > 1e10 else datetime.fromtimestamp(item[0])

                candles.append(CryptoOHLCV(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    open=item[1] or 0,
                    high=item[2] or 0,
                    low=item[3] or 0,
                    close=item[4] or 0,
                    volume=item[5] or 0 if len(item) > 5 else 0,
                    interval=interval,
                ))
            except (ValueError, TypeError, IndexError):
                continue

        return candles

    def get_asset(self, symbol: str) -> CryptoAsset:
        """Get comprehensive asset profile."""
        slug = self._symbol_to_slug(symbol)

        data = self.get(f"/v1/assets/{slug}/profile")
        self._check_response(data)

        asset_data = data.get("data", {})
        profile = asset_data.get("profile", {})
        general = profile.get("general", {})
        economics = profile.get("economics", {})
        tech = profile.get("technology", {})

        # Get links
        overview = general.get("overview", {})
        links = overview.get("official_links", [])

        website = None
        whitepaper = None
        github = None

        for link in links:
            link_name = link.get("name", "").lower()
            url = link.get("link", "")
            if "website" in link_name or "official" in link_name:
                website = url
            elif "whitepaper" in link_name or "paper" in link_name:
                whitepaper = url
            elif "github" in link_name:
                github = url

        return CryptoAsset(
            symbol=symbol.upper(),
            name=asset_data.get("name", ""),
            slug=asset_data.get("slug"),
            description=overview.get("tagline") or overview.get("project_details"),
            category=overview.get("category"),
            sector=overview.get("sector"),
            tags=overview.get("tags") or [],
            website=website,
            whitepaper=whitepaper,
            github=github,
            blockchain=tech.get("overview", {}).get("blockchain_type"),
            consensus_mechanism=economics.get("consensus_and_emission", {}).get("consensus", {}).get("consensus_details"),
            genesis_date=datetime.strptime(
                general.get("launch_details", {}).get("launch_date"), "%Y-%m-%d"
            ).date() if general.get("launch_details", {}).get("launch_date") else None,
        )

    def get_market_data(self, symbol: str) -> CryptoMarketData:
        """Get comprehensive market data with metrics."""
        slug = self._symbol_to_slug(symbol)

        data = self.get(f"/v1/assets/{slug}/metrics")
        self._check_response(data)

        asset_data = data.get("data", {})
        market_data = asset_data.get("market_data", {})
        supply = asset_data.get("supply", {})
        roi = asset_data.get("roi_data", {})
        all_time = asset_data.get("all_time_high", {})

        if not market_data:
            raise ProviderError(self.provider_name, f"No data for {symbol}")

        return CryptoMarketData(
            symbol=symbol.upper(),
            name=asset_data.get("name"),
            price_usd=market_data.get("price_usd", 0),
            price_btc=market_data.get("price_btc"),
            market_cap=market_data.get("market_cap"),
            fully_diluted_valuation=market_data.get("marketcap_dominance_percent"),
            volume_24h=market_data.get("volume_last_24_hours"),
            market_cap_rank=asset_data.get("marketcap", {}).get("rank"),
            change_1h=market_data.get("percent_change_usd_last_1_hour"),
            change_24h=market_data.get("percent_change_usd_last_24_hours"),
            change_7d=roi.get("percent_change_last_1_week"),
            change_30d=roi.get("percent_change_last_1_month"),
            change_1y=roi.get("percent_change_last_1_year"),
            ath=all_time.get("price"),
            ath_date=datetime.fromisoformat(
                all_time.get("at", "").replace("Z", "+00:00")
            ) if all_time.get("at") else None,
            ath_change_percent=all_time.get("percent_down"),
            circulating_supply=supply.get("circulating"),
            total_supply=supply.get("total"),
            max_supply=supply.get("max"),
            last_updated=datetime.fromisoformat(
                market_data.get("last_trade_at", "").replace("Z", "+00:00")
            ) if market_data.get("last_trade_at") else None,
        )

    def get_top_coins(self, limit: int = 100) -> ResultList[CryptoMarketData]:
        """Get top cryptocurrencies by market cap."""
        data = self.get("/v2/assets", params={
            "limit": min(limit, 500),
            "sort": "marketcap",
            "order": "descending",
        })

        self._check_response(data)

        coins = ResultList(provider=self.provider_name)

        for item in data.get("data", []):
            metrics = item.get("metrics", {})
            market_data = metrics.get("market_data", {})

            if not market_data:
                continue

            coins.append(CryptoMarketData(
                symbol=item.get("symbol", "").upper(),
                name=item.get("name"),
                price_usd=market_data.get("price_usd", 0),
                price_btc=market_data.get("price_btc"),
                market_cap=metrics.get("marketcap", {}).get("current_marketcap_usd"),
                volume_24h=market_data.get("volume_last_24_hours"),
                market_cap_rank=metrics.get("marketcap", {}).get("rank"),
                change_1h=market_data.get("percent_change_usd_last_1_hour"),
                change_24h=market_data.get("percent_change_usd_last_24_hours"),
                circulating_supply=metrics.get("supply", {}).get("circulating"),
            ))

        return coins
