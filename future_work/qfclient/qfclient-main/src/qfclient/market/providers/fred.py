"""
FRED (Federal Reserve Economic Data) provider.

Rate limits: 120 requests/minute
Features: Economic indicators, interest rates, inflation, employment data
"""

import os
from datetime import date, datetime

from ...common.base import ResultList, ProviderError
from ..models import EconomicIndicator, COMMON_SERIES
from .base import BaseProvider


class FREDProvider(BaseProvider):
    """
    Federal Reserve Economic Data (FRED) provider.

    Provides access to 800,000+ economic time series including:
    - Interest rates (Fed Funds, Treasury yields)
    - Inflation (CPI, PCE)
    - Employment (unemployment rate, payrolls)
    - GDP and output measures
    - Money supply
    - Housing data
    """

    provider_name = "fred"
    base_url = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get(self, url: str, params: dict | None = None, **kwargs) -> dict:
        """Override to add API key and format to params."""
        params = params or {}
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        return super().get(url, params=params, **kwargs)

    @property
    def supports_economic(self) -> bool:
        return True

    def get_series_info(self, series_id: str) -> dict:
        """Get metadata about a series."""
        data = self.get("/series", params={"series_id": series_id})

        seriess = data.get("seriess", [])
        if not seriess:
            raise ProviderError(self.provider_name, f"Series not found: {series_id}")

        return seriess[0]

    def get_economic_indicator(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
        limit: int | None = None,
    ) -> ResultList[EconomicIndicator]:
        """
        Get economic indicator data for a series.

        Args:
            series_id: FRED series ID (e.g., "FEDFUNDS", "GDP", "UNRATE")
            start: Start date for observations
            end: End date for observations
            limit: Maximum number of observations

        Returns:
            ResultList of EconomicIndicator observations
        """
        # Get series metadata
        series_info = self.get_series_info(series_id)
        series_name = series_info.get("title", series_id)
        units = series_info.get("units")
        frequency = series_info.get("frequency")

        # Get observations
        params = {"series_id": series_id}
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()
        if limit:
            params["limit"] = limit
            params["sort_order"] = "desc"  # Get most recent first

        data = self.get("/series/observations", params=params)

        observations = data.get("observations", [])

        # If we limited and sorted desc, reverse to get chronological order
        if limit:
            observations = list(reversed(observations))

        indicators = ResultList(provider=self.provider_name)
        for obs in observations:
            value_str = obs.get("value", ".")
            if value_str == ".":
                continue  # Skip missing values

            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue

            indicators.append(EconomicIndicator(
                series_id=series_id,
                name=series_name,
                value=value,
                observation_date=date.fromisoformat(obs["date"]),
                units=units,
                frequency=frequency,
                source="FRED",
            ))

        return indicators

    def get_latest(self, series_id: str) -> EconomicIndicator:
        """Get the most recent observation for a series."""
        results = self.get_economic_indicator(series_id, limit=1)
        if not results:
            raise ProviderError(self.provider_name, f"No data for series: {series_id}")
        return results[0]

    def search_series(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search for series by keywords.

        Returns list of series metadata dicts with:
        - id: Series ID
        - title: Series name
        - frequency: Data frequency
        - units: Units of measurement
        """
        data = self.get("/series/search", params={
            "search_text": query,
            "limit": limit,
        })

        return [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "frequency": s.get("frequency"),
                "units": s.get("units"),
            }
            for s in data.get("seriess", [])
        ]

    def get_common_indicators(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, ResultList[EconomicIndicator]]:
        """
        Get data for commonly used economic indicators.

        Returns dict mapping series_id to ResultList of observations.
        """
        common_ids = [
            "FEDFUNDS",  # Fed Funds Rate
            "DGS10",     # 10-Year Treasury
            "UNRATE",    # Unemployment Rate
            "CPIAUCSL",  # CPI
        ]

        results = {}
        for series_id in common_ids:
            try:
                results[series_id] = self.get_economic_indicator(
                    series_id, start=start, end=end
                )
            except Exception:
                continue

        return results
