"""
Economic indicator models.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, computed_field


class EconomicIndicator(BaseModel):
    """
    Economic indicator data point.

    Normalized from:
    - FRED: get_series
    """
    series_id: str = Field(..., description="Indicator series ID (e.g., GDP, UNRATE)")
    name: str = Field(..., description="Indicator name")
    value: float = Field(..., description="Indicator value")
    observation_date: date = Field(..., description="Observation date")
    units: Optional[str] = Field(default=None, description="Units of measurement")
    frequency: Optional[str] = Field(default=None, description="Data frequency")
    source: Optional[str] = Field(default=None, description="Data source")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "series_id": "FEDFUNDS",
            "name": "Federal Funds Effective Rate",
            "value": 5.25,
            "observation_date": "2024-01-15",
            "units": "Percent",
            "frequency": "Monthly",
            "source": "FRED"
        }
    })


# Common FRED series IDs for reference
COMMON_SERIES = {
    # Interest rates
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DFF": "Federal Funds Rate (Daily)",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury",

    # Inflation
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "CPILFESL": "Core CPI (Excluding Food and Energy)",
    "PCEPI": "Personal Consumption Expenditures Price Index",
    "PCEPILFE": "Core PCE Price Index",

    # Employment
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Total Nonfarm Payrolls",
    "ICSA": "Initial Jobless Claims",
    "CIVPART": "Labor Force Participation Rate",

    # GDP and output
    "GDP": "Gross Domestic Product",
    "GDPC1": "Real Gross Domestic Product",
    "INDPRO": "Industrial Production Index",

    # Money supply
    "M2SL": "M2 Money Stock",
    "WALCL": "Federal Reserve Total Assets",

    # Housing
    "HOUST": "Housing Starts",
    "CSUSHPINSA": "S&P/Case-Shiller U.S. National Home Price Index",

    # Consumer
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "RSXFS": "Retail Sales (Excluding Food Services)",
}
