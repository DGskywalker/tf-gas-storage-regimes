"""Data schemas and validation contracts using Pydantic and Pandera.

Provides runtime type checking, structural validation, and domain-specific
boundary assertions for storage, price, and aligned analytical time-series.
"""

from datetime import date as dt_date

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Pydantic Row-Level Schemas
# ============================================================================


class GIEStorageRecord(BaseModel):
    """Pydantic model representing a single daily AGSI+ storage observation."""

    model_config = ConfigDict(extra="ignore")

    gas_day: dt_date = Field(..., description="Gas observation date (YYYY-MM-DD)")
    country_code: str = Field(..., min_length=2, max_length=5, description="ISO-2 country or 'EU'")
    gas_in_storage: float = Field(..., ge=0.0, description="Volume of working gas in storage (TWh)")
    full: float = Field(..., ge=0.0, le=105.0, description="Storage fill percentage (0 - 100%)")
    trend: float | None = Field(None, description="Daily change in fill level percentage points")
    injection: float | None = Field(
        None, ge=0.0, description="Daily injection volume (GWh/d or TWh/d)"
    )
    withdrawal: float | None = Field(
        None, ge=0.0, description="Daily withdrawal volume (GWh/d or TWh/d)"
    )
    working_gas_volume: float | None = Field(
        None, ge=0.0, description="Total technical working gas capacity (TWh)"
    )
    status: str | None = Field(
        None, description="Data status: C (confirmed), E (estimated), N (no data)"
    )


class TTFPriceRecord(BaseModel):
    """Pydantic model representing a daily Dutch TTF wholesale gas price."""

    model_config = ConfigDict(extra="ignore")

    date: dt_date = Field(..., description="Trading date (YYYY-MM-DD)")
    price_eur_mwh: float = Field(..., ge=0.0, description="Settlement price in EUR/MWh")
    contract_type: str = Field(
        default="front_month", description="Contract: 'day_ahead', 'front_month', 'winter_season'"
    )
    volume: float | None = Field(None, ge=0.0, description="Traded volume in MWh or contracts")


class HenryHubPriceRecord(BaseModel):
    """Pydantic model representing a daily EIA Henry Hub natural gas spot price."""

    model_config = ConfigDict(extra="ignore")

    date: dt_date = Field(..., description="Pricing date (YYYY-MM-DD)")
    price_usd_mmbtu: float = Field(..., ge=0.0, description="Spot price in USD/MMBtu")


class AlignedDataRecord(BaseModel):
    """Pydantic model representing a point-in-time aligned storage and price observation."""

    model_config = ConfigDict(extra="ignore")

    date: dt_date = Field(..., description="Market execution date t (TTF price date)")
    storage_date: dt_date = Field(
        ..., description="Storage observation date t-1 (preventing look-ahead bias)"
    )
    ttf_price: float = Field(..., ge=0.0, description="TTF Front-Month Price (EUR/MWh)")
    storage_pct: float = Field(
        ..., ge=0.0, le=105.0, description="EU aggregate fill percentage (0 - 100%)"
    )
    storage_twh: float = Field(..., ge=0.0, description="EU aggregate stored volume (TWh)")
    henry_hub_price: float | None = Field(
        None, ge=0.0, description="Henry Hub Spot Price (USD/MMBtu)"
    )
    is_imputed: bool = Field(
        default=False, description="Flag indicating holiday/weekend forward-filled price"
    )


# ============================================================================
# Pandera DataFrame-Level Schemas
# ============================================================================

storage_schema = DataFrameSchema(
    {
        "gas_day": Column(
            pa.DateTime,
            checks=[
                Check(
                    lambda s: s.is_monotonic_increasing,
                    error="gas_day must be monotonically increasing",
                ),
            ],
            nullable=False,
            description="Gas observation date",
        ),
        "country_code": Column(pa.String, nullable=False),
        "gas_in_storage": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0.0)],
            nullable=False,
        ),
        "full": Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(
                    105.0
                ),  # Some reporting accounts for uprated capacities >100%
            ],
            nullable=False,
        ),
        "trend": Column(pa.Float, nullable=True),
        "injection": Column(pa.Float, checks=[Check.greater_than_or_equal_to(0.0)], nullable=True),
        "withdrawal": Column(pa.Float, checks=[Check.greater_than_or_equal_to(0.0)], nullable=True),
        "working_gas_volume": Column(
            pa.Float, checks=[Check.greater_than_or_equal_to(0.0)], nullable=True
        ),
    },
    coerce=True,
    strict=False,
)

price_schema = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            checks=[
                Check(
                    lambda s: s.is_monotonic_increasing,
                    error="date must be monotonically increasing",
                ),
            ],
            nullable=False,
        ),
        "price_eur_mwh": Column(
            pa.Float,
            checks=[Check.greater_than(0.0), Check.less_than(500.0)],
            nullable=False,
        ),
        "contract_type": Column(pa.String, nullable=True),
    },
    coerce=True,
    strict=False,
)

aligned_schema = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "storage_date": Column(pa.DateTime, nullable=False),
        "ttf_price": Column(pa.Float, checks=[Check.greater_than(0.0)], nullable=False),
        "storage_pct": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0.0), Check.less_than_or_equal_to(105.0)],
            nullable=False,
        ),
        "storage_twh": Column(
            pa.Float, checks=[Check.greater_than_or_equal_to(0.0)], nullable=False
        ),
        "is_imputed": Column(pa.Bool, nullable=False),
    },
    coerce=True,
    strict=False,
)


def validate_storage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a storage DataFrame against the pandera storage_schema."""
    validated = storage_schema.validate(df)
    return validated


def validate_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a price DataFrame against the pandera price_schema."""
    validated = price_schema.validate(df)
    return validated


def validate_aligned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an aligned analysis DataFrame against the pandera aligned_schema."""
    # Ensure lag condition holds: date >= storage_date + 1 day
    if not df.empty:
        delta = (df["date"] - df["storage_date"]).dt.total_seconds() / 86400.0
        if (delta < 0.99).any():
            raise ValueError(
                "Look-ahead bias detected! storage_date must precede date by at least 1 day."
            )
    validated = aligned_schema.validate(df)
    return validated
