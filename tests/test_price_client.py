"""Unit tests for PriceClient and Wholesale Pricing conversions."""

from pathlib import Path

import numpy as np

from src.data.price_client import MMBTU_PER_MWH, PriceClient
from src.data.schemas import validate_price_dataframe


def test_price_client_ttf(tmp_dir: Path) -> None:
    """Test fetching TTF settlement prices."""
    client = PriceClient(cache_dir=tmp_dir)
    df = client.fetch_ttf_prices(start="2022-01-01", end="2022-03-31")

    assert not df.empty
    assert "price_eur_mwh" in df.columns
    assert "date" in df.columns
    assert (df["price_eur_mwh"] > 0.0).all()
    assert (df["price_eur_mwh"] < 500.0).all()

    validated = validate_price_dataframe(df)
    assert len(validated) == len(df)


def test_henry_hub_conversion(tmp_dir: Path) -> None:
    """Test Henry Hub ingestion and EUR/MWh equivalence conversion."""
    client = PriceClient(cache_dir=tmp_dir)
    df = client.fetch_henry_hub_prices(start="2023-01-01", end="2023-03-31")

    assert not df.empty
    assert "price_usd_mmbtu" in df.columns
    assert "price_eur_mwh_equiv" in df.columns

    # Check unit conversion ratio
    # 1 MWh = 3.412142 MMBtu, EUR/USD approx 1.10
    expected_ratio = MMBTU_PER_MWH / 1.10
    actual_ratio = (df["price_eur_mwh_equiv"] / df["price_usd_mmbtu"]).iloc[0]
    assert np.isclose(actual_ratio, expected_ratio, atol=0.01)


def test_winter_summer_spread(tmp_dir: Path) -> None:
    """Test calculation of market-implied seasonal spread."""
    client = PriceClient(cache_dir=tmp_dir)
    df = client.fetch_winter_summer_spread(start="2023-01-01", end="2023-03-31")

    assert not df.empty
    assert "winter_premium" in df.columns
    assert df["winter_premium"].notna().all()
