"""Pytest configuration and shared test fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for isolated test file creation."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_storage_df() -> pd.DataFrame:
    """Generate a small valid sample storage DataFrame."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "gas_day": dates,
            "country_code": "EU",
            "gas_in_storage": np.linspace(600.0, 750.0, 60),
            "full": np.linspace(55.0, 68.0, 60),
            "trend": np.full(60, 0.2),
            "injection": np.full(60, 2.5),
            "withdrawal": np.full(60, 0.0),
            "working_gas_volume": 1110.0,
        }
    )


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Generate a small valid sample TTF price DataFrame."""
    dates = pd.date_range("2023-01-02", periods=60, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "price_eur_mwh": np.linspace(35.0, 42.0, 60),
            "contract_type": "front_month",
            "volume": np.random.uniform(50000, 150000, 60),
        }
    )


@pytest.fixture
def sample_aligned_df(
    sample_storage_df: pd.DataFrame, sample_price_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate aligned dataset with proper D-1 lag."""
    # Market date t observes storage on date t-1
    storage_sub = sample_storage_df[["gas_day", "full", "gas_in_storage"]].copy()
    storage_sub["date"] = storage_sub["gas_day"] + pd.Timedelta(days=1)
    storage_sub = storage_sub.rename(
        columns={"gas_day": "storage_date", "full": "storage_pct", "gas_in_storage": "storage_twh"}
    )

    merged = pd.merge(sample_price_df, storage_sub, on="date", how="inner").rename(
        columns={"price_eur_mwh": "ttf_price"}
    )
    merged["is_imputed"] = False
    merged["winter_premium"] = 2.5
    merged["henry_hub_price"] = 2.6
    merged["henry_hub_eur_mwh"] = 8.5
    return merged
