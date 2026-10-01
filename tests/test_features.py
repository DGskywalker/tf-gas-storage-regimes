"""Unit tests and Hypothesis property-based tests for feature engineering."""

from pathlib import Path

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from src.features.storage_features import (
    FeatureEngineer,
    compute_days_to_winter,
    compute_gas_year,
    compute_gas_year_week,
)


def test_compute_gas_year_boundaries() -> None:
    """Test gas year boundary transitions on Oct 1."""
    # Sept 30, 2021 belongs to GY 2020
    assert compute_gas_year(pd.Timestamp("2021-09-30")) == 2020
    # Oct 1, 2021 belongs to GY 2021
    assert compute_gas_year(pd.Timestamp("2021-10-01")) == 2021
    # Jan 15, 2022 belongs to GY 2021
    assert compute_gas_year(pd.Timestamp("2022-01-15")) == 2021


def test_compute_gas_year_week() -> None:
    """Test gas year week calculations."""
    # Week 1 starts Oct 1
    assert compute_gas_year_week(pd.Timestamp("2021-10-01")) == 1
    # Week 2 starts Oct 8
    assert compute_gas_year_week(pd.Timestamp("2021-10-08")) == 2
    # Late September should be near week 52
    wk = compute_gas_year_week(pd.Timestamp("2022-09-25"))
    assert 51 <= wk <= 53


@given(
    year=st.integers(min_value=2015, max_value=2030),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
)
def test_hypothesis_gas_year_properties(year: int, month: int, day: int) -> None:
    """Property-based test verifying that gas year is always year or year - 1."""
    dt = pd.Timestamp(year=year, month=month, day=day)
    gy = compute_gas_year(dt)
    if month >= 10:
        assert gy == year
    else:
        assert gy == year - 1


def test_compute_days_to_winter() -> None:
    """Test countdown to November 1 injection deadline."""
    # On Nov 1 itself, target rolls to next year (365 days)
    assert compute_days_to_winter(pd.Timestamp("2022-10-31")) == 1
    assert compute_days_to_winter(pd.Timestamp("2022-11-01")) in (365, 366)
    assert compute_days_to_winter(pd.Timestamp("2022-10-01")) == 31


def test_feature_engineer_transform(sample_aligned_df: pd.DataFrame, tmp_dir: Path) -> None:
    """Test complete feature matrix generation."""
    engineer = FeatureEngineer(db_path=tmp_dir / "test.duckdb", processed_dir=tmp_dir)
    features_df = engineer.transform(sample_aligned_df)

    assert not features_df.empty
    expected_cols = [
        "gas_year",
        "gas_year_week",
        "days_to_winter",
        "month_sin",
        "month_cos",
        "storage_change_7d",
        "storage_deviation_5yr",
        "storage_pct_quantile",
        "storage_stl_trend",
        "storage_stl_seasonal",
        "regime_label",
        "prob_crisis_filtered",
        "prob_crisis_smoothed",
        "target_log_price",
        "target_return_5d",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature: {col}"


def test_run_feature_engineering_full() -> None:
    """Test top-level run_feature_engineering function."""
    from src.features.storage_features import run_feature_engineering

    df_feat = run_feature_engineering()
    assert not df_feat.empty
    assert "storage_pct" in df_feat.columns
    assert "regime_label" in df_feat.columns
