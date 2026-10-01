"""Integration tests for DataPipeline and end-to-end execution."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline


def test_pipeline_alignment_and_duckdb(tmp_dir: Path) -> None:
    """Test full ingestion phase and DuckDB table creation in isolated environment."""
    test_db = tmp_dir / "test_pipeline.duckdb"
    processed_dir = tmp_dir / "processed"

    pipeline = DataPipeline(db_path=test_db, processed_dir=processed_dir)
    df_aligned = pipeline.run_ingestion(
        start_date="2023-01-01", end_date="2023-03-31", countries=("DE", "NL")
    )

    assert not df_aligned.empty
    assert (processed_dir / "aligned_daily_market.parquet").exists()

    # Query DuckDB
    con = duckdb.connect(str(test_db))
    try:
        tables = con.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        assert "raw_eu_storage" in table_names
        assert "raw_ttf_prices" in table_names
        assert "aligned_daily_market" in table_names

        count = con.execute("SELECT COUNT(*) FROM aligned_daily_market;").fetchone()[0]
        assert count == len(df_aligned)
    finally:
        con.close()


def test_look_ahead_bias_rejection(
    sample_storage_df: pd.DataFrame, sample_price_df: pd.DataFrame, tmp_dir: Path
) -> None:
    """Test that align_storage_and_prices detects and prevents look-ahead bias."""
    # Intentionally misaligned: storage on date t matching price on date t (0 lag)
    bad_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-05")],
            "storage_date": [pd.Timestamp("2023-01-05")],  # Look-ahead violation!
            "ttf_price": [40.0],
            "storage_pct": [65.0],
            "storage_twh": [700.0],
            "is_imputed": [False],
        }
    )

    from src.data.schemas import validate_aligned_dataframe

    with pytest.raises(ValueError, match="Look-ahead bias detected"):
        validate_aligned_dataframe(bad_df)
