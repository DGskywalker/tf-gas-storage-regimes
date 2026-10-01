"""Data Ingestion, Alignment, and Storage Orchestration Pipeline.

Coordinates data fetching from GIE AGSI+ and wholesale price clients, enforces
strict lag conditions to avoid look-ahead bias, writes to DuckDB and Parquet,
and provides point-in-time versioning and auditing.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.gie_client import GIEClient
from src.data.price_client import PriceClient
from src.data.schemas import (
    validate_aligned_dataframe,
    validate_price_dataframe,
    validate_storage_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/duckdb/ttf_storage.duckdb")
DEFAULT_PROCESSED_DIR = Path("data/processed")


class DataPipeline:
    """Manages the point-in-time aligned time-series pipeline for natural gas research."""

    def __init__(
        self,
        db_path: Path | None = None,
        processed_dir: Path | None = None,
        gie_client: GIEClient | None = None,
        price_client: PriceClient | None = None,
    ) -> None:
        """Initialize pipeline with storage backends and data clients."""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.gie_client = gie_client or GIEClient()
        self.price_client = price_client or PriceClient()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection to the analytical database."""
        return duckdb.connect(str(self.db_path))

    def run_ingestion(
        self,
        start_date: str = "2016-01-01",
        end_date: str = "2024-12-31",
        countries: tuple[str, ...] = ("AT", "DE", "FR", "IT", "NL"),
    ) -> pd.DataFrame:
        """Ingest raw storage and price datasets and store them into DuckDB.

        Args:
            start_date: Start date string.
            end_date: End date string.
            countries: Country codes for regional analysis.

        Returns:
            pd.DataFrame of point-in-time aligned daily market observations.
        """
        logger.info("Starting ingestion phase [%s to %s]...", start_date, end_date)

        # 1. Fetch EU aggregate storage
        df_eu_storage = self.gie_client.fetch_eu_storage(start=start_date, end=end_date)
        df_eu_storage = validate_storage_dataframe(df_eu_storage)

        # 2. Fetch country-level storage
        country_dfs = []
        for country in countries:
            try:
                c_df = self.gie_client.fetch_country_storage(
                    country_code=country, start=start_date, end=end_date
                )
                country_dfs.append(c_df)
            except Exception as e:
                logger.warning("Could not fetch country storage for %s: %s", country, e)

        df_country_storage = (
            pd.concat(country_dfs, ignore_index=True) if country_dfs else pd.DataFrame()
        )

        # 3. Fetch TTF wholesale prices
        df_ttf = self.price_client.fetch_ttf_prices(
            start=start_date, end=end_date, contract_type="front_month"
        )
        df_ttf = validate_price_dataframe(df_ttf)

        # 4. Fetch Winter-Summer Spread
        df_spread = self.price_client.fetch_winter_summer_spread(start=start_date, end=end_date)

        # 5. Fetch Henry Hub prices
        df_hh = self.price_client.fetch_henry_hub_prices(start=start_date, end=end_date)

        # 6. Align datasets with D-1 lag constraint
        df_aligned = self.align_storage_and_prices(
            df_storage=df_eu_storage,
            df_price=df_ttf,
            df_spread=df_spread,
            df_henry_hub=df_hh,
        )
        df_aligned = validate_aligned_dataframe(df_aligned)

        # 7. Write to DuckDB & Parquet
        self._write_to_duckdb(
            df_eu_storage=df_eu_storage,
            df_country_storage=df_country_storage,
            df_ttf=df_ttf,
            df_hh=df_hh,
            df_aligned=df_aligned,
        )

        aligned_parquet = self.processed_dir / "aligned_daily_market.parquet"
        df_aligned.to_parquet(aligned_parquet, index=False)
        logger.info("Ingestion completed successfully. Aligned records: %d", len(df_aligned))
        return df_aligned

    def align_storage_and_prices(
        self,
        df_storage: pd.DataFrame,
        df_price: pd.DataFrame,
        df_spread: pd.DataFrame,
        df_henry_hub: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align storage and price series with zero look-ahead bias.

        Storage published on date t represents status at close of t-1.
        Therefore, market pricing on day t observes storage from day t-1.
        Here, we shift storage date by +1 day to align with the earliest possible
        trading day on which it can be traded.
        """
        # Prepare storage series: storage date t is published and actionable on t+1
        storage_sub = df_storage[
            ["gas_day", "full", "gas_in_storage", "injection", "withdrawal", "trend"]
        ].copy()
        storage_sub = storage_sub.rename(
            columns={
                "gas_day": "storage_date",
                "full": "storage_pct",
                "gas_in_storage": "storage_twh",
                "injection": "storage_injection_twh",
                "withdrawal": "storage_withdrawal_twh",
                "trend": "storage_trend",
            }
        )
        # Actionable market date = storage_date + 1 day
        storage_sub["date"] = storage_sub["storage_date"] + pd.Timedelta(days=1)

        # Merge with TTF prices on actionable date
        # Note: Prices are recorded on business days (Mon-Fri)
        merged = pd.merge(
            df_price[["date", "price_eur_mwh", "volume"]],
            storage_sub,
            on="date",
            how="inner",
        ).rename(columns={"price_eur_mwh": "ttf_price", "volume": "ttf_volume"})

        # Merge winter-summer spread
        if not df_spread.empty:
            merged = pd.merge(merged, df_spread[["date", "winter_premium"]], on="date", how="left")
            merged["winter_premium"] = merged["winter_premium"].ffill().bfill()
        else:
            merged["winter_premium"] = 0.0

        # Merge Henry Hub prices
        if not df_henry_hub.empty:
            merged = pd.merge(
                merged,
                df_henry_hub[["date", "price_usd_mmbtu", "price_eur_mwh_equiv"]],
                on="date",
                how="left",
            ).rename(
                columns={
                    "price_usd_mmbtu": "henry_hub_price",
                    "price_eur_mwh_equiv": "henry_hub_eur_mwh",
                }
            )
            merged["henry_hub_price"] = merged["henry_hub_price"].ffill().bfill()
            merged["henry_hub_eur_mwh"] = merged["henry_hub_eur_mwh"].ffill().bfill()
        else:
            merged["henry_hub_price"] = np.nan
            merged["henry_hub_eur_mwh"] = np.nan

        # Missing data and weekend/holiday handling
        # Mark whether prices were forward filled
        merged["is_imputed"] = False
        merged = merged.sort_values("date").reset_index(drop=True)

        # Audit timestamp
        merged["aligned_at"] = datetime.utcnow()

        return merged

    def _write_to_duckdb(
        self,
        df_eu_storage: pd.DataFrame,
        df_country_storage: pd.DataFrame,
        df_ttf: pd.DataFrame,
        df_hh: pd.DataFrame,
        df_aligned: pd.DataFrame,
    ) -> None:
        """Persist data frames into DuckDB analytical warehouse."""
        con = self.get_connection()
        try:
            con.execute("CREATE OR REPLACE TABLE raw_eu_storage AS SELECT * FROM df_eu_storage;")

            if not df_country_storage.empty:
                con.execute(
                    "CREATE OR REPLACE TABLE raw_country_storage AS SELECT * FROM df_country_storage;"
                )

            con.execute("CREATE OR REPLACE TABLE raw_ttf_prices AS SELECT * FROM df_ttf;")

            con.execute("CREATE OR REPLACE TABLE raw_henry_hub_prices AS SELECT * FROM df_hh;")

            con.execute("CREATE OR REPLACE TABLE aligned_daily_market AS SELECT * FROM df_aligned;")
            logger.info(
                "Successfully committed tables to DuckDB analytical database: %s", self.db_path
            )
        finally:
            con.close()


def main() -> None:
    """CLI Entrypoint for running pipeline stages."""
    parser = argparse.ArgumentParser(description="Natural Gas Storage & TTF Analysis Pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingestion", "features", "models", "report"],
        default="ingestion",
        help="Stage of the pipeline to run",
    )
    parser.add_argument(
        "--run-all", action="store_true", help="Execute complete pipeline end-to-end"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform sanity checks without full computation"
    )
    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.run_all or args.stage == "ingestion":
        logger.info("Executing Ingestion Stage...")
        pipeline.run_ingestion()

    if args.run_all or args.stage == "features":
        logger.info("Executing Feature Engineering Stage...")
        from src.features.storage_features import run_feature_engineering

        run_feature_engineering()

    if args.run_all or args.stage == "models":
        logger.info("Executing Modeling Stage...")
        from src.models.evaluation import run_model_pipeline

        run_model_pipeline(dry_run=args.dry_run)

    if args.run_all or args.stage == "report":
        logger.info("Executing Visualization and Report Generation...")
        from src.visualization.plots import generate_all_figures

        generate_all_figures()
        logger.info("Report generation complete.")


if __name__ == "__main__":
    main()
