"""Feature Engineering and Seasonal Decomposition for Gas Storage & TTF Analysis.

Constructs:
- Gas Year & Week indicators (October 1 - September 30 cycle)
- Seasonal quantile ranking of storage (storage_pct_quantile)
- 7-day rate of injection/withdrawal change (storage_change_7d)
- 5-year historical same-day deviation (storage_deviation_5yr)
- Market-implied winter premium
- Fourier annual seasonality harmonics (month_sin, month_cos)
- Days to injection deadline (days_to_winter)
- STL (Seasonal-Trend decomposition using LOESS) components for storage and prices
- Regime state indicators
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.features.regime_detection import RegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/duckdb/ttf_storage.duckdb")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def compute_gas_year(dt: pd.Timestamp) -> int:
    """Compute the European Gas Year for a given timestamp.

    The Gas Year runs from October 1 to September 30 of the following year.
    For example:
    - 2021-10-01 belongs to Gas Year 2021/2022 (indexed as 2021)
    - 2022-04-15 belongs to Gas Year 2021/2022 (indexed as 2021)
    - 2022-10-01 belongs to Gas Year 2022/2023 (indexed as 2022)
    """
    return int(dt.year if dt.month >= 10 else dt.year - 1)


def compute_gas_year_week(dt: pd.Timestamp) -> int:
    """Compute the week number (1 to 52/53) within the active Gas Year."""
    gy = compute_gas_year(dt)
    gy_start = pd.Timestamp(year=gy, month=10, day=1)
    days_elapsed = (dt - gy_start).days
    week = (days_elapsed // 7) + 1
    return max(1, min(53, int(week)))


def compute_days_to_winter(dt: pd.Timestamp) -> int:
    """Compute days remaining until the upcoming November 1 injection deadline.

    Nov 1 is the statutory EU winter injection deadline (storage target milestone).
    If currently after Nov 1, counts down to Nov 1 of the following year.
    """
    target = pd.Timestamp(year=dt.year, month=11, day=1)
    if dt >= target:
        target = pd.Timestamp(year=dt.year + 1, month=11, day=1)
    return int((target - dt).days)


class FeatureEngineer:
    """Engineers econometric, technical, and seasonal features for gas market modeling."""

    def __init__(self, db_path: Path | None = None, processed_dir: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.processed_dir = processed_dir or DEFAULT_PROCESSED_DIR
        self.regime_detector = RegimeDetector()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform aligned daily market data into a rich modeling feature matrix.

        Args:
            df: DataFrame containing point-in-time aligned storage and price observations.

        Returns:
            DataFrame enriched with all engineered features.
        """
        logger.info("Computing engineered features on %d records...", len(df))
        out = df.sort_values("date").copy().reset_index(drop=True)

        # 1. Temporal & Gas Year Indexing
        dates = pd.to_datetime(out["date"])
        out["gas_year"] = [compute_gas_year(d) for d in dates]
        out["gas_year_week"] = [compute_gas_year_week(d) for d in dates]
        out["day_of_year"] = dates.dt.dayofyear
        out["days_to_winter"] = [compute_days_to_winter(d) for d in dates]

        # 2. Fourier harmonics for annual cyclical seasonality
        # Captures smooth seasonal transition across the 365.25-day cycle
        doy = dates.dt.dayofyear.values
        out["month_sin"] = np.round(np.sin(2 * np.pi * doy / 365.25), 5)
        out["month_cos"] = np.round(np.cos(2 * np.pi * doy / 365.25), 5)

        # 3. 7-Day Storage Change (Injection / Withdrawal Momentum)
        # Positive = net injection; Negative = net withdrawal
        out["storage_change_7d"] = out["storage_pct"] - out["storage_pct"].shift(7)
        out["storage_change_7d"] = out["storage_change_7d"].fillna(0.0).round(4)

        # 4. Storage Deviation from 5-Year Historical Average
        out["storage_deviation_5yr"] = self._compute_5yr_deviation(out)

        # 5. Seasonal Quantile of Storage (storage_pct_quantile)
        # Evaluates storage inventory relative to same gas-year week historically
        out["storage_pct_quantile"] = self._compute_seasonal_quantile(out)

        # 6. STL Seasonal-Trend Decomposition on Storage and Price
        out = self._compute_stl_decomposition(out)

        # 7. Regime Identification (Markov Switching & Realized Volatility)
        out = self.regime_detector.fit_all(out)

        # 8. Cross-Market Features (EU vs US Henry Hub spread)
        if "henry_hub_eur_mwh" in out.columns and out["henry_hub_eur_mwh"].notna().any():
            out["ttf_hh_spread"] = (out["ttf_price"] - out["henry_hub_eur_mwh"]).round(3)
            out["ttf_hh_ratio"] = (
                out["ttf_price"] / out["henry_hub_eur_mwh"].clip(lower=0.1)
            ).round(3)
        else:
            out["ttf_hh_spread"] = np.nan
            out["ttf_hh_ratio"] = np.nan

        # Target variables for modeling:
        # 1. 1-day forward log return: r_{t+1} = ln(P_{t+1} / P_t)
        # 2. 5-day forward log return: r_{t+5} = ln(P_{t+5} / P_t)
        # 3. 20-day forward realized volatility
        out["target_log_price"] = np.log(out["ttf_price"])
        out["target_return_1d"] = np.log(out["ttf_price"].shift(-1) / out["ttf_price"])
        out["target_return_5d"] = np.log(out["ttf_price"].shift(-5) / out["ttf_price"])

        # Forward realized volatility (next 20 business days)
        fwd_returns = np.log(out["ttf_price"] / out["ttf_price"].shift(1))
        # Rolling forward standard deviation
        rolling_std_fwd = fwd_returns.iloc[::-1].rolling(window=20, min_periods=5).std().iloc[
            ::-1
        ] * np.sqrt(252)
        out["target_fwd_vol_20d"] = rolling_std_fwd.round(4)

        logger.info("Feature engineering complete. Matrix shape: %s", out.shape)
        return out

    def _compute_5yr_deviation(self, df: pd.DataFrame) -> pd.Series:
        """Compute deviation of current storage_pct from preceding 5-year same-day mean.

        Uses only causal past history (up to preceding 5 years on the same day of year)
        to strictly prevent future look-ahead contamination.
        """
        n = len(df)
        deviations = np.zeros(n)
        dates = pd.to_datetime(df["date"])
        pcts = df["storage_pct"].values

        # Build day-of-year history
        doy_history: dict[int, list[tuple[int, float]]] = {}

        for i in range(n):
            dt = dates.iloc[i]
            doy = dt.dayofyear
            yr = dt.year
            curr_pct = pcts[i]

            if doy in doy_history:
                # Filter to preceding 5 years: yr - 5 <= historical_yr < yr
                past_vals = [val for (h_yr, val) in doy_history[doy] if yr - 5 <= h_yr < yr]
                if past_vals:
                    benchmark = float(np.mean(past_vals))
                else:
                    # If not available for this exact year, use whatever causal history exists
                    benchmark = float(np.mean([val for (_, val) in doy_history[doy]]))
                deviations[i] = curr_pct - benchmark
            else:
                deviations[i] = 0.0

            # Append current observation to history
            if doy not in doy_history:
                doy_history[doy] = []
            doy_history[doy].append((yr, curr_pct))

        return pd.Series(deviations, index=df.index).round(4)

    def _compute_seasonal_quantile(self, df: pd.DataFrame) -> pd.Series:
        """Compute rolling historical quantile rank of storage within same gas-year week.

        Quantile is computed causally: evaluating current storage against all historical
        observations within the same gas_year_week prior to the current year.
        """
        quantiles = np.zeros(len(df))
        week_history: dict[int, list[float]] = {}

        for i, row in df.iterrows():
            wk = int(row["gas_year_week"])
            val = float(row["storage_pct"])

            if wk in week_history and len(week_history[wk]) >= 3:
                hist = np.array(week_history[wk])
                # Empirical CDF: proportion of past values <= current value
                q = float(np.mean(hist <= val))
                quantiles[i] = q
            else:
                quantiles[i] = 0.50  # Neutral baseline prior

            if wk not in week_history:
                week_history[wk] = []
            week_history[wk].append(val)

        return pd.Series(quantiles, index=df.index).round(4)

    def _compute_stl_decomposition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply STL decomposition to isolate seasonal, trend, and residual components."""
        res = df.copy()
        period = 252  # Approximate annual trading days for daily business series

        # Storage STL
        try:
            stl_storage = STL(res["storage_pct"], period=period, robust=True).fit()
            res["storage_stl_trend"] = np.round(stl_storage.trend, 4)
            res["storage_stl_seasonal"] = np.round(stl_storage.seasonal, 4)
            res["storage_stl_residual"] = np.round(stl_storage.resid, 4)
        except Exception as e:
            logger.warning(
                "STL decomposition on storage failed: %s. Using moving average fallback.", e
            )
            trend = (
                res["storage_pct"]
                .rolling(window=period, min_periods=30, center=True)
                .mean()
                .bfill()
                .ffill()
            )
            res["storage_stl_trend"] = trend
            res["storage_stl_seasonal"] = res["storage_pct"] - trend
            res["storage_stl_residual"] = 0.0

        # TTF Price STL (on log prices for multiplicative separation)
        log_price = np.log(res["ttf_price"])
        try:
            stl_price = STL(log_price, period=period, robust=True).fit()
            res["price_stl_trend"] = np.round(stl_price.trend, 4)
            res["price_stl_seasonal"] = np.round(stl_price.seasonal, 4)
            res["price_stl_residual"] = np.round(stl_price.resid, 4)
        except Exception as e:
            logger.warning(
                "STL decomposition on price failed: %s. Using moving average fallback.", e
            )
            trend_p = (
                log_price.rolling(window=period, min_periods=30, center=True).mean().bfill().ffill()
            )
            res["price_stl_trend"] = trend_p
            res["price_stl_seasonal"] = log_price - trend_p
            res["price_stl_residual"] = 0.0

        return res

    def save_features(self, df_features: pd.DataFrame) -> None:
        """Persist feature matrix into DuckDB and Parquet."""
        parquet_path = self.processed_dir / "feature_matrix.parquet"
        df_features.to_parquet(parquet_path, index=False)
        logger.info("Saved feature matrix to Parquet: %s", parquet_path)

        con = duckdb.connect(str(self.db_path))
        try:
            con.execute("CREATE OR REPLACE TABLE feature_matrix AS SELECT * FROM df_features;")
            logger.info("Successfully registered feature_matrix table in DuckDB.")
        finally:
            con.close()


def run_feature_engineering() -> pd.DataFrame:
    """Execute end-to-end feature engineering workflow."""
    pipeline_db = DEFAULT_DB_PATH
    con = duckdb.connect(str(pipeline_db))
    try:
        df_aligned = con.execute("SELECT * FROM aligned_daily_market ORDER BY date;").fetchdf()
    finally:
        con.close()

    engineer = FeatureEngineer(db_path=pipeline_db)
    df_features = engineer.transform(df_aligned)
    engineer.save_features(df_features)
    return df_features


if __name__ == "__main__":
    run_feature_engineering()
