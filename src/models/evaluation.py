"""Evaluation Framework, Baselines, and Cross-Validation.

Provides:
- Time-series expanding window cross-validation
- ARIMA(1,1,1) baseline on prices
- GARCH(1,1) baseline for volatility forecasting
- Comprehensive evaluation metrics: RMSE, MAE, Directional Accuracy
- Mincer-Zarnowitz forecast efficiency regression (unbiasedness & explanatory power)
- Diebold-Mariano test for predictive superiority
- Unified model pipeline runner storing metrics to DuckDB
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from src.models.bayesian_ts import run_bayesian_model
from src.models.gbm_model import GBMModel, run_gbm_model

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/duckdb/ttf_storage.duckdb")


class TimeSeriesCV:
    """Expanding-window forward-chaining time-series cross-validator."""

    def __init__(
        self,
        min_train_size: int = 500,
        test_size: int = 125,
        step_size: int = 125,
    ) -> None:
        """Initialize expanding window splitter.

        Args:
            min_train_size: Minimum observations for initial training set (~2 years).
            test_size: Forecast horizon test set size (~6 months).
            step_size: Stride between successive folds.
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, n: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate (train_indices, test_indices) tuples."""
        start = self.min_train_size
        while start + self.test_size <= n:
            train_idx = np.arange(0, start)
            test_idx = np.arange(start, start + self.test_size)
            yield train_idx, test_idx
            start += self.step_size


class ARIMABaseline:
    """ARIMA(1,1,1) price forecasting baseline."""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = order

    def fit_predict(self, train_series: pd.Series, steps: int) -> np.ndarray:
        """Fit ARIMA model and generate out-of-sample forecast.

        Args:
            train_series: Historical price series.
            steps: Forecast horizon.

        Returns:
            np.ndarray of point forecasts.
        """
        try:
            model = ARIMA(train_series, order=self.order)
            res = model.fit()
            forecast = res.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            logger.warning("ARIMA fit failed: %s. Using naive random walk drift fallback.", e)
            last_val = train_series.iloc[-1]
            return np.full(steps, last_val)


class GARCHBaseline:
    """GARCH(1,1) realized volatility forecasting baseline."""

    def __init__(self) -> None:
        pass

    def fit_predict(self, returns: pd.Series, steps: int) -> np.ndarray:
        """Fit GARCH(1,1) on log returns and forecast annualized forward volatility."""
        clean = returns.dropna() * 100.0  # Scale returns to percentage for GARCH stability
        try:
            am = arch_model(clean, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off")
            forecasts = res.forecast(horizon=steps)
            variance_fwd = forecasts.variance.iloc[-1].values
            annualized_vol = (np.sqrt(variance_fwd) / 100.0) * np.sqrt(252)
            return np.array(annualized_vol)
        except Exception as e:
            logger.warning("GARCH fit failed: %s. Using historical standard deviation fallback.", e)
            hist_std = float(clean.std() / 100.0) * np.sqrt(252)
            return np.full(steps, hist_std)


class MincerZarnowitzTest:
    """Evaluates forecast efficiency and unbiasedness via Mincer-Zarnowitz regression:

    y_{t+h} = alpha + beta * y_hat_{t+h} + eps_{t+h}
    Unbiasedness test: H_0: (alpha, beta) = (0, 1)
    Efficiency test: R^2 indicates proportion of actual variation explained by forecast.
    """

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        """Estimate MZ regression and test statistics."""
        mask = np.isfinite(actual) & np.isfinite(predicted)
        y = actual[mask]
        y_hat = predicted[mask]

        if len(y) < 10:
            return {"mz_alpha": 0.0, "mz_beta": 1.0, "mz_r2": 0.0, "p_value_unbiased": 1.0}

        X = sm.add_constant(y_hat)
        model = sm.OLS(y, X).fit()

        alpha = float(model.params[0])
        beta = float(model.params[1])
        r2 = float(model.rsquared)

        # F-test for joint hypothesis: alpha = 0 and beta = 1
        hypothesis = "(const = 0), (x1 = 1)"
        try:
            f_test = model.f_test(hypothesis)
            p_val = float(f_test.pvalue)
        except Exception:
            p_val = float(model.pvalues[1])

        return {
            "mz_alpha": round(alpha, 4),
            "mz_beta": round(beta, 4),
            "mz_r2": round(r2, 4),
            "p_value_unbiased": round(p_val, 4),
        }


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Calculate Diebold-Mariano test statistic for forecast error comparison.

    H_0: Both models have equal predictive accuracy.
    H_1: Model 1 has superior accuracy over Model 2.
    """
    d = (e2**2) - (e1**2)  # Positive d means model 1 has smaller squared error
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d <= 1e-12:
        return 0.0, 1.0

    stat = float(mean_d / np.sqrt(var_d / n))
    p_value = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return round(stat, 3), round(p_value, 4)


class ModelEvaluator:
    """Orchestrates cross-validation, baseline comparison, and metric generation."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def evaluate_expanding_window(self, target_col: str = "target_return_5d") -> dict[str, Any]:
        """Perform expanding-window cross-validation comparing GBM against ARIMA baseline."""
        logger.info("Executing expanding-window CV for target: %s", target_col)
        n = len(self.df)
        cv = TimeSeriesCV(min_train_size=750, test_size=125, step_size=125)

        gbm_actuals: list[float] = []
        gbm_preds: list[float] = []
        arima_preds: list[float] = []

        arima = ARIMABaseline(order=(1, 1, 1))

        for _fold, (train_idx, test_idx) in enumerate(cv.split(n)):
            train_df = self.df.iloc[train_idx]
            test_df = self.df.iloc[test_idx]

            # Fit GBM
            gbm = GBMModel(target_col=target_col, n_estimators=100)
            gbm.fit(train_df, track_mlflow=False)
            preds_gbm = gbm.predict(test_df)

            # Fit Baseline
            if "return" in target_col:
                # ARIMA on log returns
                preds_base = arima.fit_predict(train_df["ttf_log_return"], steps=len(test_idx))
            else:
                preds_base = arima.fit_predict(train_df["ttf_price"], steps=len(test_idx))

            actuals = test_df[target_col].values

            # Accumulate
            valid_mask = np.isfinite(actuals) & np.isfinite(preds_gbm)
            gbm_actuals.extend(actuals[valid_mask])
            gbm_preds.extend(preds_gbm[valid_mask])
            arima_preds.extend(preds_base[valid_mask])

        y_true = np.array(gbm_actuals)
        y_gbm = np.array(gbm_preds)
        y_base = np.array(arima_preds)

        # Compute metrics
        e_gbm = y_true - y_gbm
        e_base = y_true - y_base

        rmse_gbm = float(np.sqrt(np.mean(e_gbm**2)))
        mae_gbm = float(np.mean(np.abs(e_gbm)))
        da_gbm = float(np.mean(np.sign(y_gbm) == np.sign(y_true)))

        rmse_base = float(np.sqrt(np.mean(e_base**2)))
        mae_base = float(np.mean(np.abs(e_base)))
        da_base = float(np.mean(np.sign(y_base) == np.sign(y_true)))

        mz_gbm = MincerZarnowitzTest.evaluate(y_true, y_gbm)
        dm_stat, dm_pval = diebold_mariano_test(e_gbm, e_base)

        results = {
            "target": target_col,
            "n_eval_samples": len(y_true),
            "gbm_rmse": round(rmse_gbm, 4),
            "gbm_mae": round(mae_gbm, 4),
            "gbm_directional_accuracy": round(da_gbm, 4),
            "base_rmse": round(rmse_base, 4),
            "base_mae": round(mae_base, 4),
            "base_directional_accuracy": round(da_base, 4),
            "rmse_improvement_pct": round(((rmse_base - rmse_gbm) / rmse_base) * 100, 2),
            "mz_r2": mz_gbm["mz_r2"],
            "mz_alpha": mz_gbm["mz_alpha"],
            "mz_beta": mz_gbm["mz_beta"],
            "mz_unbiased_pval": mz_gbm["p_value_unbiased"],
            "dm_stat": dm_stat,
            "dm_pval": dm_pval,
        }
        logger.info("CV Results for %s: %s", target_col, results)
        return results


def run_model_pipeline(dry_run: bool = False) -> dict[str, Any]:
    """Execute end-to-end model training, Bayesian inference, and cross-validation."""
    logger.info("Starting complete modeling pipeline...")
    df = pd.read_parquet("data/processed/feature_matrix.parquet")

    # 1. Bayesian Structural Time Series Model
    logger.info("Phase 1: Estimating Bayesian Structural Time Series...")
    bsts, idata = run_bayesian_model(df=df, dry_run=dry_run)
    ci_results = bsts.get_credible_intervals(prob=0.95)
    convergence = bsts.verify_convergence()

    # 2. LightGBM Model with SHAP Regime Decomposition
    logger.info("Phase 2: Training LightGBM and performing SHAP regime decomposition...")
    gbm, shap_explanation = run_gbm_model(df=df, target_col="target_return_5d")

    # Save SHAP decomposition table
    decomposed = shap_explanation["decomposed_importance"]["comparison"]
    decomposed.to_csv("reports/shap_regime_comparison.csv", index=False)

    # 3. Time Series Cross-Validation & Baseline Comparison
    logger.info("Phase 3: Performing expanding-window cross validation against ARIMA baselines...")
    evaluator = ModelEvaluator(df)
    cv_returns = evaluator.evaluate_expanding_window(target_col="target_return_5d")

    # Save metrics to DuckDB and CSV
    metrics_df = pd.DataFrame([cv_returns])
    metrics_df.to_csv("reports/model_metrics.csv", index=False)

    con = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE model_metrics AS SELECT * FROM metrics_df;")
        logger.info("Persisted model_metrics into DuckDB table.")
    finally:
        con.close()

    summary_output = {
        "bayesian_credible_intervals": ci_results,
        "bayesian_convergence": convergence,
        "cv_performance": cv_returns,
        "top_shap_crisis_features": decomposed["feature"].head(5).tolist(),
    }
    logger.info("Model pipeline completed successfully: %s", summary_output)
    return summary_output


if __name__ == "__main__":
    run_model_pipeline()
