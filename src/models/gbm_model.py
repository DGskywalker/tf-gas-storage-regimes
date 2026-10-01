"""Gradient-Boosted Tree Models with SHAP Interpretability and MLflow Tracking.

Trains LightGBM models to predict forward returns and realized volatility,
logs parameters and metrics with MLflow, and calculates regime-decomposed
SHAP (SHapley Additive exPlanations) values to quantify the state-dependent
marginal contribution of gas storage features.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "storage_pct",
    "storage_pct_quantile",
    "storage_change_7d",
    "storage_deviation_5yr",
    "winter_premium",
    "month_sin",
    "month_cos",
    "days_to_winter",
    "regime_label",
    "prob_crisis_filtered",
    "realized_vol_60d",
]


class SHAPAnalyzer:
    """Computes and decomposes SHAP values stratified across market regimes."""

    def __init__(self, model: lgb.LGBMRegressor, feature_names: list[str]) -> None:
        """Initialize SHAP analyzer with trained model.

        Args:
            model: Trained LightGBM regressor.
            feature_names: List of feature names matching X matrix columns.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for given dataset.

        Args:
            X: Feature matrix DataFrame.

        Returns:
            np.ndarray of SHAP values with shape (N, P).
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.array(shap_values)

    def decompose_by_regime(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        regimes: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        """Decompose feature importance (mean absolute SHAP) stratified by normal vs crisis regimes.

        Args:
            X: Feature matrix.
            shap_values: Calculated SHAP matrix.
            regimes: Series of binary regime indicators (0=normal, 1=crisis).

        Returns:
            Dictionary with 'overall', 'normal_regime', and 'crisis_regime' importance tables.
        """
        regime_arr = np.array(regimes)
        abs_shap = np.abs(shap_values)

        overall_imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.mean(abs_shap, axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        mask_normal = regime_arr == 0
        mask_crisis = regime_arr == 1

        normal_imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap_normal": np.mean(abs_shap[mask_normal], axis=0)
                    if np.any(mask_normal)
                    else 0.0,
                }
            )
            .sort_values("mean_abs_shap_normal", ascending=False)
            .reset_index(drop=True)
        )

        crisis_imp = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap_crisis": np.mean(abs_shap[mask_crisis], axis=0)
                    if np.any(mask_crisis)
                    else 0.0,
                }
            )
            .sort_values("mean_abs_shap_crisis", ascending=False)
            .reset_index(drop=True)
        )

        merged = pd.merge(overall_imp, normal_imp, on="feature")
        merged = pd.merge(merged, crisis_imp, on="feature")
        merged["crisis_multiplier"] = (
            merged["mean_abs_shap_crisis"] / merged["mean_abs_shap_normal"].clip(lower=1e-6)
        ).round(2)

        return {
            "overall": overall_imp,
            "normal": normal_imp,
            "crisis": crisis_imp,
            "comparison": merged.sort_values("mean_abs_shap_crisis", ascending=False),
        }


class GBMModel:
    """Gradient-Boosted Tree predictor for natural gas market returns and volatility."""

    def __init__(
        self,
        target_col: str = "target_return_5d",
        feature_cols: list[str] | None = None,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        num_leaves: int = 20,
        random_state: int = 42,
    ) -> None:
        """Initialize GBM Model configuration."""
        self.target_col = target_col
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            objective="regression",
            n_jobs=-1,
            verbosity=-1,
        )
        self.shap_analyzer: SHAPAnalyzer | None = None
        self.last_shap_values: np.ndarray | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        track_mlflow: bool = True,
    ) -> GBMModel:
        """Fit LightGBM model with optional early stopping and MLflow logging."""
        clean_train = train_df.dropna(subset=self.feature_cols + [self.target_col])
        X_train = clean_train[self.feature_cols]
        y_train = clean_train[self.target_col]

        eval_set = None
        if val_df is not None:
            clean_val = val_df.dropna(subset=self.feature_cols + [self.target_col])
            X_val = clean_val[self.feature_cols]
            y_val = clean_val[self.target_col]
            eval_set = [(X_val, y_val)]

        if track_mlflow:
            try:
                mlflow.set_experiment("ttf_storage_prediction")
                with mlflow.start_run(run_name=f"lgb_{self.target_col}", nested=True):
                    mlflow.log_params(
                        {
                            "target": self.target_col,
                            "n_estimators": self.n_estimators,
                            "learning_rate": self.learning_rate,
                            "num_leaves": self.num_leaves,
                            "n_features": len(self.feature_cols),
                        }
                    )
                    self.model.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                        if eval_set
                        else None,
                    )
                    preds_train = self.model.predict(X_train)
                    train_rmse = float(np.sqrt(np.mean((preds_train - y_train) ** 2)))
                    mlflow.log_metric("train_rmse", train_rmse)
            except Exception as e:
                logger.warning("MLflow tracking encountered non-fatal error: %s", e)
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        self.shap_analyzer = SHAPAnalyzer(self.model, self.feature_cols)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        clean_X = X[self.feature_cols]
        return np.asarray(self.model.predict(clean_X))

    def explain(self, test_df: pd.DataFrame) -> dict[str, Any]:
        """Compute SHAP explanations and regime decomposition on evaluation dataset."""
        if self.shap_analyzer is None:
            raise ValueError("Model must be fitted before running SHAP explanation.")

        clean_test = test_df.dropna(subset=self.feature_cols)
        X_test = clean_test[self.feature_cols]
        regimes = (
            clean_test["regime_label"]
            if "regime_label" in clean_test.columns
            else pd.Series(0, index=clean_test.index)
        )

        shap_vals = self.shap_analyzer.explain(X_test)
        self.last_shap_values = shap_vals

        decomposed = self.shap_analyzer.decompose_by_regime(X_test, shap_vals, regimes)
        return {
            "shap_values": shap_vals,
            "X_test": X_test,
            "regimes": regimes,
            "decomposed_importance": decomposed,
        }


def run_gbm_model(
    df: pd.DataFrame | None = None,
    target_col: str = "target_return_5d",
) -> tuple[GBMModel, dict[str, Any]]:
    """Execute standard train-test evaluation with SHAP regime decomposition."""
    if df is None:
        df = pd.read_parquet("data/processed/feature_matrix.parquet")

    # Split: 80% train, 20% test (chronological split)
    n = len(df)
    train_size = int(n * 0.80)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    gbm = GBMModel(target_col=target_col)
    gbm.fit(train_df, val_df=test_df, track_mlflow=False)

    explanation = gbm.explain(test_df)
    logger.info(
        "GBM model trained. Top features across regimes:\n%s",
        explanation["decomposed_importance"]["comparison"].head(8),
    )
    return gbm, explanation


if __name__ == "__main__":
    run_gbm_model()
