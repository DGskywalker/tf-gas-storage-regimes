"""Unit tests for econometric models, Bayesian BSTS, GBM, and cross-validation."""

import numpy as np
import pandas as pd

from src.models.bayesian_ts import BayesianStructuralTimeSeries
from src.models.evaluation import (
    ARIMABaseline,
    GARCHBaseline,
    MincerZarnowitzTest,
    TimeSeriesCV,
    diebold_mariano_test,
)
from src.models.gbm_model import GBMModel


def test_bayesian_bsts_convergence() -> None:
    """Test NumPyro BSTS model fitting and convergence diagnostics on synthetic data."""
    np.random.seed(42)
    T = 80
    y = np.linspace(3.0, 3.5, T) + np.random.normal(0, 0.05, T)
    storage = np.sin(np.linspace(0, 2 * np.pi, T))
    regime = (np.arange(T) > 45).astype(float)

    bsts = BayesianStructuralTimeSeries(num_warmup=150, num_samples=300, num_chains=2)
    prior_draws = bsts.run_prior_predictive(storage=storage, regime=regime, num_samples=50)
    assert "y" in prior_draws

    bsts.fit(y=y, storage=storage, regime=regime)
    assert bsts.summary_df is not None

    ci = bsts.get_credible_intervals(prob=0.95)
    assert "beta_normal" in ci
    assert "beta_crisis" in ci
    assert ci["beta_normal"][0] < ci["beta_normal"][2]


def test_gbm_model_and_shap() -> None:
    """Test LightGBM model fitting and SHAP decomposition."""
    np.random.seed(99)
    N = 150
    df = pd.DataFrame(
        {
            "storage_pct": np.random.uniform(30, 95, N),
            "storage_pct_quantile": np.random.uniform(0, 1, N),
            "storage_change_7d": np.random.normal(0, 1, N),
            "storage_deviation_5yr": np.random.normal(0, 5, N),
            "winter_premium": np.random.uniform(1, 5, N),
            "month_sin": np.random.uniform(-1, 1, N),
            "month_cos": np.random.uniform(-1, 1, N),
            "days_to_winter": np.random.randint(10, 300, N),
            "regime_label": np.random.choice([0, 1], N, p=[0.7, 0.3]),
            "prob_crisis_filtered": np.random.uniform(0, 1, N),
            "realized_vol_60d": np.random.uniform(0.2, 0.8, N),
            "target_return_5d": np.random.normal(0, 0.03, N),
        }
    )

    train_df = df.iloc[:100]
    test_df = df.iloc[100:]

    gbm = GBMModel(target_col="target_return_5d", n_estimators=20)
    gbm.fit(train_df, track_mlflow=False)
    preds = gbm.predict(test_df)

    assert len(preds) == len(test_df)
    assert np.all(np.isfinite(preds))

    explanation = gbm.explain(test_df)
    assert "shap_values" in explanation
    assert "decomposed_importance" in explanation
    comp = explanation["decomposed_importance"]["comparison"]
    assert "feature" in comp.columns
    assert "mean_abs_shap_normal" in comp.columns
    assert "mean_abs_shap_crisis" in comp.columns


def test_time_series_cv_splits() -> None:
    """Test expanding window forward-chaining splits."""
    cv = TimeSeriesCV(min_train_size=100, test_size=25, step_size=25)
    splits = list(cv.split(n=200))

    assert len(splits) == 4
    for i, (train_idx, test_idx) in enumerate(splits):
        # Monotonic expanding train set
        assert len(train_idx) == 100 + i * 25
        assert len(test_idx) == 25
        # Strict zero look-ahead bias: max(train) < min(test)
        assert train_idx.max() < test_idx.min()


def test_baselines_and_mincer_zarnowitz() -> None:
    """Test ARIMA, GARCH, and Mincer-Zarnowitz efficiency regression."""
    np.random.seed(12)
    series = pd.Series(np.cumsum(np.random.normal(0, 1, 100)) + 30.0)

    # ARIMA baseline
    arima = ARIMABaseline(order=(1, 1, 1))
    fwd = arima.fit_predict(series, steps=5)
    assert len(fwd) == 5
    assert np.all(np.isfinite(fwd))

    # GARCH baseline
    returns = series.pct_change().dropna()
    garch = GARCHBaseline()
    vol_fwd = garch.fit_predict(returns, steps=5)
    assert len(vol_fwd) == 5
    assert (vol_fwd >= 0.0).all()

    # Mincer-Zarnowitz test
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    predicted = actual + np.random.normal(0, 0.1, 10)
    mz = MincerZarnowitzTest.evaluate(actual, predicted)
    assert "mz_r2" in mz
    assert mz["mz_r2"] > 0.90

    # Diebold-Mariano test
    e1 = np.random.normal(0, 0.5, 30)
    e2 = np.random.normal(0, 1.0, 30)
    stat, pval = diebold_mariano_test(e1, e2)
    assert isinstance(stat, float)
    assert 0.0 <= pval <= 1.0


def test_model_pipeline_dry_run() -> None:
    """Test run_model_pipeline with dry_run flag."""
    from src.models.evaluation import run_model_pipeline

    res = run_model_pipeline(dry_run=True)
    assert "bayesian_credible_intervals" in res
    assert "cv_performance" in res
    assert "bayesian_convergence" in res
