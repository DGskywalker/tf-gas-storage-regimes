"""Unit tests for latent regime detection and volatility classification."""

import numpy as np
import pandas as pd

from src.features.regime_detection import (
    MarkovSwitchingDetector,
    RegimeDetector,
    RollingVolTercileDetector,
)


def test_markov_switching_detector() -> None:
    """Test 2-regime Markov Switching model on simulated returns with a volatility shift."""
    np.random.seed(42)
    # 100 observations low-vol (sigma=0.01), 100 observations high-vol (sigma=0.06)
    r1 = np.random.normal(0, 0.01, 100)
    r2 = np.random.normal(0, 0.06, 100)
    returns = pd.Series(np.concatenate([r1, r2]))

    detector = MarkovSwitchingDetector(k_regimes=2)
    labels, filtered, smoothed = detector.fit_predict(returns)

    assert len(labels) == 200
    assert len(filtered) == 200
    assert len(smoothed) == 200

    # Probabilities must lie in [0, 1]
    assert (filtered >= 0.0).all() and (filtered <= 1.0).all()
    assert (smoothed >= 0.0).all() and (smoothed <= 1.0).all()

    # Labels must be binary {0, 1}
    assert set(labels.unique()).issubset({0, 1})

    # High volatility regime should dominate second half
    assert labels.iloc[120:].mean() > labels.iloc[:80].mean()


def test_rolling_vol_terciles() -> None:
    """Test 60-day rolling realized volatility tercile calculation."""
    np.random.seed(123)
    returns = pd.Series(np.random.normal(0, 0.02, 180))
    detector = RollingVolTercileDetector(window=60)
    vol, tercile = detector.compute_terciles(returns)

    assert len(vol) == 180
    assert len(tercile) == 180
    assert (vol >= 0.0).all()
    assert set(tercile.unique()).issubset({0, 1, 2})


def test_regime_detector_integration() -> None:
    """Test unified RegimeDetector on DataFrame."""
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    prices = np.exp(np.cumsum(np.random.normal(0, 0.02, 120)) + np.log(30.0))
    df = pd.DataFrame({"date": dates, "ttf_price": prices})

    detector = RegimeDetector()
    res = detector.fit_all(df)

    assert "regime_label" in res.columns
    assert "prob_crisis_filtered" in res.columns
    assert "realized_vol_60d" in res.columns
    assert "vol_tercile" in res.columns
