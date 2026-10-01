"""Unit tests for visualization functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.plots import (
    plot_bayesian_posteriors,
    plot_cross_market_comparison,
    plot_regime_probabilities,
    plot_shap_regime_decomposition,
    plot_stl_decomposition,
    plot_ttf_storage_trajectory,
)


@pytest.fixture
def mock_plot_data() -> pd.DataFrame:
    """Create rich DataFrame for plotting tests."""
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ttf_price": np.linspace(30.0, 90.0, 100),
            "ttf_log_return": np.random.normal(0, 0.03, 100),
            "storage_pct": np.linspace(40.0, 85.0, 100),
            "storage_deviation_5yr": np.random.normal(0, 4, 100),
            "storage_stl_trend": np.linspace(45.0, 80.0, 100),
            "storage_stl_seasonal": np.sin(np.linspace(0, 2 * np.pi, 100)) * 10,
            "storage_stl_residual": np.random.normal(0, 1, 100),
            "price_stl_trend": np.linspace(3.4, 4.2, 100),
            "price_stl_seasonal": np.cos(np.linspace(0, 2 * np.pi, 100)) * 0.1,
            "price_stl_residual": np.random.normal(0, 0.05, 100),
            "regime_label": (np.arange(100) > 60).astype(int),
            "prob_crisis_filtered": np.linspace(0.1, 0.9, 100),
            "prob_crisis_smoothed": np.linspace(0.1, 0.9, 100),
            "henry_hub_eur_mwh": np.linspace(10.0, 15.0, 100),
        }
    )


def test_plot_generators(mock_plot_data: pd.DataFrame, tmp_dir: Path) -> None:
    """Test that all plot generators execute without error and produce output PNG files."""
    plot_ttf_storage_trajectory(mock_plot_data, save_dir=tmp_dir)
    assert (tmp_dir / "fig1_ttf_storage_trajectory.png").exists()

    plot_regime_probabilities(mock_plot_data, save_dir=tmp_dir)
    assert (tmp_dir / "fig2_regime_smoothed_probabilities.png").exists()

    plot_bayesian_posteriors(save_dir=tmp_dir)
    assert (tmp_dir / "fig3_bayesian_posterior_storage_effect.png").exists()

    plot_shap_regime_decomposition(save_dir=tmp_dir)
    assert (tmp_dir / "fig4_shap_regime_decomposition.png").exists()

    plot_stl_decomposition(mock_plot_data, save_dir=tmp_dir)
    assert (tmp_dir / "fig5_stl_decomposition.png").exists()

    plot_cross_market_comparison(mock_plot_data, save_dir=tmp_dir)
    assert (tmp_dir / "fig6_cross_market_comparison.png").exists()
