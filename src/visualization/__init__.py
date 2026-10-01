"""Visualization and publication figure generation module."""

from src.visualization.plots import (
    generate_all_figures,
    plot_bayesian_posteriors,
    plot_cross_market_comparison,
    plot_regime_probabilities,
    plot_shap_regime_decomposition,
    plot_stl_decomposition,
    plot_ttf_storage_trajectory,
)

__all__ = [
    "plot_ttf_storage_trajectory",
    "plot_regime_probabilities",
    "plot_bayesian_posteriors",
    "plot_shap_regime_decomposition",
    "plot_stl_decomposition",
    "plot_cross_market_comparison",
    "generate_all_figures",
]
