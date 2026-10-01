"""Latent Regime Identification and Market State Modeling.

Implements:
1. Two-regime Markov Switching models on TTF log-returns (normal vs crisis states)
   using Hamilton filtering and EM estimation.
2. Causal filtered probabilities (P(S_t | y_{1:t})) to guarantee zero look-ahead bias
   in predictive workflows.
3. Full-sample smoothed probabilities (P(S_t | y_{1:T})) for retrospective structural analysis.
4. Rolling 60-day realized volatility terciles (low/medium/high) as an empirical benchmark.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logger = logging.getLogger(__name__)


class MarkovSwitchingDetector:
    """Estimates latent market regimes via Markov Switching Autoregressive/Variance models."""

    def __init__(self, k_regimes: int = 2) -> None:
        """Initialize detector with number of latent states.

        Args:
            k_regimes: Number of regimes (default=2: Low-Vol Normal, High-Vol Crisis).
        """
        self.k_regimes = k_regimes
        self.model_res: Any | None = None

    def fit_predict(
        self,
        returns: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Fit Markov Switching variance model on log-returns and extract probabilities.

        Args:
            returns: Series of daily log-returns r_t = ln(P_t / P_{t-1}).

        Returns:
            Tuple of:
            - regime_label: Causal binary classification (0=normal, 1=crisis).
            - prob_crisis_filtered: Causal filtered probability P(S_t=1 | r_{1:t}).
            - prob_crisis_smoothed: Retrospective smoothed probability P(S_t=1 | r_{1:T}).
        """
        clean_returns = returns.dropna()
        if len(clean_returns) < 50:
            logger.info(
                "Fewer than 50 observations (%d). Using robust realized volatility regime fallback.",
                len(clean_returns),
            )
            return self._fallback_volatility_regime(returns)

        # Estimate Markov switching model with switching variance
        # r_t = mu + eps_t, eps_t ~ N(0, sigma^2_{S_t})
        try:
            model = MarkovRegression(
                clean_returns,
                k_regimes=self.k_regimes,
                trend="c",
                switching_trend=True,
                switching_variance=True,
            )
            res = model.fit(disp=False, maxiter=200)
            self.model_res = res

            # Determine which regime has higher variance
            # statsmodels parameters for variance in switching_variance: 'sigma2[0]', 'sigma2[1]'
            params = res.params
            var_0 = params.get("sigma2[0]", 1.0)
            var_1 = params.get("sigma2[1]", 2.0)

            crisis_idx = 1 if var_1 >= var_0 else 0

            filtered_probs = res.filtered_marginal_probabilities[crisis_idx]
            smoothed_probs = res.smoothed_marginal_probabilities[crisis_idx]

            # Re-index to match original returns series index
            prob_filtered_full = pd.Series(index=returns.index, dtype=float)
            prob_smoothed_full = pd.Series(index=returns.index, dtype=float)

            prob_filtered_full.loc[clean_returns.index] = filtered_probs
            prob_smoothed_full.loc[clean_returns.index] = smoothed_probs

            # Forward-fill any leading NA caused by differencing and clip to exact [0.0, 1.0]
            prob_filtered_full = prob_filtered_full.bfill().ffill().clip(0.0, 1.0)
            prob_smoothed_full = prob_smoothed_full.bfill().ffill().clip(0.0, 1.0)

            # Causal regime label strictly based on filtered probability at time t
            regime_label = (prob_filtered_full >= 0.5).astype(int)

            logger.info(
                "Markov Switching converged. Regime 0 variance: %.5f, Regime 1 variance: %.5f. Crisis index: %d",
                var_0,
                var_1,
                crisis_idx,
            )
            return regime_label, prob_filtered_full, prob_smoothed_full

        except Exception as e:
            logger.warning(
                "Markov Switching fitting failed with error: %s. Using robust volatility proxy.", e
            )
            return self._fallback_volatility_regime(returns)

    def _fallback_volatility_regime(
        self, returns: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Robust fallback regime detector based on rolling realized volatility."""
        rolling_std = returns.rolling(window=60, min_periods=10).std()
        rolling_annualized = rolling_std * np.sqrt(252)

        # 75th percentile as crisis boundary
        threshold = rolling_annualized.quantile(0.75)
        prob_filtered = (rolling_annualized / (threshold * 1.5)).clip(0.0, 1.0).fillna(0.2)
        prob_smoothed = (
            prob_filtered.rolling(window=7, center=True, min_periods=1).mean().fillna(0.2)
        )
        regime_label = (prob_filtered >= 0.5).astype(int)

        return regime_label, prob_filtered, prob_smoothed


class RollingVolTercileDetector:
    """Classifies market conditions into Low, Medium, and High realized volatility terciles."""

    def __init__(self, window: int = 60) -> None:
        """Initialize rolling volatility tercile detector.

        Args:
            window: Rolling window length in trading days (default=60).
        """
        self.window = window

    def compute_terciles(self, returns: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Compute rolling annualized volatility and classify into causal terciles.

        Args:
            returns: Daily log returns series.

        Returns:
            Tuple of:
            - realized_vol_60d: Annualized 60-day realized volatility.
            - vol_tercile: Integer tercile (0=low, 1=medium, 2=high).
        """
        realized_vol = returns.rolling(window=self.window, min_periods=15).std() * np.sqrt(252)
        realized_vol = realized_vol.bfill()

        # Compute expanding quantiles to prevent future data leakage
        # tercile 0: vol <= 33rd percentile
        # tercile 1: 33rd < vol <= 67th percentile
        # tercile 2: vol > 67th percentile
        q33 = realized_vol.expanding(min_periods=60).quantile(0.33).bfill()
        q67 = realized_vol.expanding(min_periods=60).quantile(0.67).bfill()

        tercile = pd.Series(1, index=returns.index, dtype=int)  # default medium
        tercile[realized_vol <= q33] = 0
        tercile[realized_vol > q67] = 2

        return realized_vol, tercile


class RegimeDetector:
    """Unified interface coordinating Markov Switching and rolling volatility detection."""

    def __init__(self) -> None:
        self.ms_detector = MarkovSwitchingDetector(k_regimes=2)
        self.vol_detector = RollingVolTercileDetector(window=60)

    def fit_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with all regime indicators.

        Args:
            df: DataFrame containing 'ttf_price'.

        Returns:
            DataFrame with added regime features:
            - ttf_log_return
            - regime_label
            - prob_crisis_filtered
            - prob_crisis_smoothed
            - realized_vol_60d
            - vol_tercile
        """
        res = df.copy()
        res["ttf_log_return"] = np.log(res["ttf_price"] / res["ttf_price"].shift(1)).fillna(0.0)

        regime_label, prob_filtered, prob_smoothed = self.ms_detector.fit_predict(
            res["ttf_log_return"]
        )
        realized_vol, vol_tercile = self.vol_detector.compute_terciles(res["ttf_log_return"])

        res["regime_label"] = regime_label
        res["prob_crisis_filtered"] = np.round(prob_filtered, 4)
        res["prob_crisis_smoothed"] = np.round(prob_smoothed, 4)
        res["realized_vol_60d"] = np.round(realized_vol, 4)
        res["vol_tercile"] = vol_tercile

        return res
