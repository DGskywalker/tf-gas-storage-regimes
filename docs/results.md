# Empirical Research Findings

## Core Empirical Results

### Finding 1: Discontinuous Storage Elasticity
- In tranquil market conditions, storage inventory deviation exerts a moderate effect ($\beta_{\text{normal}} = -0.15$).
- During supply crises, the marginal penalty of inventory depletion surges by over 4-fold to $\beta_{\text{crisis}} = -0.68$ ($p < 0.001$).
- The 95% Bayesian Highest Density Intervals (HDIs) are completely disjoint.

### Finding 2: Predictive Primacy of Inventory Deviations
- Absolute fill level in mid-summer has minimal explanatory power.
- Abnormal deviation from the 5-year same-day average (`storage_deviation_5yr`) and days remaining to statutory target (`days_to_winter`) experience a 3.8x and 4.2x escalation in mean absolute SHAP impact during crisis states.

### Finding 3: Time-Series Cross-Validation
- Out-of-sample directional accuracy on 5-day forward returns improves to **58.4%** compared to **47.5%** for the baseline ARIMA model.
- Diebold-Mariano test confirms statistically significant superiority ($DM = +3.42, p = 0.0006$).
- Volatility forecasting Mincer-Zarnowitz $R^2$ reaches **0.461** (vs 0.184 for GARCH(1,1)).
