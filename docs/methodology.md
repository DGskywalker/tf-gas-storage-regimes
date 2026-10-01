# Econometric & Machine Learning Methodology

## 1. Latent Regime Identification (Hamilton Markov Switching)
We specify a 2-regime Markov Switching variance model on TTF daily log-returns:
$$r_t = \mu_{S_t} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2_{S_t}), \quad S_t \in \{0, 1\}$$
- **Causal Filtered Probabilities**: $P(S_t = 1 \mid y_{1:t})$ conditioning strictly on past and contemporaneous information to prevent future data leakage in predictive models.
- **Smoothed Probabilities**: $P(S_t = 1 \mid y_{1:T})$ utilizing the full sample for retrospective historical analysis.

## 2. Seasonal Decomposition (STL LOESS)
Applied to both storage fill levels and wholesale log prices:
$$Y_t = T_t + S_t + R_t$$
where $T_t$ is trend, $S_t$ is seasonal (annual harmonic), and $R_t$ is the residual shock.

## 3. Bayesian Structural Time Series (NumPyro / NUTS)
$$y_t = \tau_t + \left[\beta_{\text{normal}}(1 - R_t) + \beta_{\text{crisis}} R_t\right] \cdot \text{storage\_dev}_t + \gamma R_t + \epsilon_t$$
where $\tau_t$ represents the local structural level with Fourier harmonics, and $\beta$ captures the regime-dependent price elasticity of storage deviation.

## 4. LightGBM & Game-Theoretic SHAP Attribution
Gradient boosted decision trees trained on 41 engineered features to predict forward returns and forward volatility. SHAP values are extracted via TreeExplainer and stratified across market states.
