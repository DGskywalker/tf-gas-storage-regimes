# Natural Gas Storage & TTF Price Relationship Analysis

Welcome to the documentation for the **Natural Gas Storage and TTF Price Relationship Analysis** framework.

This research framework investigates the dynamic, non-stationary equilibrium between physical natural gas storage inventories in the European Union and wholesale front-month settlement prices at the Dutch Title Transfer Facility (TTF).

## Core Capabilities
- **GIE AGSI+ Ingestion**: Robust automated pipeline with pagination, rate limiting, and exponential backoff.
- **Strict Look-Ahead Bias Prevention**: Point-in-time temporal alignment respecting the $D-1$ reporting lag.
- **Markov Switching Regime Detection**: Hamilton filtering separating low-volatility normal states from high-volatility crisis periods.
- **Bayesian Structural Time Series**: Fully Bayesian inference using NumPyro (NUTS) with Gelman-Rubin $\hat{R} < 1.01$ and ESS $> 400$.
- **Game-Theoretic Explainability**: Regime-stratified SHAP feature decomposition via LightGBM.
- **Cross-Market Comparison**: Comparative elasticity analysis against the US Henry Hub benchmark.

## Quickstart
```bash
make install
make run
```
