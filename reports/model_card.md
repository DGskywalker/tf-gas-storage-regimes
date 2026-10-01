# Model Card: European Gas Storage & TTF Price Relationship Models

## Model Overview
- **Model Name**: `ttf-storage-regime-suite`
- **Model Version**: `1.0.0`
- **Release Date**: October 2026
- **Model Types**:
  1. **Bayesian Structural Time Series (BSTS)** with regime-dependent exogenous storage effects (NumPyro / NUTS).
  2. **Regime-Aware Gradient Boosted Trees (GBM)** with SHAP interpretability (LightGBM).
  3. **Markov Switching Autoregressive / Variance Model** (Hamilton Filter).
  4. **Classical Benchmarks**: ARIMA(1,1,1) and GARCH(1,1).

---

## Intended Use & Application Scope
### Primary Intended Uses:
- **Energy Econometric Research**: Quantifying structural elasticity shifts between physical inventory levels and wholesale European gas benchmarks (Dutch TTF).
- **Macro Risk Management**: Stress testing utility procurement portfolios and margin requirements against low-inventory, high-volatility scenarios.
- **Policy Evaluation**: Assessing the market price impacts of statutory European Union storage refill mandates (e.g., 90% by November 1).

### Out-of-Scope and Prohibited Uses:
- High-frequency algorithmic intraday execution (models are calibrated for daily and weekly horizons).
- Direct extrapolation to illiquid European emerging gas hubs without local recalibration.
- Fully automated trading without human oversight during geopolitical force majeure events.

---

## Data Provenance & Ingestion Specs
- **Physical Storage Inventories**: Gas Infrastructure Europe (GIE) AGSI+ Transparency Platform. Covers aggregate European Union inventories and country-level facilities (AT, DE, FR, IT, NL).
- **Wholesale Price Series**: Dutch Title Transfer Facility (TTF) front-month settlement prices (ICE Endex / settlement reports).
- **US Benchmark**: US Energy Information Administration (EIA) Henry Hub spot prices.
- **Temporal Alignment**: Daily frequency. To eliminate look-ahead bias, market trade dates on day $t$ strictly observe storage finalized and published on day $t-1$ ($D-1$ reporting lag constraint).

---

## Quantitative Performance Summary
- **Bayesian MCMC Diagnostics**:
  - Gelman-Rubin $\hat{R} \le 1.00 \ll 1.01$ across all structural parameters.
  - Effective Sample Size (ESS Bulk) $\ge 1,368 \gg 400$.
- **Time-Series Cross-Validation (Expanding Window)**:
  - Directional Accuracy on 5-Day Forward Returns: **$58.4\%$** (vs $47.5\%$ for ARIMA baseline).
  - Forecast Superiority: Diebold-Mariano test statistic $DM = +3.42$ ($p = 0.0006$).
  - Mincer-Zarnowitz $R^2 = 0.461$ for 20-day forward realized volatility forecasting (vs $0.184$ for GARCH(1,1)).

---

## Known Limitations and Edge Cases

### 1. Self-Reporting and Strategic Behavior
Storage data on AGSI+ is self-reported by facility operators (SSOs) pursuant to EU Regulation (EC) No 715/2009. Operators may strategically delay revisions or misreport operational capacity constraints during supply tight spots.

### 2. Physical vs Usable Working Gas
While GIE reports "working gas in storage," technical withdrawal rates decline as cavern pressure drops. A reservoir at 25% capacity cannot withdraw gas at the same peak rate as one at 85% capacity. Models utilizing scalar percentage fill do not fully capture non-linear cushion gas extraction physics.

### 3. Geopolitical and Infrastructure Discontinuities
Extreme structural breaks (such as the 2022 Nord Stream pipeline sabotages, sudden LNG export terminal outages, or transit halts across Ukraine) induce non-stationary regime shifts that cannot be anticipated purely through historical inventory levels.

### 4. Regulatory Intervention Distortions
Implementation of administrative gas price caps (e.g. the EU Market Correction Mechanism established in 2023) truncates the empirical distribution of prices, creating artificial ceilings absent in historical training data.

---

## Ethical and Environmental Considerations
- **Fair Allocation of Essential Goods**: Natural gas is a fundamental driver of household winter heating and industrial electricity generation across Europe. Modeling price spikes must not be utilized for anti-competitive hoarding or market manipulation in violation of REMIT (Regulation on Wholesale Energy Market Integrity and Transparency).
- **Energy Transition & Stranded Asset Risk**: Underground storage facilities represent multi-decade infrastructure assets. As Europe transitions toward biomethane, hydrogen, and electrification under the European Green Deal, long-term models must account for phased decommissioning and storage retrofitting.

---

## Monitoring and Maintenance
- **Drift Detection**: Monthly automated Kolmogorov-Smirnov tests comparing incoming storage deviation and volatility distributions against historical baselines.
- **Model Recalibration**: Quarterly retraining of Markov Switching transition matrices and LightGBM hyperparameters via MLflow orchestration.
