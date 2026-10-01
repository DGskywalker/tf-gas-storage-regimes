# API Reference

## Data Layer (`src.data`)
- `GIEClient`: Ingestion client for GIE AGSI+ API with pagination, rate limiting, and calibrated fallback generation.
- `PriceClient`: Ingestion and unit conversion for Dutch TTF and US Henry Hub prices.
- `DataPipeline`: DuckDB warehouse manager and D-1 lag aligner.
- `schemas`: Pydantic and Pandera validation contracts.

## Feature Layer (`src.features`)
- `FeatureEngineer`: Computes 41 features including gas year cycles, 5-year deviations, and STL components.
- `MarkovSwitchingDetector`: Fits 2-regime Hamilton filter on log-returns.
- `RollingVolTercileDetector`: Computes causal expanding realized volatility terciles.

## Modeling Layer (`src.models`)
- `BayesianStructuralTimeSeries`: NumPyro NUTS state space estimator with regime-dependent storage elasticities.
- `GBMModel`: LightGBM regressor with MLflow tracking.
- `SHAPAnalyzer`: Computes and stratifies SHAP feature importance by market regime.
- `ModelEvaluator`: Expanding-window time-series CV, baselines (ARIMA/GARCH), and Mincer-Zarnowitz tests.

## Visualization Layer (`src.visualization`)
- `plots`: Generates publication figures (Figures 1 through 6) in Matplotlib and Plotly formats.
