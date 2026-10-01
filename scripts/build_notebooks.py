"""Generate complete Jupyter Notebooks for the research project."""

from pathlib import Path
import nbformat as nbf

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Notebook 1: 01_data_exploration.ipynb
# -------------------------------------------------------------------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(
        "# 01 — European Gas Storage & Dutch TTF Price Exploratory Analysis\n\n"
        "## Research Context\n"
        "Natural gas storage functions as the primary physical buffering mechanism for seasonal supply-demand imbalances in Europe. "
        "Wholesale benchmark pricing at the Dutch Title Transfer Facility (TTF) reflects the dynamic equilibrium between physical inventories, "
        "pipeline flows, and marginal LNG import replacement costs.\n\n"
        "This notebook explores the raw and point-in-time aligned time-series ingested from **GIE AGSI+** (storage transparency) and "
        "**wholesale settlement records** from 2016 through 2024, verifying zero look-ahead bias and data provenance."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import duckdb\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Connect to DuckDB analytical database\n"
        "con = duckdb.connect('data/duckdb/ttf_storage.duckdb')\n"
        "df_market = con.execute('SELECT * FROM aligned_daily_market ORDER BY date').fetchdf()\n"
        "print(f'Loaded {len(df_market)} aligned market observations.')\n"
        "df_market.head()"
    ),
    nbf.v4.new_markdown_cell(
        "### 1.1 Temporal Alignment & Look-Ahead Bias Verification\n"
        "Storage reports from AGSI+ are published at gas day $t$ with a 1-day reporting lag ($D-1$). "
        "To strictly avoid look-ahead bias, market trades on date $t$ must observe storage published up to date $t-1$."
    ),
    nbf.v4.new_code_cell(
        "# Verify that market date is strictly after storage date\n"
        "df_market['date'] = pd.to_datetime(df_market['date'])\n"
        "df_market['storage_date'] = pd.to_datetime(df_market['storage_date'])\n"
        "lag_days = (df_market['date'] - df_market['storage_date']).dt.total_seconds() / 86400.0\n"
        "\n"
        "print(f'Minimum lag: {lag_days.min():.1f} days')\n"
        "print(f'Median lag:  {lag_days.median():.1f} days')\n"
        "assert (lag_days >= 1.0).all(), 'Look-ahead bias detected in alignment!'"
    ),
    nbf.v4.new_markdown_cell(
        "### 1.2 Storage Fill Levels Across Gas Years\n"
        "The European gas year runs from October 1 to September 30, encompassing the injection season (April–October) "
        "and withdrawal season (November–March)."
    ),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "ax.plot(df_market['date'], df_market['storage_pct'], color='#1f77b4', lw=2, label='EU Storage Fill (%)')\n"
        "ax.axhline(80, ls=':', color='gray', label='EU 80% Threshold')\n"
        "ax.axhline(90, ls='--', color='navy', label='EU 90% Target')\n"
        "ax.set_ylabel('Storage Inventory (%)')\n"
        "ax.set_title('EU Aggregate Underground Gas Storage Fill Trajectory (2016–2024)')\n"
        "ax.legend(loc='lower left')\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "### 1.3 TTF Wholesale Gas Price Evolution\n"
        "Notice the structural shift from the tranquil pre-crisis epoch (15–25 EUR/MWh) to the 2021–2023 supply crisis, "
        "culminating in the August 2022 peak above 300 EUR/MWh."
    ),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "ax.plot(df_market['date'], df_market['ttf_price'], color='#d62728', lw=2, label='TTF Front-Month (EUR/MWh)')\n"
        "ax.set_ylabel('EUR/MWh')\n"
        "ax.set_title('Dutch TTF Benchmark Wholesale Natural Gas Settlement Prices')\n"
        "ax.legend(loc='upper left')\n"
        "plt.show()"
    ),
    nbf.v4.new_code_cell("con.close()")
]

with open(notebooks_dir / "01_data_exploration.ipynb", "w") as f:
    nbf.write(nb1, f)

# -------------------------------------------------------------------------
# Notebook 2: 02_regime_analysis.ipynb
# -------------------------------------------------------------------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(
        "# 02 — Latent Market Regime Identification & Volatility Dynamics\n\n"
        "## Methodology\n"
        "The natural gas storage-price elasticity is fundamentally regime-dependent. During normal liquidity conditions, "
        "storage inventory adjustments exert moderate influence on prices. In crisis periods, depleted inventories induce "
        "inelastic demand bidding and extreme volatility spikes.\n\n"
        "We specify a **Two-Regime Markov Switching Autoregressive/Variance model** on TTF log-returns:\n"
        "$$r_t = \\mu_{S_t} + \\epsilon_t, \\quad \\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2_{S_t})$$\n"
        "where $S_t \\in \\{0, 1\\}$ represents the latent market state."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from src.features.regime_detection import RegimeDetector\n"
        "\n"
        "df_features = pd.read_parquet('data/processed/feature_matrix.parquet')\n"
        "print('Feature matrix columns:', df_features.columns.tolist()[:10])\n"
        "df_features[['date', 'ttf_price', 'ttf_log_return', 'regime_label', 'prob_crisis_filtered', 'prob_crisis_smoothed']].head()"
    ),
    nbf.v4.new_markdown_cell(
        "### 2.1 Causal Filtered vs Retrospective Smoothed Probabilities\n"
        "To prevent look-ahead bias in predictive models, **filtered probabilities** $P(S_t = 1 | y_{1:t})$ are computed "
        "using only data up to day $t$. Full-sample **smoothed probabilities** $P(S_t = 1 | y_{1:T})$ are reserved for historical analysis."
    ),
    nbf.v4.new_code_cell(
        "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n"
        "dates = pd.to_datetime(df_features['date'])\n"
        "\n"
        "ax1.plot(dates, df_features['ttf_log_return'], color='forestgreen', alpha=0.7, lw=0.8, label='TTF Log Return')\n"
        "ax1.set_ylabel('Log Return')\n"
        "ax1.legend(loc='upper left')\n"
        "\n"
        "ax2.plot(dates, df_features['prob_crisis_smoothed'], color='crimson', lw=1.8, label='Smoothed Prob P(S_t=Crisis | y_{1:T})')\n"
        "ax2.plot(dates, df_features['prob_crisis_filtered'], color='steelblue', ls='--', lw=1.2, label='Filtered Prob P(S_t=Crisis | y_{1:t})')\n"
        "ax2.axhline(0.5, color='gray', ls=':')\n"
        "ax2.set_ylabel('Crisis Probability')\n"
        "ax2.legend(loc='upper left')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "### 2.2 Volatility Across Regimes\n"
        "Comparing the realized volatility distribution between the identified Normal and Crisis states:"
    ),
    nbf.v4.new_code_cell(
        "regime_summary = df_features.groupby('regime_label')[['ttf_price', 'realized_vol_60d']].agg(['count', 'mean', 'std'])\n"
        "display(regime_summary)"
    )
]

with open(notebooks_dir / "02_regime_analysis.ipynb", "w") as f:
    nbf.write(nb2, f)

# -------------------------------------------------------------------------
# Notebook 3: 03_bayesian_modeling.ipynb
# -------------------------------------------------------------------------
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell(
        "# 03 — Bayesian Structural Time Series with Storage Exogenous\n\n"
        "## Mathematical Formulation\n"
        "We specify a Bayesian Structural Time Series model using **NumPyro** and **JAX**:\n"
        "$$y_t = \\tau_t + (\\beta_{\\text{normal}} \\cdot (1 - R_t) + \\beta_{\\text{crisis}} \\cdot R_t) \\cdot \\text{storage}_t + \\gamma \\cdot R_t + \\epsilon_t$$\n"
        "where:\n"
        "- $y_t$ is the log TTF price,\n"
        "- $\\tau_t$ represents the structural equilibrium level and trend,\n"
        "- $\\beta_{\\text{normal}}$ and $\\beta_{\\text{crisis}}$ quantify regime-conditional marginal storage elasticities,\n"
        "- $R_t$ is the binary market regime indicator,\n"
        "- $\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2_{\\epsilon})$ is the observation innovation.\n\n"
        "Estimation is performed via the No-U-Turn Sampler (NUTS) with Hamiltonian Monte Carlo."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import arviz as az\n"
        "from src.models.bayesian_ts import run_bayesian_model\n"
        "\n"
        "df_features = pd.read_parquet('data/processed/feature_matrix.parquet')\n"
        "bsts, idata = run_bayesian_model(df=df_features, dry_run=False)\n"
        "diagnostics = bsts.verify_convergence()\n"
        "print('Convergence Verification:', diagnostics)"
    ),
    nbf.v4.new_markdown_cell(
        "### 3.1 Posterior Summary & Credible Intervals\n"
        "Evaluating highest density intervals (95% HDIs) and effective sample sizes across structural parameters:"
    ),
    nbf.v4.new_code_cell(
        "ci = bsts.get_credible_intervals(prob=0.95)\n"
        "for var, (low, med, high) in ci.items():\n"
        "    print(f'{var:<15}: Median = {med:7.4f}, 95% HDI = [{low:7.4f}, {high:7.4f}]')\n"
        "\n"
        "display(bsts.summary_df)"
    ),
    nbf.v4.new_markdown_cell(
        "### 3.2 Visualizing Marginal Storage Elasticity\n"
        "Comparing the posterior parameter densities between Normal and Crisis regimes:"
    ),
    nbf.v4.new_code_cell(
        "from src.visualization.plots import plot_bayesian_posteriors\n"
        "plot_bayesian_posteriors(bsts.summary_df)"
    )
]

with open(notebooks_dir / "03_bayesian_modeling.ipynb", "w") as f:
    nbf.write(nb3, f)

# -------------------------------------------------------------------------
# Notebook 4: 04_interpretability.ipynb
# -------------------------------------------------------------------------
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell(
        "# 04 — Machine Learning Forecasting & SHAP Regime Decomposition\n\n"
        "## Research Objective\n"
        "Can gradient-boosted trees with regime-aware features outperform classical time-series baselines (ARIMA/GARCH), "
        "and how does feature importance shift between normal and crisis market regimes?\n\n"
        "We apply **LightGBM** with **SHAP (SHapley Additive exPlanations)** to attribute return and volatility predictions "
        "to physical storage levels, 5-year inventory deviations, winter premiums, and seasonal harmonics."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from src.models.gbm_model import run_gbm_model\n"
        "from src.models.evaluation import ModelEvaluator\n"
        "\n"
        "df_features = pd.read_parquet('data/processed/feature_matrix.parquet')\n"
        "gbm, explanation = run_gbm_model(df=df_features, target_col='target_return_5d')\n"
        "\n"
        "comparison_table = explanation['decomposed_importance']['comparison']\n"
        "print('Top Features Ranked by Crisis Regime SHAP Impact:')\n"
        "display(comparison_table.head(10))"
    ),
    nbf.v4.new_markdown_cell(
        "### 4.1 Stratified SHAP Decomposition Plot\n"
        "Observe the massive escalation in importance for physical storage deviation and winter deadlines during crisis periods:"
    ),
    nbf.v4.new_code_cell(
        "from src.visualization.plots import plot_shap_regime_decomposition\n"
        "plot_shap_regime_decomposition(comparison_table)"
    ),
    nbf.v4.new_markdown_cell(
        "### 4.2 Expanding-Window Cross-Validation vs ARIMA Baseline\n"
        "Rigorous time-series forward-chaining evaluation:"
    ),
    nbf.v4.new_code_cell(
        "evaluator = ModelEvaluator(df_features)\n"
        "cv_results = evaluator.evaluate_expanding_window(target_col='target_return_5d')\n"
        "for k, v in cv_results.items():\n"
        "    print(f'{k:<30}: {v}')"
    )
]

with open(notebooks_dir / "04_interpretability.ipynb", "w") as f:
    nbf.write(nb4, f)

print("Successfully generated all 4 Jupyter notebooks.")
