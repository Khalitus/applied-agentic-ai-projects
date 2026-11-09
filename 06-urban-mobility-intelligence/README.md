# Urban Mobility Demand Intelligence

A historical bicycle-rental demand intelligence application combining validated UCI Bike Sharing data, SQLite analytics, XGBoost regression and an interactive Gradio prediction form.

The system estimates **total hourly rentals** for a user-defined calendar/weather scenario and compares the estimate with historical observations that match the selected hour, working-day status and weather category. It is a **historical-pattern scenario estimator**, not a live demand forecast.

## Data source

Fanaee-T, H. (2013), *Bike Sharing*, UCI Machine Learning Repository. [Dataset and citation](https://doi.org/10.24432/C5W894). License: CC BY 4.0. The source contains 17,379 hourly observations from Capital Bikeshare, January 2011–December 2012.

The original dataset is downloaded directly from UCI. No synthetic training data is substituted. Downloaded datasets, generated model artifacts, reports and secrets are excluded from version control.

## Capabilities

- Validate and ingest source observations into constrained, indexed SQLite tables.
- Query historical rental summaries by hour, weather and working-day status using parameterized SQL.
- Build consistent ten-column model inputs for training and interactive scenarios.
- Train baseline and optimized XGBoost regression models with date-separated evaluation.
- Return an optimized model estimate alongside a clearly scoped historical reference and observation count.
- Reject invalid inputs or missing model/database artifacts with actionable setup errors.

Agent tool calling is an architectural integration point and **is not enabled** in the current interface. It will not be presented as implemented functionality.

## Setup

From the `06-urban-mobility-intelligence` project directory, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m src.db
python -m src.train
python -m src.optimize
python -m pytest -q
python app.py
```

Open the local Gradio URL printed in the terminal. If using an existing activated project environment, skip virtual-environment creation and installation steps as appropriate.

## Architecture

| Component | Responsibility |
|---|---|
| `src/data.py`, `scripts/download_data.py` | Source acquisition and integrity checks |
| `src/db.py` | Schema-enforced SQLite ingestion |
| `src/analytics.py` | Read-only, parameterized historical aggregations |
| `src/features.py` | Shared training/scenario feature engineering |
| `src/train.py` | Baseline regression and chronological holdout evaluation |
| `src/optimize.py` | Inner-validation model selection, ablation and error analysis |
| `src/predict.py` | Optimized inference and matched historical reference |
| `app.py` | Gradio scenario interface |
| `src/agent.py` | Inactive restricted agent integration point |

See [Architecture and Data Flow](docs/architecture.md) for component boundaries and data-flow details.

## Model evaluation

Results below were measured against the later-date historical holdout (3,488 observations starting 2012-08-07). Training uses 13,891 earlier observations ending 2012-08-06.

| Method | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Global mean baseline | 174.971 | 232.559 | — |
| Hour-specific mean baseline | 113.678 | 167.319 | — |
| Baseline XGBoost | 89.589 | 126.341 | 0.6717 |
| **Optimized XGBoost** | **88.445** | **124.719** | **0.6801** |

The optimized regressor reduces holdout MAE by approximately 1.28% relative to the baseline regressor. Selection relied on an earlier chronological validation period; the holdout had already been inspected in preceding evaluation, so it is not a wholly untouched test. See [Model Optimization and Error Analysis](docs/model_optimization.md).

## Reproducible outputs

The training and optimization scripts write serialized models and JSON/CSV reports into `models/` and `reports/`, both Git-ignored. Serialized joblib models should only be loaded from trusted sources.

## Limitations

Historical data covers 2011–2012; there is no real-time weather feed, bicycle inventory or station-level information. A scenario's date is used for calendar features and is not a learned projection of future demand. Year-on-year network growth is not modeled explicitly. Historical SQL comparisons match categorical conditions but not exact temperature, humidity, wind or date. The rarest weather category has only three observations in the source dataset.

## Documentation

- [Architecture](docs/architecture.md)
- [Data analytics](docs/analytics_notes.md)
- [Feature engineering](docs/feature_engineering.md)
- [Baseline evaluation](docs/model_baseline.md)
- [Optimization and error analysis](docs/model_optimization.md)
- [Prediction interface](docs/prediction_interface.md)

## License and attribution

Source data: UCI Machine Learning Repository / Fanaee-T, 2013, CC BY 4.0. Any separate license for project code is governed by the repository's licensing terms.
