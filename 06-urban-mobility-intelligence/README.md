# Urban Mobility Demand Intelligence

A historical bicycle-rental demand intelligence application combining validated UCI Bike Sharing data, SQLite analytics, XGBoost regression and a three-part Gradio application with prediction, historical visualization and a bounded tool-calling agent.

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
- Answer independent natural-language analytics questions through an allowlisted, validated tool-calling agent.
- Prefer Groq or Gemini through local configuration, with the other provider available as API-error fallback.
- Visualize recorded mean hourly demand on working and non-working days from SQLite.

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
cp .env.example .env             # Add GROQ_API_KEY or GEMINI_API_KEY
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
| `app.py` | Gradio prediction, agent Q&A and SQL visualization |
| `src/agent.py` | Gemini/Groq tool-calling orchestration and provider fallback |
| `src/agent_tools.py` | Typed tool declarations, argument validation and allowlisted execution |

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

The agent answers **each question independently**; prior chat messages appear in the interface but are not passed to the language model as conversational context. LLM-written explanations can contain mistakes even when SQL/model figures are correct. The working-day percentage comparison is calculated by Python and supplied to the agent, but final answers still require review. Provider access, usage quotas and API availability vary.

Historical data covers 2011–2012; there is no real-time weather feed, bicycle inventory or station-level information. A scenario's date is used for calendar features and is not a learned projection of future demand. Year-on-year network growth is not modeled explicitly. Historical SQL comparisons match categorical conditions but not exact temperature, humidity, wind or date. The rarest weather category has only three observations in the source dataset.

## Documentation

- [Architecture](docs/architecture.md)
- [Data analytics](docs/analytics_notes.md)
- [Feature engineering](docs/feature_engineering.md)
- [Baseline evaluation](docs/model_baseline.md)
- [Optimization and error analysis](docs/model_optimization.md)
- [Prediction interface](docs/prediction_interface.md)
- [Agent tool execution](docs/agent_execution.md)
- [Provider routing](docs/provider_fallback.md)
- [Agent application interface](docs/agent_interface.md)

## License and attribution

Source data: UCI Machine Learning Repository / Fanaee-T, 2013, CC BY 4.0. Any separate license for project code is governed by the repository's licensing terms.
