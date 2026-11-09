# Release Validation and Acceptance Criteria

## Scope

This guide documents reproducible checks for the local historical bike-rental demand intelligence application. It distinguishes automated local tests, live provider checks and manual browser verification. Passing simulated tests does not prove that a hosted model will always choose the correct tools or produce a correct answer.

## Prerequisites

- Python 3.12 and installed dependencies from `requirements.txt`.
- Downloaded official UCI `data/raw/hour.csv` (17,379 hourly observations).
- Initialized SQLite database `data/processed/bike_demand.sqlite3`.
- Optimized XGBoost artifact `models/demand_xgb_optimized.joblib` generated locally.
- For chat only, `GROQ_API_KEY` or `GEMINI_API_KEY` stored in the ignored `.env`.

The source CSV, SQLite database, trained model, evaluation reports and credentials are not version-controlled.

## Deterministic checks

```bash
python -m pytest -q
python -m scripts.release_check
```

The release diagnostic checks source coverage, SQLite row/date counts, the ten-feature saved model, scenario inference, working-day comparison arithmetic, and rejection of arbitrary SQL tool requests. It **does not call either LLM provider**. Failed checks report only a generic exception class without local path or credential details.

If data or artifacts are missing, run the respective preparation commands:

```bash
python -m scripts.download_data
python -m src.db
python -m src.train
python -m src.optimize
```

## Live agent verification

The preferred provider is configured by `AGENT_PRIMARY_PROVIDER`. Groq is the recommended setting for the observed local environment; Gemini can be kept as fallback. Use the following sparingly because each tool-calling interaction may involve multiple provider requests:

```bash
python -m scripts.agent_smoke_test
python -m scripts.evaluate_agent --live
```

The optional agent evaluator asks four independent historical questions: working-day comparison, busiest working-day hours, the 08:00 hourly reference and the small weather-category-4 sample. It compares numerical evidence in responses against **fresh local SQL results** and writes `reports/agent_evaluation.json` (ignored by Git).

**Interpretation:** A `PASS` only means the expected numerical values appear within stated tolerances. It does not prove the model used the right tool, interpreted a value correctly, or avoided unsupported conclusions. Review answer phrasing, units, sample sizes and limitations manually. A `REVIEW` means examine the answer and tool behavior, not necessarily that the provider is down.

## Browser acceptance

Run `python app.py` and check the three Gradio tabs:

| View | Acceptance checks |
|---|---|
| Demand estimate | Submit `2012-08-15`, 08:00, non-holiday, clear weather, 22°C, 60% humidity, 12 km/h wind. With the verified model, expect approximately **471.5 rentals/hour**; historical comparison approximately **503.86**, with **280 observations**. Confirm that the values are labeled differently. |
| Ask the agent | Ask for working-/non-working-day means; verify **5,514** and **11,865** observations, **181.41** and **193.21** rentals/hour, and **6.50%**, not 12%. Ask for the busiest working-day hours and weather categories. Answers must identify 2011–2012 as the source. |
| Historical patterns | Click **Load hourly profiles**. Check that both day types appear across hours 0–23 and that their line values come from SQLite, not generated model text. |

Also check the invalid-input path, missing-model/database messages, rare category-4 sample warnings, and provider failure behavior. Chat follow-ups must repeat relevant context; history is displayed but intentionally not used as model input.

## Screenshots and evidence

The repository includes a real captured `screenshots/demand_estimate.png`. Additional real screenshots can be committed under descriptive names, such as `agent_workday_comparison.png`, `agent_busiest_hours.png`, `agent_weather_categories.png`, and `historical_patterns.png`. Never create artificial screenshots or misrepresent illustrative content as actual program output.

## Security and source controls

- Confirm `.env`, serialized models, downloaded data and generated reports are not tracked (`git status --short`, `git check-ignore .env models/demand_xgb_optimized.joblib`).
- Review `git diff --cached` before committing; do not stage any credentials.
- Verify only the six approved analytics/prediction tools are exposed; no model-authored SQL or Python is executed.
- Preserve the historical-data and non-operational limitations in the interface and README.
- Do not claim that a mocked test suite or isolated one-off live smoke test establishes production-grade reliability.

## Final repository checks

```bash
git status --short
python -m pytest -q
python -m scripts.release_check
git diff --cached --check
```

A release is ready for publication only after the local full test suite, data/model diagnostic and manual browser acceptance checks pass. Provider errors or incorrect agent facts should be documented instead of silently suppressed.
