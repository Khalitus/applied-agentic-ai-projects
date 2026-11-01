# Project 6: Urban Mobility Demand Intelligence — implementation blueprint

**Learning approach:** 55% understanding, 35% programming, 10% recall.
**Main concepts:** SQLite, Pandas, XGBoost regression, LLM function calling,
validated tool execution, Gradio and testing.
**Scope boundary:** one agent; no multi-agent orchestration, MCP or LangGraph yet.

## Core question

Given a proposed date, hour and weather conditions, how many bicycle rentals
would the model expect under historical Capital Bikeshare patterns? What do
historical SQL aggregates say about similar periods, and can an agent reliably
choose suitable tools to give a grounded explanation?

## Dataset

UCI Bike Sharing, 17,379 hourly rows, Jan 2011–Dec 2012. Download the official
`hour.csv` with `python -m scripts.download_data`. Dataset DOI:
https://doi.org/10.24432/C5W894. License: CC BY 4.0. No fake rows.

| Source column | Meaning | Usage |
| --- | --- | --- |
| dteday, hr | historical date, hour | sort and historical SQL |
| mnth, weekday, holiday, workingday | calendar | model and queries |
| weathersit | coded weather category (1–4) | model and queries |
| temp, hum, windspeed | normalized meteorological data | model and queries |
| cnt | hourly rental total | regression target |
| casual, registered | count breakdown | descriptive analysis ONLY |
| instant | original row ID | auditing only |

## Architectural contract

1. Raw CSV is immutable. `src/data.py` validates it.
2. `src/db.py` imports all validated data into SQLite for historical questions.
3. `src/features.py` maps user inputs to EXACTLY the train-time feature order.
4. `src/train.py` makes a chronological 80/20 split, benchmarks a naive mean
   predictor and trains XGBoost. It never includes target-derived features.
5. `src/predict.py` is the sole callable for ML inference; use only a trusted
   model artifact from this training pipeline.
6. `src/analytics.py` exposes whitelist SQL tools with bound parameters; no
   user-provided SQL execution.
7. `src/agent.py` will use the LLM to select tools, validate arguments and
   ground answers in returned results, with a maximum number of rounds.
8. `app.py` provides an independent prediction form and an AI-question tab.

## Implementation tasks

| Task | Focus | What you should implement / verify |
| --- | --- | --- |
| 1 | Setup | Install deps, fetch UCI CSV, inspect source and disclaimer |
| 2 | Data audit | Inspect shape, schema, missingness, target distribution, duplicates |
| 3 | SQLite | Explain schema, add two safe aggregations, check indexes |
| 4 | Historical analytics | Implement `compare_workdays`, `weather_impact`, group-by queries |
| 5 | Feature contract | Explain calendar encoding and target leakage; test 3 edge cases |
| 6 | ML baseline | Run initial training, interpret MAE/RMSE and compare with mean baseline |
| 7 | Model improvement | Analyze hour/weather errors, tune XGBoost, record before vs after |
| 8 | Predict form | Validate human inputs, explain old-data limitation, test scenarios |
| 9 | Agent tool schemas | Define typed JSON specs and whitelist `hourly_profile`, `predict_demand`, analytics |
| 10 | Agent loop | Implement LLM tool calling, tool dispatch, limits, grounded answer, failures |
| 11 | Gradio integration | Add historical charts, use informative errors, clean UI, screenshots |
| 12 | QA and publish | Tests, README, documentation, controlled demo, source credit, commits |

The starter implements infrastructure and a first baseline. Tasks 2, 4, 7,
9–12 intentionally require meaningful writing and testing by the learner.

## Suggested challenge questions for the agent

1. "Which hours have the highest average rental counts?"
2. "Do working days and non-working days have different morning demand?"
3. "How does severe weather affect historical demand?"
4. "Estimate demand on 2012-08-15 at 08:00 given clear weather, 22°C,
    60% humidity and 12 km/h wind. Compare with historical 08:00 demand."
5. "Why shouldn't casual and registered be ML features?"
6. "What can't this model tell us about 2026 actual demand?"

## Grounded-agent constraints

- Typed, allowlisted tools; never expose arbitrary `run_sql(query)` to an LLM.
- Validate ranges and ensure the agent cannot access filesystem secrets.
- Make at most 4 rounds of LLM tool calls, with explicit tool error responses.
- Quote numbers only after a model/SQL tool provides them.
- No real-time prediction or station-level operational recommendation claims.
- Distinguish historical averages from model conditional predictions.

## Model evaluation contract

Chronological holdout means the test period is after the train period. Use
MAE and RMSE, plus an explicit naive-mean baseline. Since user inputs do not
include years and the source contains secular system growth, test error
may be worse than random splits. That is meaningful and not a reason to leak
future data. Future stretch: stronger seasonal baseline, TimeSeriesSplit, error
by hour/weather/day type, robust uncertainty estimates.

## Deliverables

- Clean GitHub repo with source attribution and no `.env` or trained models.
- SQL analytics examples and a model metrics report.
- Screenshots: user input form, model estimate, grounded agent answer.
- Tests for feature order, leakage, day encoding and SQL restrictions.
- A README explaining architecture, metrics, data age and limitations.
