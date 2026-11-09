# Architecture and Data Flow

## System scope

Urban Mobility Demand Intelligence provides hourly rental estimates for user-defined historical-pattern scenarios and evidence-backed aggregate summaries over 2011–2012 UCI Bike Sharing records. The current delivered interface supports predictive analytics; a restricted tool-calling agent is a separate integration point, not a deployed feature.

## Components

```text
UCI hour.csv -> src/data.py -> src/db.py -> SQLite
       |                                  |
       |                                  +-> src/analytics.py -> Gradio hourly chart
       |                                                     |
       +-> src/features.py -> src/train.py / src/optimize.py  |
                                   |                         |
                                   +-> optimized XGBoost     |
                                            |                 |
Scenario form -> src/predict.py -------------+----- SQL comparison
        |
        +-> Gradio demand estimate

Natural-language question -> src/agent.py (Groq / Gemini)
                                   |
                          src/agent_tools.py
                                   |
                  validated analytics and scenario tools
                                   |
                          evidence-backed answer
                                   |
                             Gradio agent tab
```

## Data and source integrity

`src/data.py` validates required columns, unique record identifiers, missing values and the equality `casual + registered == cnt`. `src/db.py` writes the validated 17,379 source observations to SQLite with schema constraints, uniqueness on date-hour combinations and an hour index. `src/analytics.py` exposes a small set of parameterized queries rather than arbitrary user-executed SQL.

## Feature and model contract

`RAW_FEATURES` contains nine inputs; `ENGINEERED_FEATURES` adds `commute_peak`. `training_features()` creates and orders the ten model columns for both source observations and single-row user scenarios. Neither the target nor the target-derived rental counts are prediction inputs.

Model training is chronological by complete dates, with comparisons against both a global and hour-specific historical mean. Optimization selects configurations on a separate, earlier validation interval. A separate optimized artifact is used for interactive inference and is checked for the correct feature names before prediction.

## Application boundary

`src/predict.predict_demand()` returns a validated numeric estimate. `estimate_scenario()` combines that estimate with `historical_reference()` using matching hour, working-day status and weather category. The Gradio layer formats the result and reports missing artifacts or invalid inputs. SQL comparison conditions are intentionally narrower than the model's feature set and must not be represented as exact scenario matches.

## Restricted agent integration

`src/agent.py` implements manual function calling for Groq and Gemini with configurable provider priority and API-failure fallback. `src/agent_tools.py` enforces a six-tool allowlist, bounded JSON arguments, range validation and sanitized execution errors. A single provider attempt is limited to four tool rounds and eight tool calls. All local tools are read-only, so a provider restart can safely reissue historical queries.

Tool responses include data computed by Python or SQL; specifically the working-day versus non-working-day absolute and relative differences are calculated in `src/analytics.py`. The model is instructed to use these supplied comparison values, but prompt-level grounding cannot guarantee all prose is accurate. Human review remains appropriate.

The Gradio agent interface displays chat exchanges, but `ask_agent()` processes each question independently and receives no previous messages. The UI explicitly discloses this limitation. Only `AGENT_PRIMARY_PROVIDER`, `GEMINI_MODEL` and `GROQ_MODEL` select providers/models; the local `.env` holds credentials and is excluded from version control.

## Historical visualization

The `Historical patterns` tab builds a two-series hourly profile from parameterized SQLite aggregates, not model-generated data. Missing databases and query errors are surfaced without leaking internal connection details.

## Constraints

This system does not access current weather, inventory, station locations or real-time rental data. It cannot forecast present-day activity with established reliability or make station-level repositioning recommendations. The dataset contains only three observations for weather category 4. Recorded holdout performance is an historical evaluation, not a deployment guarantee.
