# Architecture and Data Flow

## System scope

Urban Mobility Demand Intelligence provides hourly rental estimates for user-defined historical-pattern scenarios and evidence-backed aggregate summaries over 2011–2012 UCI Bike Sharing records. The current delivered interface supports predictive analytics; a restricted tool-calling agent is a separate integration point, not a deployed feature.

## Components

```text
Official UCI hour.csv
    |
    +--> src/data.py ----------> src/db.py ----------> SQLite
    |                                |                     |
    |                                +--> src/analytics.py-+
    |
    +--> src/features.py
             |
             +--> src/train.py -----> baseline model + report
             |
             +--> src/optimize.py --> optimized model + analysis
                                          |
Scenario form (app.py) --> src/predict.py --+
                            |               |
                            +---------------+--> Historical SQL comparison
                                                |
                                            Displayed result

src/agent.py: restricted tool-calling integration point (not yet enabled)
```

## Data and source integrity

`src/data.py` validates required columns, unique record identifiers, missing values and the equality `casual + registered == cnt`. `src/db.py` writes the validated 17,379 source observations to SQLite with schema constraints, uniqueness on date-hour combinations and an hour index. `src/analytics.py` exposes a small set of parameterized queries rather than arbitrary user-executed SQL.

## Feature and model contract

`RAW_FEATURES` contains nine inputs; `ENGINEERED_FEATURES` adds `commute_peak`. `training_features()` creates and orders the ten model columns for both source observations and single-row user scenarios. Neither the target nor the target-derived rental counts are prediction inputs.

Model training is chronological by complete dates, with comparisons against both a global and hour-specific historical mean. Optimization selects configurations on a separate, earlier validation interval. A separate optimized artifact is used for interactive inference and is checked for the correct feature names before prediction.

## Application boundary

`src/predict.predict_demand()` returns a validated numeric estimate. `estimate_scenario()` combines that estimate with `historical_reference()` using matching hour, working-day status and weather category. The Gradio layer formats the result and reports missing artifacts or invalid inputs. SQL comparison conditions are intentionally narrower than the model's feature set and must not be represented as exact scenario matches.

## Restricted agent integration

`src/agent.py` reserves the allowlisted tool boundary. LLM-driven tool selection and execution are not enabled. Integration should use typed schemas, bounded tool-call loops, validated arguments, structured tool responses and numeric claims grounded in SQL/model outputs. No arbitrary SQL or filesystem access should be exposed.

## Constraints

This system does not access current weather, inventory, station locations or real-time rental data. It cannot forecast present-day activity with established reliability or make station-level repositioning recommendations. The dataset contains only three observations for weather category 4. Recorded holdout performance is an historical evaluation, not a deployment guarantee.
