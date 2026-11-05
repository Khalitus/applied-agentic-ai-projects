# Historical Analytics — Implementation and Validation Notes

**Project:** Urban Mobility Demand Intelligence  
**Task:** 4 — Historical SQL and Pandas Analytics  
**Status:** Implemented and validated

## 1. Purpose

This module provides a small, controlled set of read-only historical analytics functions for the Urban Mobility Demand Intelligence application. It retrieves and summarizes the official UCI Bike Sharing hourly observations from SQLite and returns structured Python results. The same functions will later be exposed to the AI agent through validated tools.

The analytics layer is distinct from the machine-learning predictor: **SQL reports observed historical patterns; it does not generate forecasts.**

## 2. Data source and storage

| Property | Details |
|---|---|
| Source | UCI Bike Sharing Dataset — https://doi.org/10.24432/C5W894 |
| Coverage | January 1, 2011 – December 31, 2012 |
| Granularity | Aggregated rental observations by hour |
| Records | 17,379 |
| SQLite table | `bike_rentals` |
| Local database | `data/processed/bike_demand.sqlite3` |
| Target/measure | `cnt` — total rentals recorded in an hour |
| Source data | `data/raw/hour.csv` |

The SQLite database is built from the validated CSV through `python -m src.db`. It is a reproducible local artifact and is excluded from Git.

## 3. Analytics function contracts

Implemented in `src/analytics.py`:

| Function | Inputs | Return type | Purpose |
|---|---|---|---|
| `hourly_profile(hour, path=DB_PATH)` | Required hour, 0–23 | `dict` | Count, mean, minimum and maximum rentals for one hour across available dates |
| `compare_workdays(path=DB_PATH)` | Optional database path | `list[dict]` | Compare working-day and non-working-day hourly observations |
| `weather_impact(path=DB_PATH)` | Optional database path | `list[dict]` | Summarize hourly rental demand by weather category |
| `hourly_demand(workingday=None, path=DB_PATH)` | Optional working-day filter: `None`, `0`, or `1` | `list[dict]` | Return hourly demand profiles for all observations or a specific day type |
| `historical_reference(hour, workingday=None, weathersit=None, path=DB_PATH)` | Hour 0–23; optional working day 0/1 and weather category 1–4 | `dict` | Historical demand summary for a user-selected scenario |

### 3.1 Response fields

- **`observations`:** Number of recorded hourly rows in the group.
- **`avg_rentals`:** Mean of `cnt`, rounded to two decimal places in SQL.
- **`min_rentals` / `max_rentals`:** Minimum and maximum `cnt`, where provided.
- **`hr`, `workingday`, `weathersit`:** Group identifiers or requested hour, depending on the function.

`hourly_demand()` returns `hr`, `observations`, and `avg_rentals` for each available hour. The workday and weather comparisons return their grouping category, observation count, mean, minimum and maximum.

For `historical_reference()`, a scenario with **no matching rows** returns `observations = 0` and `None` for the average, minimum and maximum. Zero observations must never be represented as a zero-rental average.

### 3.2 Usage examples

```python
from src.analytics import (
    compare_workdays,
    historical_reference,
    hourly_demand,
    hourly_profile,
    weather_impact,
)

hourly_profile(8)
compare_workdays()
weather_impact()
hourly_demand(workingday=1)
historical_reference(hour=8, workingday=1, weathersit=1)
```

All functions accept an optional database path so they can be tested against isolated SQLite fixtures rather than the project's production-like local dataset.

## 4. Verified historical findings

### 4.1 Working-day comparison

| Working-day status | Observations | Mean hourly rentals | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `0` — Weekend or holiday | 5,514 | 181.41 | 1 | 783 |
| `1` — Working day | 11,865 | 193.21 | 1 | 977 |

The difference between the displayed means is **11.80 rentals per observed hour**. This is a descriptive difference, not an estimate of the causal effect of working-day status.

### 4.2 Weather comparison

| Weather category | Observations | Mean hourly rentals | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 1 | 11,413 | 204.87 | 1 | 977 |
| 2 | 4,544 | 175.17 | 1 | 957 |
| 3 | 1,419 | 111.58 | 1 | 891 |
| 4 | 3 | 74.33 | 23 | 164 |

Demand is lower on average in the higher-numbered weather categories in these recorded observations. **Category 4 contains only three observations**; its mean must not be presented as a reliable general estimate.

### 4.3 Hourly reference checks

| Hour | Observations | Mean hourly rentals | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 04:00 | 697 | 6.35 | 1 | 28 |
| 08:00 | 727 | 359.01 | 5 | 839 |
| 17:00 | 730 | 461.45 | 15 | 976 |

These results were previously verified against the Pandas exploratory analysis.

## 5. Query safety and design

- User-supplied filters are bound using SQLite **`?` parameters**, rather than interpolated into SQL text.
- Optional query clauses are assembled only from fixed, developer-controlled SQL fragments.
- Validated input ranges are used for working-day status, hour and weather category in the scenario functions.
- Queries return structured rows (`dict` or `list[dict]`) suitable for downstream programmatic consumption.
- The analytics functions perform only `SELECT` operations; they do not expose arbitrary SQL execution to the application or future AI agent.
- SQLite indexes on `(dteday, hr)` and `hr` were established in Task 3. Additional indexes should be justified by measured query plans, not added speculatively.

## 6. Validation and testing

**Cross-system verification:** Pandas and SQLite returned identical working-day and weather-category averages to two decimal places. Selected hourly queries also matched the previous Pandas audit.

**Automated test status:** `python -m pytest -q` completed with **11 passed** (9 analytics tests plus 2 existing feature tests).

Analytics tests use a six-row, temporary SQLite fixture with known expected aggregates. They cover:

- Single-hour averages using `hourly_profile()`.
- Working-day grouping using `compare_workdays()`.
- Weather grouping using `weather_impact()`.
- Optional working-day filtering using `hourly_demand()`.
- Filtered historical averages using `historical_reference()`.
- No-match behavior with zero observations and `None` aggregates.
- Rejection of invalid working-day, hour and weather values in the tested entry points.

The synthetic fixture exists **only for unit testing**; the application's historical analytics use the original UCI dataset. These tests do not constitute a performance benchmark or an evaluation of ML prediction accuracy.

To reproduce verification:

```bash
python -m src.db
python -m pytest -q
```

## 7. Interpretation and limitations

1. **Historical, not real-time:** Observations are from 2011–2012. Results should not be represented as today's bike demand.
2. **Aggregated, not station-level:** The dataset records network-level hourly activity and cannot support individual station inventory or relocation recommendations.
3. **Observational comparisons:** Weather and working-day averages can be confounded by hour, season, temperature and other factors; they do not establish causality.
4. **Uneven sample sizes:** Rare weather categories and narrowly filtered scenarios may provide weak reference estimates. Always report `observations` alongside means.
5. **Incomplete hourly coverage:** The earlier audit identified 165 absent date-hour slots across 76 days, despite no null values in existing rows.
6. **Scope of scenario matching:** `historical_reference()` filters by hour, working-day status and weather category only. It does **not** match temperature, humidity, wind speed or every feature used by the ML model.
7. **Averages are not predictions:** A historical mean provides context for an ML estimate but is not independently a calibrated forecast or prediction interval.

## 8. Next integration step

In Tasks 8–10, the application will combine user-entered scenario features, ML inference and these read-only analytics functions. The AI agent will call approved tools and explain the returned results, preserving sample sizes and missing-data limitations rather than inventing numerical evidence.

**Task 4 outcome:** A validated, parameterized SQLite analytics layer ready for future prediction comparison and agent-tool integration.
