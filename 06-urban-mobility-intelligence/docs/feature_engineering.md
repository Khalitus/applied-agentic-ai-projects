# Feature Engineering and Chronological Validation

## 1. Purpose

This document records the feature contract and temporal evaluation strategy for the **Urban Mobility Demand Intelligence** project. The objective is to prepare consistent inputs for a supervised hourly bike-rental demand model and for predictions generated from user-submitted scenarios.

**Dataset:** UCI Bike Sharing Dataset (`hour.csv`), Capital Bikeshare activity from January 2011 through December 2012.  
**Observations:** 17,379 hourly records.  
**Prediction target:** `cnt` (total rentals recorded in an hour).  
**Implementation:** `src/config.py`, `src/data.py`, `src/features.py`, `src/train.py`.

## 2. Feature Contract

The project separates source features from engineered features in `src/config.py`.

| Type | Features | Count |
|---|---|---:|
| Raw inputs | `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `hum`, `windspeed` | 9 |
| Engineered input | `commute_peak` | 1 |
| Model input | `RAW_FEATURES + ENGINEERED_FEATURES` | **10** |
| Prediction target | `cnt` | 1 |

The model receives features in the explicitly configured order. Both the historical training DataFrame and a one-row user scenario use `training_features()` to apply the same engineered-feature rules.

### Excluded source columns

- `casual` and `registered` are excluded because their sum equals `cnt`; including them would directly leak the target.
- `instant` identifies source records and is not a demand predictor.
- `dteday` supplies calendar context rather than being passed as a raw string.
- `yr`, `season`, and `atemp` are excluded from this baseline feature contract. Their inclusion may be evaluated in a later experiment, but they are not part of the present model input.

Excluding `yr` means the model does not explicitly represent the difference in overall rental activity between 2011 and 2012. This is a limitation, not evidence that the exclusion improves predictive performance.

## 3. Engineered Commute-Peak Indicator

`commute_peak` is set to `1` when both conditions hold:

1. `workingday == 1`.
2. `hr` is one of `7`, `8`, `9`, `16`, `17`, or `18`.

It is `0` otherwise. The indicator represents a plausible working-day peak window identified from exploratory patterns; it does **not** establish that an individual trip was a commute.

| Indicator | Observations | Dataset share |
|---|---:|---:|
| `commute_peak = 0` | 14,395 | 82.83% |
| `commute_peak = 1` | 2,984 | 17.17% |

Because `commute_peak` is derived entirely from `workingday` and `hr`, it adds no new external information. Whether it improves XGBoost must be measured against a model trained without the indicator during model optimization.

## 4. Training and User-Scenario Parity

The data flows are intentionally unified:

```text
Historical CSV -> Select raw inputs ----\
                                     training_features() -> 10-column model input
User form -> Validate -> Normalize ---/
```

For user-submitted scenarios, `scenario_features()`:

1. Parses the ISO-format date.
2. Converts Python weekday numbering (Monday = 0) to the dataset's convention (Sunday = 0).
3. Determines working-day status from the weekday and user-supplied holiday flag.
4. Converts physical weather measurements to the normalized units used by the dataset.
5. Passes the resulting nine raw inputs through `training_features()` to create `commute_peak` and enforce model feature order.

| User input | Transformation | Model field |
|---|---|---|
| Date | Month from date | `mnth` |
| Date | `(date.weekday() + 1) % 7` | `weekday` |
| Hour | Validated integer 0–23 | `hr` |
| Holiday | Boolean mapped to 0/1 | `holiday` |
| Weekday and holiday | 1 on non-holiday Monday–Friday, otherwise 0 | `workingday` |
| Weather code | Validated integer 1–4 | `weathersit` |
| Temperature in °C | Divide by 41 | `temp` |
| Relative humidity in % | Divide by 100 | `hum` |
| Wind speed in km/h | Divide by 67 | `windspeed` |
| Working-day status and hour | Peak-window rule | `commute_peak` |

The unit conversions follow the source dataset's normalization conventions. They describe a historical-scenario interface rather than live weather observations.

### Verified example

Input: `2012-08-15`, 08:00, non-holiday, weather category 1, 22°C, 60% humidity and 12 km/h wind.

| `mnth` | `hr` | `holiday` | `weekday` | `workingday` | `weathersit` | `temp` | `hum` | `windspeed` | `commute_peak` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 0 | 3 | 1 | 1 | 0.536585 | 0.6 | 0.179104 | 1 |

The feature ordering matched the training DataFrame, and neither `cnt` nor target-derived count columns appeared in the generated model input.

## 5. Chronological Train/Test Split

The original row-based 80/20 split fell within **August 7, 2012**, placing some observations from the same date on opposite sides of the split.

The revised `chronological_split()` sorts by date and hour, calculates its cutoff using **unique calendar dates**, and assigns whole dates to the earlier training period or later holdout period.

| Measure | Verified result |
|---|---|
| Total records | 17,379 |
| Training records | 13,891 |
| Testing records | 3,488 |
| Last training date | 2012-08-06 |
| First testing date | 2012-08-07 |
| Shared dates across splits | None |

The 80/20 fraction applies to unique dates, so the record counts are not required to form an exact 80/20 partition. This is a **single chronological holdout**, not rolling-origin cross-validation. Additional temporal checks and error analysis belong to later modeling tasks.

## 6. Validation and Test Results

The implementation was checked with the following results:

| Check | Result |
|---|---|
| Historical feature matrix | `(17379, 10)` |
| Scenario feature matrix | `(1, 10)` |
| Configured feature order consistent | `True` |
| Missing model input values | `0` |
| `cnt` excluded from inputs | `True` |
| Chronological dates separated | `True` |
| Full automated test suite | **19 passed** |

Feature tests cover calendar mapping, holiday handling, commute-peak rules, training/scenario parity and representative invalid inputs. The split test verifies distinct and correctly ordered calendar dates.

## 7. Constraints and Next Steps

- **No model accuracy is claimed in this task.** MAE, RMSE and baseline comparisons will be established during Task 6.
- Predictions will represent **historical-pattern scenarios**, not current bike-share demand forecasts. The underlying data covers 2011–2012.
- The user supplies holiday status; the interface does not independently verify a holiday calendar.
- An engineered feature must be assessed for incremental value rather than assumed beneficial because it is intuitively plausible.
- A previously saved model using nine features must not be used with the new ten-feature input contract. Retrain and regenerate the model artifact during Task 6.
- Numeric performance on later dates may be affected by year-to-year demand changes that the present feature contract does not explicitly model.

**Task 5 status:** Feature engineering and chronological validation implemented, with 19 passing tests reported in the development environment. Baseline training and performance evaluation remain the scope of Task 6.
