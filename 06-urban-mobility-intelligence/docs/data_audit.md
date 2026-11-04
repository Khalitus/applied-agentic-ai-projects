# Data Quality and Exploratory Analysis Report

## 1. Dataset Overview

**Project:** Urban Mobility Demand Intelligence  
**Dataset:** UCI Bike Sharing Dataset  
**Source:** UCI Machine Learning Repository  
**Period:** January 1, 2011 – December 31, 2012  
**Primary file:** `data/raw/hour.csv`  
**Prediction target:** `cnt` — total hourly bike rentals

| Property | Value |
|---|---|
| Observations | 17,379 |
| Features/columns | 17 |
| Unique dates | 731 |
| Missing values | 0 |
| Duplicate records | 0 |
| Duplicate IDs | 0 |
| Duplicate timestamps | 0 |
| Invalid rental-count relationships | 0 |

The original dataset was preserved without modification.

## 2. Rental Demand Distribution

| Statistic | Value |
|---|---:|
| Minimum | 1 |
| Mean | 189.46 |
| Median | 142 |
| Standard deviation | 181.39 |
| 5th percentile | 5 |
| 25th percentile | 40 |
| 75th percentile | 281 |
| 95th percentile | 563.10 |
| Maximum | 977 |

Rental demand is right-skewed, with substantial variation between low-demand and peak-demand periods.

The difference between the mean and median suggests that large rental counts influence the overall average.

**Modeling consideration:** Evaluate regression performance using MAE and RMSE, including performance across different demand periods.

## 3. Hourly Demand Patterns

| Hour | Mean rentals |
|---|---:|
| 17:00 | 461.45 |
| 18:00 | 425.51 |
| 08:00 | 359.01 |
| 04:00 | 6.35 |

The highest average demand occurs during morning and evening hours, consistent with commuter activity.

The lowest average demand occurs at 04:00.

**Modeling consideration:** Hour of day is likely to be an informative prediction feature.

## 4. Weather-Related Demand

| Weather category | Observations | Average rentals | Dataset share |
|---|---:|---:|---:|
| 1 | 11,413 | 204.87 | 65.67% |
| 2 | 4,544 | 175.17 | 26.15% |
| 3 | 1,419 | 111.58 | 8.17% |
| 4 | 3 | 74.33 | 0.02% |

Average rental demand decreases across the recorded weather categories.

However, these are observational relationships rather than causal estimates.

Weather category 4 contains only three observations, so its average is not statistically robust.

**Modeling consideration:** Investigate prediction performance across weather categories, particularly categories with limited observations.

## 5. Working-Day Demand

| Working-day status | Observations | Mean rentals | Median rentals |
|---|---:|---:|---:|
| Non-working day | 5,514 | 181.41 | 119 |
| Working day | 11,865 | 193.21 | 151 |

Working days have slightly higher overall mean demand.

The non-working-day category includes weekends and public holidays.

**Modeling consideration:** Evaluate interactions between working-day classification and hour of day.

## 6. Yearly Demand Comparison

| Year | Observations | Total rentals | Mean hourly rentals |
|---|---:|---:|---:|
| 2011 | 8,645 | 1,243,103 | 143.79 |
| 2012 | 8,734 | 2,049,576 | 234.67 |

Recorded rental totals increased by approximately **64.88%** between 2011 and 2012.

This difference should not be attributed to any single factor without further investigation.

**Modeling consideration:** Use a chronological validation strategy and evaluate performance on later observations.

## 7. Temporal Completeness

| Metric | Value |
|---|---:|
| Unique dates | 731 |
| Days with 24 hourly observations | 655 |
| Days with fewer than 24 observations | 76 |
| Minimum observations on a day | 1 |
| Expected records in a complete hourly grid | 17,544 |
| Available records | 17,379 |
| Absent hourly slots | 165 |

The dataset has no null values in existing records, but 165 date-hour combinations are absent relative to a complete 24-hour-per-day calendar.

These gaps require consideration in time-series analysis and model evaluation.

Absent records should not automatically be replaced with zero rental counts.

## 8. Data Leakage and Modeling Constraints

The following considerations must be respected:

- `cnt` is the prediction target.
- `casual` and `registered` must not be used as model inputs because their sum equals `cnt`.
- `instant` is a record identifier, not a meaningful operational demand feature.
- Normalized weather variables require consistent transformation during training and inference.
- Chronological train/test splitting should be used to avoid unrealistic evaluation.
- Predictions must be treated as historical-scenario estimates rather than current operational forecasts.

## 9. Conclusion

The dataset passed the initial structural and consistency checks and is suitable for further exploratory analysis, SQL integration and baseline regression modeling.

The primary findings are strong hourly demand variation, different demand levels across weather conditions, substantially higher recorded rentals in 2012, and irregular hourly coverage.

The next development stage will load the validated observations into SQLite and implement parameterized historical analytics.

**Status:** Initial data quality audit completed.