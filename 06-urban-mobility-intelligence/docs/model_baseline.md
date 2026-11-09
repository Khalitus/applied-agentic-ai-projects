# XGBoost Baseline Training and Evaluation

## Objective

Establish a reproducible regression baseline for predicting total hourly bike rentals (`cnt`) from the UCI Bike Sharing Dataset (2011–2012). This is a historical-scenario estimator, not a live demand forecast.

## Data and Features

- Source: official UCI `hour.csv`, 17,379 hourly observations.
- Target: `cnt`, the recorded total hourly rentals.
- Predictors: nine permitted raw columns plus `commute_peak` (ten inputs total).
- Excluded from model inputs: `cnt`, `casual`, `registered`, `instant`, and raw `dteday`.
- Both training and user-supplied scenarios use the same `training_features()` feature contract.

## Evaluation Protocol

1. Sort records by calendar date and hour.
2. Split chronologically on complete dates using an 80% training-date / 20% holdout-date boundary.
3. Fit model and calculate benchmark statistics **using training records only**.
4. Evaluate all methods against the same later-date holdout observations.
5. Save the trained model, summary metrics, and row-level holdout predictions.

The verified split from Task 5 uses 13,891 training records ending on 2012-08-06 and 3,488 holdout records beginning on 2012-08-07. These counts are expected for the official dataset.

## Compared Methods

| Method | Definition |
|---|---|
| Training mean | Predict the overall mean of `cnt` from training observations for every holdout row |
| Training-hour mean | Predict the training-set average `cnt` for each hour; fall back to training global mean if an hour is unseen |
| XGBRegressor | Gradient-boosted regression trees trained on the ten-feature matrix |

The initial XGBoost configuration uses 200 trees, maximum tree depth 5, learning rate 0.05, subsample and column subsample 0.9, objective `reg:squarederror`, and random seed 42. These are fixed starter settings, **not** optimized hyperparameters.

## Metrics

- **MAE**: mean absolute difference between actual and predicted hourly rental counts. Lower is better.
- **RMSE**: square root of mean squared error. It penalizes large errors more heavily than MAE. Lower is better.
- **R²**: proportion of holdout variation explained relative to predicting the holdout mean; negative values are possible.
- **MAE reduction versus hourly benchmark**: percentage decrease in the model's MAE relative to the training-hour mean benchmark. Negative values mean the model performed worse.

Run `python -m src.train` locally to generate the actual metrics. No performance values are claimed before that execution.

## Generated Artifacts

| Path | Content | Version-controlled? |
|---|---|---|
| `models/demand_xgb.joblib` | Trained XGBoost model | No |
| `reports/model_metrics.json` | Split metadata, features, model and benchmark metrics | No |
| `reports/holdout_predictions.csv` | Holdout dates/hours, actuals, XGBoost predictions, hourly benchmark, and absolute errors | No |

These can be regenerated from the original CSV and source code. Only load model artifacts you created or obtained from trusted sources.

## Limitations and Next Stage

- The source represents the 2011–2012 historical bike-sharing network; results cannot be presented as current-demand forecasts.
- The model does not explicitly encode the yearly growth observed between 2011 and 2012.
- This is one chronological holdout, not rolling-origin evaluation.
- Comparing performance against an hourly benchmark is necessary but does not establish deployment readiness.
- Do not repeatedly tune to this holdout in Task 7. Use a separate chronological validation interval inside the training period for optimization, preserving the final holdout for an unbiased comparison.
- Task 7 will examine subgroup errors, large misses, feature usefulness, and measured model improvements.
