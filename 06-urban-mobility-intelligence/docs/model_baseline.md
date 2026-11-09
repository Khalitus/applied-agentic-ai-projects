# XGBoost Baseline Training and Evaluation

## Objective

Establish a reproducible regression benchmark for hourly bicycle rental demand (`cnt`) using the UCI Bike Sharing dataset (2011–2012). The system is a historical-pattern estimator, not a live operating forecast.

## Dataset and features

The model uses 17,379 hourly observations and ten features: month, hour, holiday, weekday, working-day status, weather category, normalized temperature, normalized humidity, normalized wind speed and an engineered commute-peak indicator. The target is `cnt`. Source identifier `instant`, date string `dteday`, and target-derived columns `casual`/`registered` are excluded from the model inputs.

Historical training and user-entered scenarios share the same feature-transformation contract.

## Chronological evaluation

The chronological holdout split is made on **complete calendar dates**, not randomly shuffled rows. The training period contains 13,891 records through 2012-08-06; the test period contains 3,488 records beginning 2012-08-07. Baseline statistics and the regressor are fitted on training observations only.

The fixed initial `XGBRegressor` settings are 200 trees, depth 5, learning rate 0.05, subsample 0.9, column subsample 0.9, squared-error objective, random seed 42 and two jobs. A global training-mean benchmark and an hour-of-day training-mean benchmark supply interpretable points of comparison.

## Recorded holdout results

| Model | MAE ↓ | RMSE ↓ | R² |
|---|---:|---:|---:|
| Global training mean | 174.971 | 232.559 | — |
| Training-hour mean | 113.678 | 167.319 | — |
| XGBoost baseline | **89.589** | **126.341** | **0.6717** |

The baseline regressor reduces MAE by **21.19%** compared with the training-hour mean. MAE and RMSE are measured in rentals per hour; R² is a holdout-fit metric, not a percentage accuracy score.

## Generated artifacts

| Path | Contents |
|---|---|
| `models/demand_xgb.joblib` | Serialized baseline regressor |
| `reports/model_metrics.json` | Split metadata and benchmark scores |
| `reports/holdout_predictions.csv` | Per-observation holdout actuals, predictions and errors |

These artifacts are generated locally, excluded from Git and regenerated with `python -m src.train`. Load only trusted serialized artifacts.

## Limitations

The source data spans 2011–2012 and is affected by changes in network usage between those years. The feature contract intentionally does not include year, which limits explicit modeling of secular growth. A single chronological holdout does not establish stability under future operating conditions. The holdout results were inspected before later optimization; accordingly, subsequent holdout comparisons should not be described as completely blind tests.
