# XGBoost Optimization and Error Analysis

## Purpose

Task 7 evaluates a small, predefined set of XGBoost configurations on a **chronological validation window** and audits model errors. The source is the historical UCI Bike Sharing `hour.csv` dataset (2011–2012). The objective is an educational scenario estimator, not a live demand forecast.

## Task 6 Reference

The unoptimized Task 6 XGBoost baseline used ten model features, including `commute_peak`, and an outer date-based split with 13,891 training observations ending 2012-08-06 and 3,488 holdout observations beginning 2012-08-07.

| Task 6 metric | Measured value |
|---|---:|
| XGBoost MAE | 89.589 |
| XGBoost RMSE | 126.341 |
| XGBoost R² | 0.6717 |
| Training-hour mean MAE | 113.678 |
| Improvement over hourly baseline (MAE) | 21.19% |

These are previously observed results, **not** selection scores for Task 7.

## Selection Protocol

1. Preserve the Task 6 outer chronological split: 80% of unique dates for all training/validation work, and 20% of dates as the final holdout.
2. Split the earlier 80%-date training period chronologically again: 80% of its dates for candidate fitting, 20% for model selection.
3. Train all candidates using the same ten features and evaluate them on the same internal validation period.
4. Select the candidate with the lowest **unrounded validation MAE**, but keep the Task 6 baseline parameters unless a challenger improves MAE by **more than 0.5%**. This threshold is chosen before inspecting Task 7 outcomes.
5. Run a diagnostic ablation for `commute_peak` using the selected hyperparameters. This does **not** alter the fixed ten-feature contract or the chosen hyperparameters.
6. Fit the selected ten-feature configuration on the complete earlier outer-training period. Evaluate once on the later holdout and save a **separate** optimized artifact.

Both split boundaries are complete calendar dates; no test-period rows enter model selection. The held-out period has already been inspected in Task 6, so this evaluation should not be described as a completely untouched, unbiased final test. Do not continue tuning candidates against its results.

## Candidate Configurations

| Candidate | Trees | Depth | Learning rate | Other settings |
|---|---:|---:|---:|---|
| `task6_baseline` | 200 | 5 | 0.05 | subsample 0.9, colsample 0.9 |
| `shallow_regularized` | 300 | 4 | 0.05 | subsample 0.9, colsample 0.9, min_child_weight 3, reg_lambda 3 |
| `deeper_regularized` | 250 | 6 | 0.05 | subsample 0.85, colsample 0.9, min_child_weight 2, reg_lambda 2 |

All candidates use `reg:squarederror`, `random_state=42`, and `n_jobs=2`. Predicted counts are clipped at zero before error measurement, matching Task 6. This is a modest controlled comparison, **not** an exhaustive hyperparameter search.

## Validation Error Analysis

`src/optimize.py` records row-level predictions and aggregates error for:

- **Hour**: where demand patterns are hardest for the model to predict.
- **Working-day status**: differing behavior on workdays and non-workdays.
- **Weather category**: possible uneven performance, with an explicit need to consider sample sizes.
- **Actual demand level**: low (0–99), medium (100–299), and high (300+) rentals per hour.
- **Largest individual misses**: ten highest-absolute-error validation observations for inspection.

Metrics include MAE, **signed bias** (`predicted - actual`), average actual demand and average prediction. Positive signed bias means overprediction, and negative means underprediction. Actual demand bands are used **only for evaluation**, never as model inputs.

Feature importance values from the final tree model are descriptive split-based model statistics; they do not establish causality or the incremental usefulness of correlated features. The `commute_peak` ablation is an additional diagnostic, not a basis for modifying the ten-column feature contract during this task.

## Running Task 7

From the activated project environment:

```bash
python -m pytest -q
python -m src.optimize
```

Read `reports/optimization_report.json` after running to determine actual selected parameters and performance; those results are not known in advance.

## Generated Artifacts

| File | Purpose |
|---|---|
| `models/demand_xgb_optimized.joblib` | Selected model refitted on the original outer-training period |
| `reports/optimization_report.json` | Candidate scores, selection decision, ablation, subgroup error profile, feature importance and holdout comparison |
| `reports/validation_predictions.csv` | Row-level predictions for the inner validation period |
| `reports/optimized_holdout_predictions.csv` | Row-level predictions for the later outer holdout |

These model and report artifacts are ignored by Git. The Task 6 files (`models/demand_xgb.joblib`, `reports/model_metrics.json`, `reports/holdout_predictions.csv`) are not overwritten. Accordingly, `src/predict.py` and the Gradio app continue using the Task 6 model until a deliberate decision is made in Task 8.

## Limitations and Next Step

- Chronological validation is more appropriate than randomized shuffling but is still **one** internal split, not rolling-origin cross-validation.
- Monthly seasonality and the substantial 2011–2012 demand increase can make an inner validation period different from the later holdout.
- No explicit year feature is included; this constrains how well growth patterns can be represented.
- The dataset contains only three category-4 weather observations in total, so category-specific claims must respect small sample sizes.
- The final holdout was observed during Task 6. Do not use Task 7's holdout comparison to launch another tuning round without a new evaluation strategy.
- Task 8 will decide whether to activate the optimized artifact in the user-input prediction path based on validation results, reliability and measured performance, while preserving an accurate description of limitations.
