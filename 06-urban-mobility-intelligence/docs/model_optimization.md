# XGBoost Optimization and Error Analysis

## Objective

Compare a limited set of regression configurations using past-only validation, document model errors and retain an independently identifiable optimized model artifact. The task is demand estimation under historical Capital Bikeshare patterns (2011–2012), not live forecasting.

## Evaluation protocol

The 17,379 hourly records are ordered chronologically and split by complete calendar dates. The outer training period contains 13,891 observations ending 2012-08-06; the later holdout contains 3,488 observations starting 2012-08-07. The training period is divided again into 11,083 fitting observations ending 2012-04-11 and 2,808 validation observations starting 2012-04-12.

Three configurations are compared on the inner validation window. A challenger replaces the baseline only when its **unrounded validation MAE** is more than 0.5% lower. The selected configuration is refitted on the complete outer training period, then evaluated against the later holdout. The final holdout was inspected in the preceding baseline analysis, so it is **not** a completely untouched test set; no subsequent tuning should use these holdout metrics as a selection target.

## Candidate configurations

| Candidate | Trees | Depth | Learning rate | Additional settings |
|---|---:|---:|---:|---|
| `baseline_xgboost` | 200 | 5 | 0.05 | subsample 0.9, colsample_bytree 0.9 |
| `shallow_regularized` | 300 | 4 | 0.05 | subsample 0.9, colsample_bytree 0.9, min_child_weight 3, reg_lambda 3 |
| `deeper_regularized` | 250 | 6 | 0.05 | subsample 0.85, colsample_bytree 0.9, min_child_weight 2, reg_lambda 2 |

All models use the `reg:squarederror` objective and seed 42; negative predictions are clipped to zero before scoring. The search is intentionally limited rather than exhaustive.

## Recorded validation results

| Candidate | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| `baseline_xgboost` | 85.709 | 118.601 | 0.7003 |
| `shallow_regularized` | 86.978 | 120.465 | 0.6908 |
| **`deeper_regularized`** | **85.248** | **116.786** | **0.7094** |

The deeper regularized configuration improves validation MAE by approximately **0.54%** against the baseline, passing the predefined 0.5% threshold.

### Engineered feature ablation

Under the selected hyperparameters, validation MAE was 85.248 **with** the engineered `commute_peak` indicator and 85.183 **without** it. The difference is small and does not support claiming that the indicator improves validation MAE. The ten-feature application contract was retained; the ablation is diagnostic only. Feature importance is not equivalent to incremental predictive value or causality.

## Recorded holdout comparison

| Measure | Baseline model | Selected model |
|---|---:|---:|
| MAE ↓ | 89.589 | **88.445** |
| RMSE ↓ | 126.341 | **124.719** |
| R² ↑ | 0.6717 | **0.6801** |
| MAE reduction versus training-hour mean | 21.19% | **22.20%** |

The optimized model reduces holdout MAE by approximately **1.28%** relative to the baseline model. This is a modest measured improvement, not evidence of production readiness or robustness under live conditions.

## Error analysis

`src/optimize.py` records validation errors by hour, weather category, working-day status and observed demand band, plus the ten largest validation misses. Signed bias is `predicted - actual` (positive means overprediction). Demand bands are evaluation-only labels and are not included as model inputs. Each subgroup must be interpreted with its observation count, especially the rare weather category 4.

## Saved artifacts

| Artifact | Purpose |
|---|---|
| `models/demand_xgb_optimized.joblib` | Selected model fitted on the complete outer training period |
| `reports/optimization_report.json` | Validation scores, selection criteria, ablation, error analysis and holdout comparison |
| `reports/validation_predictions.csv` | Row-level validation predictions and errors |
| `reports/optimized_holdout_predictions.csv` | Row-level holdout predictions and errors |

The original baseline artifacts are preserved. All generated models and reports are excluded from version control. Run `python -m src.optimize` to regenerate the optimized artifact and reports.
