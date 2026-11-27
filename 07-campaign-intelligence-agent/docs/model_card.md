# Model card — subscription probability baseline

## Overview

**Model:** scikit-learn `LogisticRegression` within a preprocessing `Pipeline`.

**Prediction target:** the `yes` class for historical term-deposit subscription (`y`) in the UCI Bank Marketing dataset.

**Inputs:** 11 customer and previous-campaign features assumed to be available before a new contact. Numeric features are standardized; categorical features are one-hot encoded with unseen categories ignored. The combined representation is sparse for the source dataset. Current-call and post-contact fields are excluded.

**Status:** implementation available; performance must be established by running the training and evaluation procedure on the original dataset. No performance figures are asserted in this document.

## Data and validation

- Source: UCI Bank Marketing, `bank-full.csv`, 45,211 historical records, CC BY 4.0.
- Split: first 70% of source rows for training, next 15% for validation, final 15% for holdout evaluation.
- Preprocessing: fitted only to the training partition.
- Default decision threshold: 0.5, not optimized for a specific operational cost.
- Metrics produced by the training command: accuracy, precision, recall, F1, average precision (reported as `pr_auc`), ROC-AUC, Brier score, and class prevalence for training and validation only. The holdout remains reserved until model selection is complete.
- Generated result file: `reports/baseline_metrics.json`.

## Intended use

Demonstrating a reproducible predictive pipeline and exploring statistical relationships in historical marketing data. The model is **not** approved for real customer targeting, financial decisions, or automated outreach.

## Risks and limitations

- The data is historical and geographically specific, limiting generalizability.
- Source order is not a substitute for explicit timestamps; repeated customers cannot be assessed without identifiers.
- Age and marital status require consideration of fairness, legal, and operational risks before any real use.
- Estimated probabilities have not been separately calibrated or assessed under deployment drift.
- A classification threshold expresses a decision convention, not certainty or causal impact.
- LLM-generated interpretation is optional and not independently verified.

## Reproducibility

Run `python -m src.ml` after downloading the official dataset. Review `reports/baseline_metrics.json` and the training environment before reporting model performance.
