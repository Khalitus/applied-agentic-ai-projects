"""Time-ordered baseline model. Do not tune against the final test partition."""

import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_CSV, MODEL, METRICS
from src.features import CATEGORICAL, FEATURES, NUMERIC, TARGET


def build_pipeline():
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
    ])


def evaluate(model, rows, labels):
    probabilities = model.predict_proba(rows)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "sample_count": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 5),
        "accuracy": round(float(accuracy_score(labels, predictions)), 5),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 5),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 5),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 5),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 5),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 5),
        "brier": round(float(brier_score_loss(labels, probabilities)), 5),
    }


def main():
    if not RAW_CSV.exists():
        raise SystemExit("Data missing: run python -m scripts.fetch_data")
    frame = pd.read_csv(RAW_CSV, sep=";")
    if not set(FEATURES + [TARGET]).issubset(frame.columns):
        raise ValueError("Unexpected source schema")
    if len(frame) < 100:
        raise ValueError("Dataset too small for these chronological splits")
    frame = frame.reset_index(drop=True)
    labels = frame[TARGET].map({"no": 0, "yes": 1})
    if labels.isna().any():
        raise ValueError("Unrecognized target label")

    # UCI documents bank-full.csv as ordered by date. There is no explicit date column.
    # Keep the original row order; do not shuffle or claim customer-level deduplication.
    train_end = int(len(frame) * 0.70)
    valid_end = int(len(frame) * 0.85)
    X, y = frame[FEATURES], labels
    if not all(y.iloc[a:b].nunique() == 2 for a, b in [
        (0, train_end), (train_end, valid_end), (valid_end, len(frame))
    ]):
        raise ValueError("A split contains only one class; evaluation cannot proceed")
    model = build_pipeline().fit(X.iloc[:train_end], y.iloc[:train_end])
    metrics = {
        "source": "UCI Bank Marketing - bank-full.csv",
        "split": "source row order; 70% train / 15% validation / 15% test",
        "features": FEATURES,
        "excluded_current_contact_features": ["duration", "campaign", "contact", "day", "month"],
        "prediction_threshold": 0.5,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train": evaluate(model, X.iloc[:train_end], y.iloc[:train_end]),
        "validation": evaluate(model, X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]),
        "test": evaluate(model, X.iloc[valid_end:], y.iloc[valid_end:]),
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": metrics["validation"], "test": metrics["test"]}, indent=2))
    print(f"Saved model: {MODEL}")
    print(f"Saved metrics: {METRICS}")


if __name__ == "__main__":
    main()
