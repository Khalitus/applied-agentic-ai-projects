"""Time-ordered baseline model. Do not tune against the final test partition."""

import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import RAW_CSV, MODEL, METRICS
from src.features import EXCLUDED, FEATURES, build_preprocessor, prepare_features, split_dataset


def build_pipeline():
    return Pipeline([
        ("preprocessing", build_preprocessor()),
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
    if len(frame) < 100:
        raise ValueError("Dataset too small for these chronological splits")
    X, y = prepare_features(frame)
    (X_train, y_train), (X_val, y_val), _ = split_dataset(X, y)
    if any(labels.nunique() != 2 for labels in (y_train, y_val)):
        raise ValueError("A development split contains only one class")
    model = build_pipeline().fit(X_train, y_train)
    metrics = {
        "source": "UCI Bank Marketing - bank-full.csv",
        "split": "source row order; 70% train / 15% validation / 15% test",
        "features": FEATURES,
        "excluded_current_contact_features": EXCLUDED,
        "prediction_threshold": 0.5,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train": evaluate(model, X_train, y_train),
        "validation": evaluate(model, X_val, y_val),
        "holdout_status": "reserved; not evaluated",
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": metrics["validation"], "holdout_status": metrics["holdout_status"]}, indent=2))
    print(f"Saved model: {MODEL}")
    print(f"Saved metrics: {METRICS}")


if __name__ == "__main__":
    main()
