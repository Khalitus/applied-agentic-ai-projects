"""First reproducible regression baseline; improvement is a project task."""
import json
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import FEATURES, MODEL_PATH, REPORT_PATH, TARGET
from src.data import load_hourly
from src.features import training_features

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    ordered = df.sort_values(["dteday", "hr"], kind="stable")
    dates = ordered["dteday"].unique()

    boundary = int(len(dates) * train_fraction)

    if boundary == 0 or boundary == len(dates):
        raise ValueError("Not enough unique dates for this split")

    cutoff_date = dates[boundary]

    train = ordered.loc[ordered["dteday"] < cutoff_date].copy()
    test = ordered.loc[ordered["dteday"] >= cutoff_date].copy()

    if train["dteday"].max() >= test["dteday"].min():
        raise ValueError("Chronological split contains overlapping dates")

    return train, test

def main() -> dict:
    df = load_hourly()
    train, test = chronological_split(df)
    x_train, x_test = training_features(train), training_features(test)
    y_train, y_test = train[TARGET], test[TARGET]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    preds = np.clip(model.predict(x_test), 0, None)
    baseline = np.full(len(test), y_train.mean())
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": str(train["dteday"].iloc[-1]),
        "test_start": str(test["dteday"].iloc[0]),
        "features": list(FEATURES),
        "model_mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "model_rmse": round(sqrt(float(mean_squared_error(y_test, preds))), 3),
        "mean_baseline_mae": round(float(mean_absolute_error(y_test, baseline)), 3),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
