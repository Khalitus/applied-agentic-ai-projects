"""Train and evaluate a reproducible hourly bike-demand regression baseline."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import FEATURES, MODEL_PATH, PREDICTIONS_PATH, REPORT_PATH, TARGET
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

    if train.empty or test.empty or train["dteday"].max() >= test["dteday"].min():
        raise ValueError("Chronological split requires non-overlapping dates")

    return train, test


def baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build global-mean and hour-of-day baselines using training data only."""
    overall_mean = float(train[TARGET].mean())
    hourly_means = train.groupby("hr")[TARGET].mean()

    global_baseline = np.full(len(test), overall_mean)
    hourly_baseline = (
        test["hr"].map(hourly_means).fillna(overall_mean).to_numpy(dtype=float)
    )
    return global_baseline, hourly_baseline


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Report regression errors in hourly rental-count units."""
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
    }


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

    predictions = np.maximum(model.predict(x_test), 0.0)
    global_baseline, hourly_baseline = baseline_predictions(train, test)

    model_scores = regression_metrics(y_test, predictions)
    global_scores = regression_metrics(y_test, global_baseline)
    hourly_scores = regression_metrics(y_test, hourly_baseline)
    hourly_mae = mean_absolute_error(y_test, hourly_baseline)
    model_mae = mean_absolute_error(y_test, predictions)

    metrics = {
        "dataset": "UCI Bike Sharing (hour.csv, 2011–2012)",
        "evaluation": "chronological holdout by complete calendar dates",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": str(train["dteday"].iloc[-1]),
        "test_start": str(test["dteday"].iloc[0]),
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "model": "XGBRegressor",
        "model_mae": model_scores["mae"],
        "model_rmse": model_scores["rmse"],
        "model_r2": round(float(r2_score(y_test, predictions)), 4),
        "mean_baseline_mae": global_scores["mae"],
        "mean_baseline_rmse": global_scores["rmse"],
        "hourly_baseline_mae": hourly_scores["mae"],
        "hourly_baseline_rmse": hourly_scores["rmse"],
        "mae_reduction_vs_hourly_pct": round(
            100 * (hourly_mae - model_mae) / hourly_mae, 2
        ) if hourly_mae > 0 else None,
    }

    holdout = test.loc[:, ["instant", "dteday", "hr", TARGET]].copy()
    holdout = holdout.rename(columns={TARGET: "actual_rentals"})
    holdout["predicted_rentals"] = predictions
    holdout["hourly_baseline_rentals"] = hourly_baseline
    holdout["absolute_error"] = np.abs(
        holdout["actual_rentals"].to_numpy() - predictions
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    holdout.to_csv(PREDICTIONS_PATH, index=False, float_format="%.4f")
    REPORT_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {REPORT_PATH}")
    print(f"Holdout predictions saved: {PREDICTIONS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
