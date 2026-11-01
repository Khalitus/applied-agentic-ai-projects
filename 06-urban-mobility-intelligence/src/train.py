"""First reproducible regression baseline; improvement is a project task."""
import json
from math import sqrt

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import FEATURES, MODEL_PATH, REPORT_PATH, TARGET
from src.data import load_hourly
from src.features import training_features


def main() -> dict:
    df = load_hourly()
    split = int(len(df) * 0.80)
    train, test = df.iloc[:split], df.iloc[split:]
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
