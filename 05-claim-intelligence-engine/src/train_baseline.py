import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    ROOT_DIR,
    TARGET_COLUMN,
)


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"

MODEL_PATH = MODEL_DIR / "claim_baseline.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"


NUMERIC_FEATURES = [
    "driver_age",
    "vehicle_age",
    "annual_mileage",
    "vehicle_value",
    "repair_estimate",
    "prior_claims",
]

CATEGORICAL_FEATURES = [
    "incident_type",
    "vehicle_type",
    "weather",
    "region",
    "police_reported",
    "damage_severity",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def load_split_data():
    claims = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    if set(claims["claim_id"]) != set(splits["claim_id"]):
        raise ValueError("Dataset and split IDs do not match.")

    if splits["claim_id"].duplicated().any():
        raise ValueError("Duplicate IDs found in split file.")

    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split labels.")

    data = claims.merge(
        splits,
        on="claim_id",
        validate="one_to_one",
    )

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]

    X_train = train[FEATURES]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET_COLUMN]

    return X_train, X_validation, y_train, y_validation

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

def build_models():
    dummy = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )

    xgboost = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                ),
            ),
        ]
    )

    return dummy, xgboost

def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    r2 = r2_score(y_true, predictions)

    return {
        "mae": round(float(mae), 2),

        "rmse": round(float(rmse), 2),

        "r2": round(float(r2), 2),
    }

def main():
    X_train, X_validation, y_train, y_validation = load_split_data()

    dummy, xgboost = build_models()

    print("\nTraining dummy model")
    dummy.fit(X_train, y_train)

    dummy_predictions = dummy.predict(X_validation)
    dummy_metrics = evaluate_model(y_validation, dummy_predictions)

    print("\nTraining XGBoost")
    xgboost.fit(X_train, y_train)

    xgb_predictions = xgboost.predict(X_validation)
    xgb_metrics = evaluate_model(y_validation, xgb_predictions)

    print("\nValidation results")
    print(f"Dummy: {dummy_metrics}")
    print(f"XGBoost: {xgb_metrics}")

    actuals = y_validation.to_numpy()
    preds = xgb_predictions

    comparison = pd.DataFrame(
        {
            "actual_amount": actuals,
            "predicted_amount": preds,
            "absolute_error": np.abs(actuals - preds),
        }
    )

    print("\nSample predictions")
    print(comparison.head(10).to_string(index=False))

    results = {
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "features": FEATURES,
        "dummy": dummy_metrics,
        "xgboost": xgb_metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgboost, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()