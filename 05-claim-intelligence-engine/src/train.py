import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from src.features import create_features
from src.train_baseline import evaluate_model


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"

BASELINE_METRICS_PATH = MODEL_DIR / "baseline_metrics.json"
BASELINE_MODEL_PATH = MODEL_DIR / "claim_baseline.joblib"

ENGINEERED_MODEL_PATH = MODEL_DIR / "claim_engineered.joblib"
ENGINEERED_METRICS_PATH = MODEL_DIR / "engineered_metrics.json"
SELECTION_PATH = MODEL_DIR / "model_selection.json"

def load_data():
    claims = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    if claims["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs in dataset.")

    if splits["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs in split file.")

    if set(claims["claim_id"]) != set(splits["claim_id"]):
        raise ValueError("Dataset and split IDs do not match.")

    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected dataset split labels.")

    data = claims.merge(
        splits,
        on="claim_id",
        validate="one_to_one",
    )

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]

    excluded = ["claim_id", TARGET_COLUMN, "split"]

    X_train = train.drop(columns=excluded)
    X_validation = validation.drop(columns=excluded)

    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]

    return X_train, X_validation, y_train, y_validation

def prepare_features(X_train, X_validation):
    train_features = create_features(X_train)
    validation_features = create_features(X_validation)

    if list(train_features.columns) != list(validation_features.columns):
        raise ValueError("Train/validation feature schemas do not match.")

    return train_features, validation_features

def build_preprocessor(X_train):
    numeric_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

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
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

def build_model(X_train):
    preprocessor = build_preprocessor(X_train)

    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,

        subsample = 0.85,
        colsample_bytree = 0.85,

        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

def load_baseline_metrics():
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Run python -m src.train_baseline first."
        )

    results = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )

    return results["xgboost"]

def select_model(baseline_metrics, engineered_metrics):
    baseline_mae = baseline_metrics["mae"]
    engineered_mae = engineered_metrics["mae"]

    if engineered_mae < baseline_mae:
        return "engineered"
    else:
        return "baseline"
    
def main():
    X_train, X_validation, y_train, y_validation = load_data()

    X_train, X_validation = prepare_features(
        X_train,
        X_validation,
    )

    print("\nEngineered model training")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")
    print(f"Input features: {X_train.shape[1]}")

    model = build_model(X_train)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    metrics = evaluate_model(
        y_validation,
        predictions,
    )

    baseline_metrics = load_baseline_metrics()

    comparison = pd.DataFrame(
        {
            "Baseline": baseline_metrics,
            "Engineered": metrics,
        }
    )

    preprocessor = model.named_steps["preprocessing"]
    xgb = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 transformed features")
    print(importance.head(10).to_string(index=False))

    print("\nModel comparison")
    print(comparison)
    print("\nEngineered model metrics")
    print(metrics)

    baseline_mae = baseline_metrics["mae"]
    engineered_mae = metrics["mae"]

    improvement_percentage = ((baseline_mae - engineered_mae)/baseline_mae)*100

    print(f"\nMAE improvement: {improvement_percentage:.2f}%")

    selected = select_model(
        baseline_metrics,
        metrics,
    )

    selected_path = (
        ENGINEERED_MODEL_PATH
        if selected == "engineered"
        else BASELINE_MODEL_PATH
    )

    print(f"\nSelected model: {selected}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        ENGINEERED_MODEL_PATH,
    )

    ENGINEERED_METRICS_PATH.write_text(
        json.dumps(
            {
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "features": X_train.columns.tolist(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    selection = {
        "selected_model": selected,
        "feature_mode": (
            "engineered" if selected == "engineered" else "raw"
        ),
        "model_path": str(selected_path.relative_to(ROOT_DIR)),
        "selection_metric": "validation_mae",
        "baseline_mae": baseline_metrics["mae"],
        "engineered_mae": metrics["mae"],
        "test_evaluated": False,
    }

    SELECTION_PATH.write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    print(f"\nEngineered model saved: {ENGINEERED_MODEL_PATH}")
    print(f"Metrics saved: {ENGINEERED_METRICS_PATH}")
    print(f"Selection saved: {SELECTION_PATH}")

if __name__ == "__main__":
    main()