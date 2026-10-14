import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    TARGET_COLUMN,
)
from src.features import create_features
from src.predict import load_selected_model
from src.train_baseline import FEATURES as RAW_FEATURES


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"
REPORT_PATH = ROOT_DIR / "reports" / "final_model_evaluation.json"

def load_test_data():
    claims = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    if claims["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs.")

    if splits["claim_id"].duplicated().any():
        raise ValueError("Duplicate split IDs.")

    if set(claims["claim_id"]) != set(splits["claim_id"]):
        raise ValueError("Dataset and split IDs differ.")

    data = claims.merge(
        splits,
        on="claim_id",
        validate="one_to_one",
    )

    test = data.loc[data["split"] == "test"].copy()

    if len(test) != 180:
        raise ValueError("Expected 180 held-out test cases.")

    return test

def prepare_test_features(test: pd.DataFrame, mode: str):
    excluded = ["claim_id", TARGET_COLUMN, "split"]

    raw = test.drop(columns=excluded)

    if mode == "engineered":
        return create_features(raw)

    if mode == "raw":
        return raw[RAW_FEATURES]

    raise ValueError(f"Unsupported feature mode: {mode}")

def evaluate_final_model():
    test = load_test_data()

    model, selection = load_selected_model()

    features = prepare_test_features(
        test,
        selection["feature_mode"],
    )

    expected_columns = list(model.feature_names_in_)

    if set(features.columns) != set(expected_columns):
        raise ValueError("Test feature schema mismatch.")

    features = features[expected_columns]
    actual = test[TARGET_COLUMN].to_numpy()

    model_predictions = model.predict(features)

    mae = mean_absolute_error(actual, model_predictions)

    rmse = np.sqrt(mean_squared_error(actual, model_predictions))

    r2 = r2_score(actual, model_predictions)

    return {
        "model_name": selection["selected_model"],
        "test_rows": len(test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4), 
    }

def main():
    if REPORT_PATH.exists():
        raise FileExistsError(
            "Final test report already exists. "
            "Review the saved report instead of rerunning "
            "the test to guide development."
        )

    results = evaluate_final_model()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nFinal held-out model evaluation")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()