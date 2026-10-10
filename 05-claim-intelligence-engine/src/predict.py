import json
import math

import joblib
import pandas as pd

from src.config import MODEL_DIR, ROOT_DIR
from src.features import create_features
from src.train_baseline import (
    CATEGORICAL_FEATURES,
    FEATURES as RAW_FEATURES,
    NUMERIC_FEATURES,
)


SELECTION_PATH = MODEL_DIR / "model_selection.json"

DATE_FIELDS = [
    "incident_date",
    "policy_start_date",
]

REQUIRED_FIELDS = RAW_FEATURES + DATE_FIELDS

def validate_claim(claim: dict) -> pd.DataFrame:
    if not isinstance(claim, dict):
        raise TypeError("Claim input must be a dictionary.")

    missing = set(REQUIRED_FIELDS) - set(claim)

    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )

    df = pd.DataFrame([claim])[REQUIRED_FIELDS].copy()

    for column in NUMERIC_FEATURES:
        if isinstance(df.at[0, column], bool):
            raise ValueError(f"{column} must be numeric.")

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        value = df.at[0, column]

        if not math.isfinite(value):
            raise ValueError(
                f"{column} must be a finite number."
            )

    if df.at[0, "driver_age"] < 18:
        raise ValueError("Driver must be at least 18.")

    if df.at[0, "vehicle_value"] <= 0:
        raise ValueError("Vehicle value must be positive.")

    for column in [
        "vehicle_age",
        "annual_mileage",
        "repair_estimate",
        "prior_claims",
    ]:
        if df.at[0, column] < 0:
            raise ValueError(
                f"{column} cannot be negative."
            )

    for column in [
        "driver_age",
        "vehicle_age",
        "prior_claims",
    ]:
        if not float(df.at[0, column]).is_integer():
            raise ValueError(
                f"{column} must be a whole number."
            )

        df[column] = df[column].astype(int)

    for column in CATEGORICAL_FEATURES:
        value = df.at[0, column]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{column} must be a non-empty string."
            )

        df.at[0, column] = value.strip().lower()

    if df.at[0, "damage_severity"] not in {
        "minor", "moderate", "severe"
    }:
        raise ValueError("Invalid damage severity.")

    if df.at[0, "police_reported"] not in {"yes", "no"}:
        raise ValueError("police_reported must be yes or no.")

    for column in DATE_FIELDS:
        try:
            parsed = pd.to_datetime(
                df.at[0, column],
                format="%Y-%m-%d",
                errors="raise",
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{column} must be a valid YYYY-MM-DD date."
            ) from exc

        df.at[0, column] = parsed.strftime("%Y-%m-%d")

    if df.at[0, "policy_start_date"] > df.at[0, "incident_date"]:
        raise ValueError(
            "Policy start date cannot follow incident date."
        )

    return df

def load_selected_model():
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            "Model selection file not found. "
            "Run python -m src.train first."
        )

    selection = json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )

    feature_mode = selection["feature_mode"]

    if feature_mode not in {"raw", "engineered"}:
        raise ValueError("Invalid feature mode.")

    model_path = (
        ROOT_DIR / selection["model_path"]
    ).resolve()

    if not model_path.is_relative_to(MODEL_DIR.resolve()):
        raise ValueError("Invalid model path in selection file.")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Selected model not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model, selection

def predict_claim(claim: dict) -> dict:
    validated = validate_claim(claim)

    model, selection = load_selected_model()

    feature_mode = selection["feature_mode"]

    if feature_mode == "engineered":
        features = create_features(validated)
    else:
        features = validated[RAW_FEATURES]

    if all(col in features.columns for col in model.feature_names_in_):
        features_ordered = features[model.feature_names_in_]
        prediction = float(model.predict(features_ordered)[0])
    else:
        missing_cols = [col for col in model.feature_names_in_ if col not in features.columns]
        raise ValueError(f"Features DataFrame is missing expected model columns: {missing_cols}")
        

    if not math.isfinite(prediction):
        raise ValueError("Model returned an invalid prediction.")

    if prediction < 0:
        raise ValueError(
            "Model returned a negative settlement estimate."
        )
    
    return {
        "predicted_amount": round(prediction, 2),
        "selected_model": selection["selected_model"],
        "feature_mode": feature_mode,
    }

def main():
    new_claim = {
        "incident_date": "2026-09-15",
        "policy_start_date": "2024-06-01",
        "driver_age": 34,
        "vehicle_age": 5,
        "annual_mileage": 12000,
        "vehicle_value": 28000,
        "repair_estimate": 7800,
        "prior_claims": 1,
        "incident_type": "collision",
        "vehicle_type": "sedan",
        "weather": "clear",
        "region": "north",
        "police_reported": "yes",
        "damage_severity": "moderate",
    }

    result = predict_claim(new_claim)
    
    print("\nClaim prediction")
    print(json.dumps(result, indent=2)) 


if __name__ == "__main__":
    main()