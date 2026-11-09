"""Validated hourly demand inference and matched historical comparisons."""

import math
import sqlite3

import joblib

from src.analytics import historical_reference
from src.config import DB_PATH, FEATURES, OPTIMIZED_MODEL_PATH
from src.features import scenario_features


MIN_REFERENCE_OBSERVATIONS = 30


def _predict_features(features) -> float:
    if not OPTIMIZED_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Optimized model unavailable. Run: python -m src.optimize"
        )

    # Serialized models must come from trusted, locally generated artifacts.
    model = joblib.load(OPTIMIZED_MODEL_PATH)
    if list(getattr(model, "feature_names_in_", [])) != list(FEATURES):
        raise RuntimeError("Model feature contract mismatch. Retrain the optimized model.")

    predictions = model.predict(features)
    if len(predictions) != 1 or not math.isfinite(float(predictions[0])):
        raise RuntimeError("Model returned an invalid prediction")

    return round(max(0.0, float(predictions[0])), 1)


def predict_demand(**scenario) -> float:
    """Return a nonnegative hourly rental estimate for a validated scenario."""
    return _predict_features(scenario_features(**scenario))


def estimate_scenario(**scenario) -> dict:
    """Combine model inference with matching 2011–2012 SQL observations."""
    features = scenario_features(**scenario)
    prediction = _predict_features(features)

    hour = int(features.iloc[0]["hr"])
    workingday = int(features.iloc[0]["workingday"])
    weather_code = int(features.iloc[0]["weathersit"])

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            "Historical database unavailable. Run: python -m src.db"
        )

    try:
        historical = historical_reference(
            hour=hour,
            workingday=workingday,
            weathersit=weather_code,
            path=DB_PATH,
        )
    except sqlite3.Error as exc:
        raise RuntimeError(
            "Historical database cannot be queried. Rebuild it with: python -m src.db"
        ) from exc

    observations = int(historical["observations"])
    if observations == 0:
        historical = {
            **historical,
            "avg_rentals": None,
            "min_rentals": None,
            "max_rentals": None,
        }

    return {
        "estimated_rentals": prediction,
        "model": "optimized_xgboost",
        "historical_period": "2011–2012",
        "scenario": {
            "date": scenario["date_text"],
            "hour": hour,
            "holiday": bool(scenario["holiday"]),
            "workingday": workingday,
            "weather_code": weather_code,
        },
        "historical_reference": historical,
        "reference_sample": (
            "no_matches" if observations == 0
            else "small_sample" if observations < MIN_REFERENCE_OBSERVATIONS
            else "available"
        ),
    }
