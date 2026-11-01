"""ML prediction callable both by the form and, later, the agent tool."""
import joblib

from src.config import MODEL_PATH
from src.features import scenario_features


def predict_demand(**scenario) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python -m src.train")
    model = joblib.load(MODEL_PATH)  # Load only artifacts created by this project.
    features = scenario_features(**scenario)
    return round(max(0.0, float(model.predict(features)[0])), 1)
