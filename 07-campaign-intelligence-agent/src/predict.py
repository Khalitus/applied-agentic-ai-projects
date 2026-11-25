"""Server-side inference using the same pipeline used during training."""

from functools import lru_cache
import joblib
import pandas as pd

from src.config import MODEL
from src.features import FEATURES
from src.schema import CustomerInput


@lru_cache(maxsize=1)
def load_model():
    if not MODEL.exists():
        raise FileNotFoundError("No trained model: run python -m src.ml first")
    # Only load artifacts created locally by our training script. joblib is pickle-based.
    return joblib.load(MODEL)


def predict_subscription(payload: dict):
    validated = CustomerInput.model_validate(payload)
    row = pd.DataFrame([validated.model_dump()])[FEATURES]
    probability = float(load_model().predict_proba(row)[0, 1])
    return {
        "probability": round(probability, 5),
        "threshold": 0.5,
        "above_default_threshold": probability >= 0.5,
        "interpretation": "Historical-model estimate, not an individual guarantee",
    }
