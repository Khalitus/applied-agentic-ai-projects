"""Pre-contact features only; see docs/data_card.md for leakage audit."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "balance", "pdays", "previous"]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan", "poutcome"
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "y"

# 'duration' is known only after a call, and 'campaign' includes current-call data.
# We exclude current-contact timing/channel fields as well.
EXCLUDED = ["duration", "campaign", "contact", "day", "month"]

JOB_CHOICES = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL_CHOICES = ["married", "single", "divorced"]
EDUCATION_CHOICES = ["primary", "secondary", "tertiary", "unknown"]
POUTCOME_CHOICES = ["unknown", "failure", "other", "success"]


def prepare_features(frame: pd.DataFrame):
    """Select the prediction contract and validate source features and labels."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate source columns")
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    X = frame[FEATURES].copy()
    y = frame[TARGET].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Unrecognized target label")
    if X.isna().any().any():
        raise ValueError("Missing prediction feature values")
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in NUMERIC):
        raise ValueError("Numerical features must have numeric dtypes")
    if not np.isfinite(X[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numerical feature values")
    if not all(X[col].map(lambda value: isinstance(value, str)).all() for col in CATEGORICAL):
        raise ValueError("Categorical features must contain strings")
    return X, y.astype(int)


def split_dataset(X: pd.DataFrame, y: pd.Series):
    """Return aligned train, validation and holdout pairs in source order."""
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Feature and target indices must be aligned and unique")
    n = len(X)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    if not 0 < train_end < valid_end < n:
        raise ValueError("Dataset too small for nonempty chronological splits")
    return tuple(
        (X.iloc[start:end].copy(), y.iloc[start:end].copy())
        for start, end in ((0, train_end), (train_end, valid_end), (valid_end, n))
    )


def build_preprocessor():
    """Create a fresh, unfitted preprocessing transformer."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
