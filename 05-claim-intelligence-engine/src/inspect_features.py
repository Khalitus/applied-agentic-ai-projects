import pandas as pd

from src.config import PROCESSED_DATA_DIR, ROOT_DIR
from src.features import create_features


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"

ENGINEERED_FEATURES = [
    "policy_tenure_days",
    "repair_value_ratio",
    "mileage_age_ratio",
    "has_prior_claim",
    "incident_month",
    "incident_quarter",
    "incident_weekday",
    "is_weekend",
    "driver_age_band",
    "vehicle_age_band",
]


def main():
    claims = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    data = claims.merge(
        splits,
        on="claim_id",
        validate="one_to_one",
    )

    train = data[data["split"] == "train"]

    X_train = train.drop(
        columns=["claim_id", "claim_amount", "split"]
    )

    engineered = create_features(X_train)

    print("\nFeature engineering summary")
    print(f"Training rows: {len(engineered)}")
    print(f"Total features: {engineered.shape[1]}")

    missing_features = (
        set(ENGINEERED_FEATURES) - set(engineered.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing engineered features: {missing_features}"
        )

    forbidden = {
        "claim_id",
        "claim_amount",
        "split",
        "incident_date",
        "policy_start_date",
    }

    if forbidden.intersection(engineered.columns):
        raise ValueError("Unexpected columns in model features.")

    print("\nEngineered feature preview")
    print(
        engineered[ENGINEERED_FEATURES]
        .head()
        .to_string(index=False)
    )

    print("\nEngineered feature types")
    print(engineered[ENGINEERED_FEATURES].dtypes)

    print("\nMissing values")
    print(engineered.isna().sum())

    single_row = create_features(X_train.head(1))

    if list(single_row.columns) != list(engineered.columns):
        raise ValueError(
            "Single-row and batch feature schemas do not match."
        )

    print("\nSingle-row feature schema verified")


if __name__ == "__main__":
    main()