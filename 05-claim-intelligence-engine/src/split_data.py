import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, ROOT_DIR


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"


def create_splits():
    claims = pd.read_csv(DATA_PATH)

    if claims["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs found.")

    #Training 70%, remaining 30%
    train, remaining = train_test_split(
        claims,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=claims["damage_severity"],
    )

    #Remaining 30% divided equally
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remaining["damage_severity"],
    )

    splits = pd.concat(
        [
            train[["claim_id"]].assign(split="train"),
            validation[["claim_id"]].assign(split="validation"),
            test[["claim_id"]].assign(split="test"),
        ],
        ignore_index=True,
    )

    splits = splits.sort_values("claim_id")

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLIT_PATH, index=False)

    print("\nDataset split complete")
    print(splits["split"].value_counts())
    print(f"\nSaved to: {SPLIT_PATH}")


if __name__ == "__main__":
    create_splits()