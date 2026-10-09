import pandas as pd

from src.config import CLAIMS_PATH, PROCESSED_DATA_DIR


CATEGORICAL_COLUMNS = [
    "incident_type",
    "vehicle_type",
    "weather",
    "region",
    "police_reported",
    "damage_severity",
]

NUMERIC_COLUMNS = [
    "driver_age",
    "vehicle_age",
    "annual_mileage",
    "vehicle_value",
    "repair_estimate",
    "prior_claims",
    "claim_amount",
]


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["claim_id"] = df["claim_id"].astype("string").str.strip()

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in ["incident_date", "policy_start_date"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


def validate_claims(df: pd.DataFrame) -> dict:
    issues = {
        "missing_values": int(df.isna().sum().sum()),

        "duplicate_claim_ids": int(
            df["claim_id"].duplicated().sum()
        ),

        "invalid_policy_dates": int(
            (
                df["policy_start_date"]
                > df["incident_date"]
            ).sum()
        ),

        "negative_repair_estimates": int(
            (df["repair_estimate"] < 0).sum()
        ),
        "invalid_driver_age": int(
            (df["driver_age"] < 18).sum()  
        ),
        "invalid_vehicle_value": int(
            (df["vehicle_value"] <= 0).sum()
        ),
        "invalid_damage_severity": int(
            (~df['damage_severity'].isin(['minor', 'moderate', 'severe'])).sum()
        ),
    }

    return issues


def main():
    claims = pd.read_csv(CLAIMS_PATH)

    print("\nRaw dataset")
    print(f"Rows: {len(claims)}")

    cleaned = clean_claims(claims)
    issues = validate_claims(cleaned)

    print("\nData quality report")

    for issue, count in issues.items():
        print(f"{issue}: {count}")

    if any(issues.values()):
        raise ValueError(
            "Data validation failed. Review the reported issues."
        )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PROCESSED_DATA_DIR / "claims_processed.csv"

    cleaned.to_csv(output_path, index=False)

    print("\nCleaning complete")
    print(f"Rows remaining: {len(cleaned)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()