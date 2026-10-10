import numpy as np
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["policy_start_date"] = pd.to_datetime(
        df["policy_start_date"]
    )

    df["policy_tenure_days"] = (
        df["incident_date"] - df["policy_start_date"]
    ).dt.days

    df["repair_value_ratio"] = (
        df["repair_estimate"]
        / df["vehicle_value"].replace(0, np.nan)
    )

    df["mileage_age_ratio"] = (
        df["annual_mileage"]/(df["vehicle_age"]+1)
    )

    df["has_prior_claim"] = (
        df["prior_claims"] > 0).astype(int)

    df["incident_month"]=(
        df["incident_date"].dt.month)

    df["incident_quarter"]=(
        df["incident_date"].dt.quarter)

    df["incident_weekday"] = (
        df["incident_date"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["incident_weekday"] >= 5).astype(int)

    df["driver_age_band"] = pd.cut(
        df["driver_age"],
        bins=[17, 24, 34, 49, 64, np.inf],
        labels=["18-24", "25-34", "35-49", "50-64", "65+"],
    ).astype("object")

    df["vehicle_age_band"] = pd.cut(
        df["vehicle_age"],
        bins=[-1, 2, 5, 10, np.inf],
        labels=["new", "recent", "mature", "old"]
    ).astype("object")


    df = df.drop(
        columns=["incident_date", "policy_start_date"]
    )

    return df