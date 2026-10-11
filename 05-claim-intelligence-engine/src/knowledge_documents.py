import pandas as pd

def build_case_records(
    claims: pd.DataFrame,
    splits: pd.DataFrame,
) -> list[dict]:

    if claims["claim_id"].duplicated().any():
        raise ValueError("Duplicate historical claim IDs.")

    if splits["claim_id"].duplicated().any():
        raise ValueError("Duplicate split IDs.")

    if set(claims["claim_id"]) != set(splits["claim_id"]):
        raise ValueError("Historical claims and split IDs differ.")

    training_ids = splits.loc[
        splits["split"] == "train",
        ["claim_id"],
    ]

    training = claims.merge(
        training_ids,
        on="claim_id",
        validate="one_to_one",
    ).sort_values("claim_id")

    records = []

    for row in training.itertuples(index=False):
        document = (
            f"Incident: {row.incident_type}. "
            f"Vehicle type: {row.vehicle_type}. "
            f"Reported damage severity: {row.damage_severity}. "
            f"Driver age: {row.driver_age}. "
            f"Vehicle age: {row.vehicle_age} years. "
            f"Vehicle value: {row.vehicle_value:.2f}. "
            f"Repair estimate: {row.repair_estimate:.2f}. "
            f"Annual mileage: {row.annual_mileage}. "
            f"Previous claims: {row.prior_claims}. "
            f"Weather: {row.weather}. "
            f"Region: {row.region}. "
            f"Police reported: {row.police_reported}."
        )

        metadata = {
            "claim_id": str(row.claim_id),
            "source_type": "synthetic_historical_case",
            "incident_type": str(row.incident_type),
            "damage_severity": str(row.damage_severity),
            "vehicle_type": str(row.vehicle_type),
            "incident_date": str(row.incident_date),
            "repair_estimate": float(row.repair_estimate),
            "claim_amount": float(row.claim_amount),
            "split": "train",
        }

        records.append({
            "id": f"case::{row.claim_id}",
            "document": document,
            "metadata": metadata,
        })

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case record IDs found.")

    return records