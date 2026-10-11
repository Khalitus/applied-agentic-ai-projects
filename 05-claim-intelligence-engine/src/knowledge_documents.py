import pandas as pd
from pathlib import Path

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

def chunk_policy_text(
    text: str,
    chunk_words: int = 130,
    overlap_words: int = 20,
) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("Chunk size must be positive.")

    if not 0 <= overlap_words < chunk_words:
        raise ValueError("Invalid chunk overlap.")

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap_words

    return chunks

def build_policy_records(knowledge_dir: Path) -> list[dict]:
    paths = sorted(
        path
        for path in knowledge_dir.glob("*.md")
        if path.is_file() and not path.name.startswith(".")
    )

    if not paths:
        raise FileNotFoundError(
            f"No valid policy documents found in {knowledge_dir}"
        )

    records = []

    for path in paths:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Empty policy document: {path.name}")

        chunks = chunk_policy_text(text)

        for index, chunk in enumerate(chunks):
            records.append({
                "id": f"policy::{path.stem}::{index:03d}",
                "document": chunk,
                "metadata": {
                    "source_file": path.name,
                    "source_type": "synthetic_policy",
                    "chunk_index": index,
                },
            })

    return records