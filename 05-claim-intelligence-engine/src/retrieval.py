from functools import lru_cache

import chromadb

from src.build_vector_store import get_embedding_function
from src.config import VECTOR_STORE_DIR
from src.predict import validate_claim


CASE_CANDIDATES = 25
CASE_TOP_K = 5
POLICY_TOP_K = 2

@lru_cache(maxsize=1)
def load_collections():
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. "
            "Run python -m src.build_vector_store first."
        )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    embedding_function = get_embedding_function()

    cases = client.get_collection(
        name="historical_claims",
        embedding_function=embedding_function,
    )

    policies = client.get_collection(
        name="claim_policies",
        embedding_function=embedding_function,
    )

    if cases.count() == 0 or policies.count() == 0:
        raise ValueError("One or more collections are empty.")

    return cases, policies

def build_case_query(claim: dict) -> str:
    return (
        f"Incident: {claim['incident_type']}. "
        f"Vehicle type: {claim['vehicle_type']}. "
        f"Reported damage severity: {claim['damage_severity']}. "
        f"Driver age: {claim['driver_age']}. "
        f"Vehicle age: {claim['vehicle_age']} years. "
        f"Vehicle value: {claim['vehicle_value']:.2f}. "
        f"Repair estimate: {claim['repair_estimate']:.2f}. "
        f"Annual mileage: {claim['annual_mileage']}. "
        f"Previous claims: {claim['prior_claims']}. "
        f"Weather: {claim['weather']}. "
        f"Region: {claim['region']}. "
        f"Police reported: {claim['police_reported']}."
    )

def retrieve_historical_cases(
    claim: dict,
    top_k: int = CASE_TOP_K,
    candidate_k: int = CASE_CANDIDATES,
) -> list[dict]:

    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Invalid retrieval limits.")

    validated = validate_claim(claim).iloc[0].to_dict()

    cases, _ = load_collections()

    query = build_case_query(validated)

    where = {
        "$and": [
            {"incident_type": validated["incident_type"]},
            {"damage_severity": validated["damage_severity"]},
        ]
    }

    results = cases.query(
        query_texts=[query],
        n_results=min(candidate_k, cases.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []

    for case_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if metadata["split"] != "train":
            raise ValueError("Non-training case retrieved.")

        candidates.append({
            "id": case_id,
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
        })

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["metadata"]["vehicle_type"]
            != validated["vehicle_type"],

            abs(
                item["metadata"]["repair_estimate"]
                - validated["repair_estimate"]
            ),

            item["distance"],
        ),
    )

    final_cases = []

    for item in ranked[:top_k]:
        metadata = item["metadata"]

        repair_difference = abs(
            metadata["repair_estimate"]
            - validated["repair_estimate"]
        )

        final_cases.append({
            "claim_id": metadata["claim_id"],
            "incident_type": metadata["incident_type"],
            "damage_severity": metadata["damage_severity"],
            "vehicle_type": metadata["vehicle_type"],
            "repair_estimate": metadata["repair_estimate"],
            "historical_claim_amount": metadata["claim_amount"],
            "repair_difference": repair_difference,
            "vector_distance": item["distance"],

            "document": item["document"],
            "source_type": metadata["source_type"],
        })

    return final_cases

def build_policy_query(claim: dict) -> str:
    return (
        f"Insurance claim procedures for "
        f"{claim['incident_type']} incidents. "
        f"Reported damage severity: {claim['damage_severity']}. "
        f"Repair estimate: {claim['repair_estimate']:.2f}. "
        f"Vehicle value: {claim['vehicle_value']:.2f}. "
        f"Prior claims: {claim['prior_claims']}. "
        f"police reported: {claim['police_reported']}. "
        f"Required supporting documentation, "
        f"reporting requirements, claim review conditions, "
        f"and manual assessment procedures."
    )

def retrieve_policies(
    claim: dict,
    top_k: int = POLICY_TOP_K,
) -> list[dict]:

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    validated = validate_claim(claim).iloc[0].to_dict()

    _, policies = load_collections()

    query = build_policy_query(validated)

    results = policies.query(
        query_texts=[query],
        n_results=min(top_k, policies.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "chunk_id": chunk_id,
            "source_file": metadata["source_file"],
            "chunk_index": metadata["chunk_index"],
            "document": document,
            "vector_distance": float(distance),
        })

    return retrieved

def retrieve_evidence(claim: dict) -> dict:
    historical_cases = retrieve_historical_cases(
        claim,
        top_k=5,
        candidate_k=25,
    )

    policies = retrieve_policies(
        claim,
        top_k=2,
    )

    return {
        "historical_cases": historical_cases,
        "policies": policies,
        "historical_case_count": len(historical_cases),
        "policy_chunk_count": len(policies),
    }
