import pytest

from src import retrieval
from src.evaluate_retrieval import BASE_CLAIM


class FakeCollection:
    def __init__(self, results):
        self.results = results

    def count(self):
        return len(self.results["ids"][0])

    def query(self, **kwargs):
        return self.results


def test_case_query():
    query = retrieval.build_case_query(BASE_CLAIM)

    assert "collision" in query
    assert "moderate" in query
    assert "7800" in query
    assert "claim_amount" not in query

def test_policy_query():
    query = retrieval.build_policy_query(BASE_CLAIM)

    assert "collision" in query
    assert "documentation" in query
    assert "police" in query

def test_invalid_retrieval_limits():
    with pytest.raises(ValueError):
        retrieval.retrieve_historical_cases(
            BASE_CLAIM,
            top_k=0,
        )

    with pytest.raises(ValueError):
        retrieval.retrieve_historical_cases(
            BASE_CLAIM,
            top_k=10,
            candidate_k=5,
        )


def test_historical_reranking(monkeypatch):
    metadata = [
        {"claim_id": "sedan-case", "incident_type": "collision", "damage_severity": "moderate", "vehicle_type": "sedan", "repair_estimate": 9000, "claim_amount": 8500, "source_type": "synthetic_claim", "split": "train"},
        {"claim_id": "truck-case", "incident_type": "collision", "damage_severity": "moderate", "vehicle_type": "truck", "repair_estimate": 7900, "claim_amount": 7600, "source_type": "synthetic_claim", "split": "train"},
        {"claim_id": "coupe-case", "incident_type": "collision", "damage_severity": "moderate", "vehicle_type": "coupe", "repair_estimate": 7700, "claim_amount": 7400, "source_type": "synthetic_claim", "split": "train"},
    ]
    cases = FakeCollection({
        "ids": [["case-1", "case-2", "case-3"]],
        "documents": [["sedan document", "truck document", "coupe document"]],
        "metadatas": [metadata],
        "distances": [[0.9, 0.8, 0.2]],
    })
    monkeypatch.setattr(retrieval, "load_collections", lambda: (cases, None))

    results = retrieval.retrieve_historical_cases(BASE_CLAIM, top_k=3, candidate_k=3)

    assert [case["claim_id"] for case in results] == [
        "sedan-case", "coupe-case", "truck-case"
    ]


def test_policy_source_attribution(monkeypatch):
    policies = FakeCollection({
        "ids": [["policy::collision_policy::000"]],
        "documents": [["Collision claims require repair documentation."]],
        "metadatas": [[{
            "source_file": "collision_policy.md",
            "chunk_index": 0,
            "source_type": "synthetic_policy",
        }]],
        "distances": [[0.4]],
    })
    monkeypatch.setattr(retrieval, "load_collections", lambda: (None, policies))

    result = retrieval.retrieve_policies(BASE_CLAIM, top_k=1)[0]

    assert result["source_file"] == "collision_policy.md"
    assert result["chunk_id"] == "policy::collision_policy::000"
    assert result["chunk_index"] == 0
    assert result["document"] == "Collision claims require repair documentation."


def test_rejects_non_training_retrieval_results(monkeypatch):
    cases = FakeCollection({
        "ids": [["test-case"]],
        "documents": [["test split document"]],
        "metadatas": [[{"split": "test"}]],
        "distances": [[0.1]],
    })
    monkeypatch.setattr(retrieval, "load_collections", lambda: (cases, None))

    with pytest.raises(ValueError, match="Non-training case retrieved"):
        retrieval.retrieve_historical_cases(BASE_CLAIM, top_k=1, candidate_k=1)


def test_combined_evidence_structure(monkeypatch):
    historical_cases = [{"claim_id": "claim-1"}, {"claim_id": "claim-2"}]
    policies = [{"chunk_id": "policy::000"}]
    monkeypatch.setattr(
        retrieval,
        "retrieve_historical_cases",
        lambda claim, top_k, candidate_k: historical_cases,
    )
    monkeypatch.setattr(
        retrieval,
        "retrieve_policies",
        lambda claim, top_k: policies,
    )

    assert retrieval.retrieve_evidence(BASE_CLAIM) == {
        "historical_cases": historical_cases,
        "policies": policies,
        "historical_case_count": 2,
        "policy_chunk_count": 1,
    }
