import copy
import math

import pytest

from src.evaluate_retrieval import BASE_CLAIM
from src.predict import validate_claim


@pytest.mark.parametrize(
    "field, value",
    [
        ("driver_age", 17),
        ("vehicle_value", 0),
        ("repair_estimate", -100),
        ("prior_claims", -1),
        ("prior_claims", 1.5),
        ("annual_mileage", math.inf),
        ("repair_estimate", math.nan),
    ],
)
def test_invalid_numeric_inputs(field, value):
    claim = {**BASE_CLAIM, field: value}

    with pytest.raises(ValueError):
        validate_claim(claim)
    
def test_missing_incident_type():
    claim = BASE_CLAIM.copy()

    dict.pop(claim, "incident_type", None)

    with pytest.raises(ValueError):
        validate_claim(claim)


@pytest.mark.parametrize(
    "field, value",
    [
        ("incident_type", "earthquake"),
        ("vehicle_type", "spaceship"),
        ("weather", "unknown_weather"),
        ("region", "unknown_region"),
    ],
)
def test_unsupported_categories(field, value):
    claim = {**BASE_CLAIM, field: value}

    with pytest.raises(ValueError, match="Unsupported category"):
        validate_claim(claim)


def test_base_claim_categories_remain_valid():
    validate_claim(BASE_CLAIM)

def test_validation_does_not_modify_input():
    claim = copy.deepcopy(BASE_CLAIM)
    original = copy.deepcopy(claim)

    validate_claim(claim)

    assert claim == original

from src import retrieval


class EmptyCaseCollection:
    def count(self):
        return 840

    def query(self, **kwargs):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }


def test_empty_historical_retrieval(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "load_collections",
        lambda: (EmptyCaseCollection(), None),
    )

    results = retrieval.retrieve_historical_cases(
        BASE_CLAIM,
        top_k=5,
    )

    assert results == []

from src.assessment import (
    get_policy_coverage_note,
    validate_llm_assessment,
)

def test_missing_incident_policy():
    claim = {
        **BASE_CLAIM,
        "incident_type": "hail",
    }

    evidence = {
        "historical_cases": [],
        "policies": [{
            "source_file": "claim_review_policy.md",
        }],
    }

    note = get_policy_coverage_note(claim, evidence)

    assert note is not None
    assert "hail" in note


def test_fabricated_citation_with_empty_evidence_is_rejected():
    empty_evidence = {
        "historical_cases": [],
        "policies": [],
    }
    assessment = {
        "summary": "Assessment summary.",
        "documentation_gaps": [],
        "recommended_next_steps": [],
        "citations": [{
            "source_type": "policy",
            "source_id": "policy::imaginary_policy::000",
        }],
        "limitations": [],
    }

    with pytest.raises(ValueError):
        validate_llm_assessment(assessment, empty_evidence)


def test_rejects_invalid_citation_structure():
    assessment = {
        "summary": "Preliminary assessment.",
        "documentation_gaps": [],
        "recommended_next_steps": [],
        "citations": [{
            "source_type": "policy",
            "source_id": "policy::collision_policy::000",
            "unexpected_field": "extra",
        }],
        "limitations": [],
    }
    evidence = {
        "historical_cases": [],
        "policies": [{
            "chunk_id": "policy::collision_policy::000",
        }],
    }

    with pytest.raises(ValueError, match="Invalid citation structure"):
        validate_llm_assessment(assessment, evidence)
