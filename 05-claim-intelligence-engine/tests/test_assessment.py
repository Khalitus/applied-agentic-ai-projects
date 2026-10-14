import json

import pytest

from src import assessment, llm_client
from src.evaluate_retrieval import BASE_CLAIM


MOCK_EVIDENCE = {
    "historical_cases": [
        {
            "claim_id": "CASE001",
            "incident_type": "collision",
            "damage_severity": "moderate",
            "vehicle_type": "sedan",
            "repair_estimate": 7700.0,
            "historical_claim_amount": 7100.0,
            "source_type": "synthetic_historical_case",
        }
    ],
    "policies": [
        {
            "chunk_id": "policy::collision_policy::000",
            "source_file": "collision_policy.md",
            "chunk_index": 0,
            "document": "Collision claims require documentation.",
        }
    ],
    "historical_case_count": 1,
    "policy_chunk_count": 1,
}

MOCK_ASSESSMENT = {
    "summary": (
        "A preliminary claim estimate is available. "
        "The supporting documentation should be reviewed."
    ),
    "documentation_gaps": [
        "Confirm an itemized repair assessment is available."
    ],
    "recommended_next_steps": [
        "Review submitted repair documentation."
    ],
    "citations": [
        {
            "source_type": "policy",
            "source_id": "policy::collision_policy::000",
        }
    ],
    "limitations": [
        "Automated analysis is not a final settlement decision."
    ],
}

def test_valid_citations():
    result = assessment.validate_llm_assessment(
        MOCK_ASSESSMENT,
        MOCK_EVIDENCE,
    )

    assert result["citations"][0]["source_id"] == (
        "policy::collision_policy::000"
    )


def mock_pipeline(monkeypatch, llm_response):
    monkeypatch.setattr(
        assessment,
        "predict_claim",
        lambda claim: {
            "predicted_amount": 7298.65,
            "selected_model": "engineered",
            "feature_mode": "engineered",
        },
    )
    monkeypatch.setattr(
        assessment,
        "retrieve_evidence",
        lambda claim: MOCK_EVIDENCE,
    )
    monkeypatch.setattr(
        assessment,
        "analyze_image",
        lambda image_path: {
            "image_provided": False,
            "caption": None,
            "model_id": None,
            "analysis_type": None,
            "limitations": "No photograph provided.",
        },
    )
    monkeypatch.setattr(
        assessment,
        "generate_llm_response",
        lambda system_prompt, user_prompt: llm_response,
    )


def test_invalid_citation():
    hallucinated = {
        **MOCK_ASSESSMENT,
        "citations": [{
            "source_type": "policy",
            "source_id": "policy::nonexistent::000",
        }],
    }

    with pytest.raises(ValueError):
        assessment.validate_llm_assessment(hallucinated, MOCK_EVIDENCE)


def test_assessment_pipeline(monkeypatch):
    mock_pipeline(monkeypatch, json.dumps(MOCK_ASSESSMENT))

    result = assessment.assess_claim(BASE_CLAIM)

    assert result["ml_prediction"]["predicted_amount"] == 7298.65
    assert result["assessment"]["summary"]
    assert result["image_analysis"]["image_provided"] is False


def test_invalid_json(monkeypatch):
    mock_pipeline(monkeypatch, "{invalid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        assessment.assess_claim(BASE_CLAIM)


def test_missing_assessment_field():
    incomplete = {
        key: value
        for key, value in MOCK_ASSESSMENT.items()
        if key != "summary"
    }

    with pytest.raises(ValueError, match="missing or unexpected fields"):
        assessment.validate_llm_assessment(incomplete, MOCK_EVIDENCE)


@pytest.mark.parametrize("incident_type", ["hail", "glass"])
def test_policy_coverage_gap(incident_type):
    claim = {**BASE_CLAIM, "incident_type": incident_type}

    note = assessment.get_policy_coverage_note(claim, MOCK_EVIDENCE)

    assert f"available for {incident_type} claims" in note


def test_assessment_prompt_has_explicit_policy_retrieval_status():
    prompt = assessment.build_assessment_prompt(
        BASE_CLAIM,
        {"predicted_amount": 7298.65},
        {"image_provided": False},
        MOCK_EVIDENCE,
    )
    payload = json.loads(prompt.split("\n\n", 1)[1])

    assert "policy_coverage_note" not in payload
    assert payload["policy_retrieval_status"] == (
        "The expected incident-specific demonstration policy "
        "was retrieved. Actual coverage terms, deductibles, "
        "and policy limits have not been verified."
    )


def test_optional_image_path(monkeypatch):
    received = []
    mock_pipeline(monkeypatch, json.dumps(MOCK_ASSESSMENT))
    monkeypatch.setattr(
        assessment,
        "analyze_image",
        lambda image_path: received.append(image_path) or {
            "image_provided": True,
            "caption": "Vehicle damage",
            "model_id": "mock",
            "analysis_type": "caption",
            "limitations": "Mock analysis.",
        },
    )

    assessment.assess_claim(BASE_CLAIM, image_path="damage.jpg")

    assert received == ["damage.jpg"]


def test_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid")

    with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
        llm_client.generate_llm_response("system", "user")
