import pandas as pd
import pytest

from src import app


def valid_values():
    return [
        app.DEMO_CLAIM[field]
        for field in app.INPUT_FIELDS
    ]


def test_build_claim():
    claim = app.build_claim(valid_values())

    assert claim["incident_type"] == "collision"
    assert claim["repair_estimate"] == 7800
    assert len(claim) == 14

def test_invalid_driver_age():
    values = valid_values()

    index = app.INPUT_FIELDS.index("driver_age")
    values[index] = 12

    with pytest.raises(ValueError):
        app.build_claim(values)

def fake_result():
    return {
        "ml_prediction": {
            "predicted_amount": 7298.65,
            "selected_model": "engineered",
        },
        "assessment": {
            "summary": "Preliminary assessment completed.",
            "documentation_gaps": [],
            "recommended_next_steps": [
                "Review supporting documentation."
            ],
            "citations": [{
                "source_type": "policy",
                "source_id": "policy::collision_policy::000",
            }],
            "limitations": [
                "Not a final settlement decision."
            ],
        },
        "evidence": {
            "historical_cases": [{
                "claim_id": "CASE001",
                "incident_type": "collision",
                "damage_severity": "moderate",
                "vehicle_type": "sedan",
                "repair_estimate": 7700.0,
                "historical_claim_amount": 7100.0,
                "repair_difference": 100.0,
            }],
            "policies": [{
                "source_file": "collision_policy.md",
                "chunk_id": "policy::collision_policy::000",
                "document": "Collision claims should include documentation.",
            }],
        },
        "image_analysis": {
            "image_provided": False,
        },
        "policy_coverage_note": None,
    }

def test_successful_submission(monkeypatch):
    monkeypatch.setattr(
        app,
        "assess_claim",
        lambda claim, image_path: fake_result(),
    )

    output = app.run_assessment(
        *valid_values(),
        None,
    )

    assert len(output) == 6
    assert "$7,298.65" in output[1]
    assert "Preliminary assessment" in output[2]
    assert isinstance(output[3], pd.DataFrame)
    assert output[3].iloc[0]["Claim ID"] == "CASE001"


def test_invalid_input_does_not_call_backend(monkeypatch):
    backend_called = False

    def fake_assess_claim(*args, **kwargs):
        nonlocal backend_called
        backend_called = True

    monkeypatch.setattr(app, "assess_claim", fake_assess_claim)

    values = valid_values()
    values[app.INPUT_FIELDS.index("vehicle_value")] = -1
    output = app.run_assessment(*values, None)

    assert "Assessment unavailable" in output[0]
    assert backend_called is False
    assert output[3].empty
    assert output[4].empty


def test_backend_failure_clears_outputs(monkeypatch):
    def failing_assess_claim(*args, **kwargs):
        raise ValueError("Invalid citation structure.")

    monkeypatch.setattr(app, "assess_claim", failing_assess_claim)

    output = app.run_assessment(*valid_values(), None)

    assert "Assessment unavailable" in output[0]
    assert "Invalid citation structure." in output[0]
    assert output[1] == ""
    assert output[2] == ""
    assert output[3].empty
    assert output[4].empty
    assert output[5] == ""


def test_optional_photograph_is_forwarded(monkeypatch):
    captured = {}

    def fake_assess_claim(claim, image_path):
        captured["image_path"] = image_path
        return fake_result()

    monkeypatch.setattr(app, "assess_claim", fake_assess_claim)

    app.run_assessment(*valid_values(), "sample_vehicle.jpg")

    assert captured["image_path"] == "sample_vehicle.jpg"
