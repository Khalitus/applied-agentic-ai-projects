import pytest

from src.predict import predict_claim


def sample_claim():
    return {
        "incident_date": "2026-09-15",
        "policy_start_date": "2024-06-01",
        "driver_age": 34,
        "vehicle_age": 5,
        "annual_mileage": 12000,
        "vehicle_value": 28000,
        "repair_estimate": 7800,
        "prior_claims": 1,
        "incident_type": "collision",
        "vehicle_type": "sedan",
        "weather": "clear",
        "region": "north",
        "police_reported": "yes",
        "damage_severity": "moderate",
    }


def test_prediction():
    result = predict_claim(sample_claim())

    assert isinstance(result["predicted_amount"], float)
    assert result["predicted_amount"] >= 0
    assert result["selected_model"] in {"baseline", "engineered"}

def test_prediction_is_deterministic():
    claim = sample_claim()

    first = predict_claim(claim)
    second = predict_claim(claim)

    assert first == second

def test_negative_repair_estimate():
    claim = sample_claim()
    claim["repair_estimate"] = -500

    with pytest.raises(ValueError):
        predict_claim(claim)

def test_policy_invalid_date():
    claim = sample_claim()
    claim["policy_start_date"] = "2026-10-01"
    claim["incident_date"] = "2026-09-15"
    
    with pytest.raises(ValueError):
        predict_claim(claim)

def test_missing_required_field():
    claim = sample_claim()
    claim.pop("vehicle_value")

    with pytest.raises(ValueError):
        predict_claim(claim)
