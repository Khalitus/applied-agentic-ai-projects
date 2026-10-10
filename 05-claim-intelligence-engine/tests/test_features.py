import pandas as pd

from src.features import create_features


def sample_claim():
    return pd.DataFrame([
        {
            "incident_date": "2026-01-11",
            "policy_start_date": "2026-01-01",
            "driver_age": 34,
            "vehicle_age": 5,
            "annual_mileage": 12000,
            "vehicle_value": 10000,
            "repair_estimate": 2500,
            "prior_claims": 2,
            "incident_type": "collision",
            "vehicle_type": "sedan",
            "weather": "clear",
            "region": "north",
            "police_reported": "yes",
            "damage_severity": "moderate",
        }
    ])


def test_engineered_features():
    result = create_features(sample_claim())

    assert result.loc[0, "policy_tenure_days"] == 10

    assert result.loc[0, "repair_value_ratio"] == 0.25

    assert result.loc[0, "mileage_age_ratio"] == 2000

    assert result.loc[0, "has_prior_claim"] == 1

    assert result.loc[0, "driver_age_band"] == "25-34"

    assert result.loc[0, "vehicle_age_band"] == "recent"

    assert result.loc[0, "is_weekend"] == 1

def test_input_not_modified():
    original = sample_claim()
    before = original.copy(deep=True)

    create_features(original)

    assert original.equals(before)

def test_new_vehicle_age_band():
    claim = sample_claim()
    claim.loc[0, "vehicle_age"] = 0

    result = create_features(claim)

    assert result.loc[0, "vehicle_age_band"] == "new"