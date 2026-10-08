from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "claims.csv"
N_ROWS = 1200


def generate_claims(n_rows=N_ROWS):
    incident_types = ["collision", "theft", "hail", "vandalism", "glass"]
    vehicle_types = ["sedan", "suv", "hatchback", "pickup", "coupe"]
    weather_types = ["clear", "rain", "storm", "fog"]
    regions = ["north", "south", "east", "west"]
    severity_levels = ["minor", "moderate", "severe"]

    end_date = pd.Timestamp("2026-09-30")

    incident_dates = end_date - pd.to_timedelta(
        RNG.integers(0, 730, n_rows),
        unit="D",
    )

    policy_tenure_days = RNG.integers(30, 2200, n_rows)
    policy_start_dates = incident_dates - pd.to_timedelta(
        policy_tenure_days,
        unit="D",
    )

    driver_age = RNG.integers(18, 81, n_rows)
    vehicle_age = RNG.integers(0, 21, n_rows)
    annual_mileage = RNG.integers(3000, 30001, n_rows)

    vehicle_value = np.maximum(
        6000,
        RNG.normal(33000, 15000, n_rows),
    ).round(2)

    prior_claims = RNG.choice(
        [0, 1, 2, 3, 4],
        size=n_rows,
        p=[0.55, 0.25, 0.12, 0.06, 0.02],
    )

    incident_type = RNG.choice(
        incident_types,
        size=n_rows,
        p=[0.52, 0.08, 0.13, 0.10, 0.17],
    )

    vehicle_type = RNG.choice(vehicle_types, n_rows)

    weather = RNG.choice(
        weather_types,
        size=n_rows,
        p=[0.62, 0.22, 0.10, 0.06],
    )

    region = RNG.choice(regions, n_rows)

    police_reported = RNG.choice(
        ["yes", "no"],
        size=n_rows,
        p=[0.58, 0.42],
    )

    damage_severity = RNG.choice(
        severity_levels,
        size=n_rows,
        p=[0.43, 0.38, 0.19],
    )

    severity_multiplier = pd.Series(damage_severity).map(
        {
            "minor": 0.08,
            "moderate": 0.25,
            "severe": 0.55,
        }
    ).to_numpy()

    repair_estimate = (
        vehicle_value
        * severity_multiplier
        * RNG.uniform(0.70, 1.30, n_rows)
    )

    repair_estimate = np.clip(
        repair_estimate,
        300,
        vehicle_value * 0.90,
    ).round(2)

    incident_adjustment = pd.Series(incident_type).map(
        {
            "collision": 900,
            "theft": 2200,
            "hail": 550,
            "vandalism": 400,
            "glass": 150,
        }
    ).to_numpy()

    severity_adjustment = pd.Series(damage_severity).map(
        {
            "minor": 100,
            "moderate": 800,
            "severe": 2500,
        }
    ).to_numpy()

    noise = RNG.normal(0, 1000, n_rows)

    claim_amount = (
        repair_estimate * 0.72
        + incident_adjustment
        + severity_adjustment
        + prior_claims * 180
        + noise
    )

    claim_amount = np.clip(
        claim_amount,
        250,
        vehicle_value * 0.95,
    ).round(2)

    return pd.DataFrame(
        {
            "claim_id": [
                f"CLM{i:05d}"
                for i in range(1, n_rows + 1)
            ],
            "incident_date": incident_dates,
            "policy_start_date": policy_start_dates,
            "driver_age": driver_age,
            "vehicle_age": vehicle_age,
            "annual_mileage": annual_mileage,
            "vehicle_value": vehicle_value,
            "repair_estimate": repair_estimate,
            "prior_claims": prior_claims,
            "incident_type": incident_type,
            "vehicle_type": vehicle_type,
            "weather": weather,
            "region": region,
            "police_reported": police_reported,
            "damage_severity": damage_severity,
            "claim_amount": claim_amount,
        }
    )


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    claims = generate_claims()
    claims.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(claims):,} claims")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
