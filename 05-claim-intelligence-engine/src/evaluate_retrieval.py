import json

from src.config import ROOT_DIR
from src.retrieval import retrieve_evidence


REPORT_PATH = ROOT_DIR / "reports" / "retrieval_evaluation.json"


BASE_CLAIM = {
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

SCENARIOS = [
    {
        "name": "collision_documentation",
        "claim": BASE_CLAIM.copy(),
        "expected_policy": "collision_policy.md",
    },
    {
        "name": "theft_reporting",
        "claim": {
            **BASE_CLAIM,
            "incident_type": "theft",
            "damage_severity": "severe",
            "repair_estimate": 14000,
        },
        "expected_policy": "theft_policy.md",
    },
    {
        "name": "severe_claim_review",
        "claim": {
            **BASE_CLAIM,
            "damage_severity": "severe",
            "repair_estimate": 24000,
            "vehicle_value": 20000,
            "prior_claims": 2,
        },
        "expected_policy": "claim_review_policy.md",
    },
]

def evaluate_scenarios():
    results = []

    for scenario in SCENARIOS:
        name = scenario["name"]
        claim = scenario["claim"]

        evidence = retrieve_evidence(claim)

        historical = evidence["historical_cases"]
        policies = evidence["policies"]

        policy_sources = [
            item["source_file"]
            for item in policies
        ]

        policy_hit = (
            scenario["expected_policy"] in policy_sources
        )

        historical_filter_valid = (
            len(historical) > 0
            and all(
                item["incident_type"] == claim["incident_type"]
                and item["damage_severity"] == claim["damage_severity"]
                for item in historical
            )
        )

        vehicle_match_rate = sum(
            1 for item in historical
            if item["vehicle_type"] == claim["vehicle_type"]
        ) / len(historical) if historical else 0


        repair_differences = [
            abs(item["repair_estimate"] - claim["repair_estimate"])
            for item in historical
        ]
        mean_repair_difference = sum(repair_differences) / len(repair_differences) if repair_differences else 0

        results.append({
            "scenario": name,
            "expected_policy": scenario["expected_policy"],
            "retrieved_policy_sources": policy_sources,
            "policy_hit_at_2": policy_hit,
            "historical_filter_valid": historical_filter_valid,
            "vehicle_match_rate": round(vehicle_match_rate, 4),
            "mean_repair_difference": round(mean_repair_difference, 2),
        })

    return results

def main():
    results = evaluate_scenarios()

    total_scenarios = len(results)
    policy_hits = sum(
        item["policy_hit_at_2"]
        for item in results
    )

    policy_hit_rate = policy_hits / total_scenarios

    report = {
        "scenario_count": total_scenarios,
        "policy_hit_at_2": round(policy_hit_rate, 4),
        "scenarios": results,
        "evaluation_note": (
            "Small synthetic scenario evaluation. "
            "Not a production relevance benchmark."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nRetrieval evaluation")
    print(f"Scenarios evaluated: {total_scenarios}")
    print(f"Policy Hit@2: {policy_hit_rate:.2%}")

    for result in results:
        print(f"\nScenario: {result['scenario']}")
        print(f"Policy hit: {result['policy_hit_at_2']}")
        print(
            "Historical filters valid: "
            f"{result['historical_filter_valid']}"
        )
        print(
            f"Retrieved policies: "
            f"{result['retrieved_policy_sources']}"
        )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()