"""Optional live, evidence-based agent evaluation; consumes provider quota."""

import argparse
import json
import re
from datetime import datetime, timezone

from src.agent import ask_agent
from src.analytics import compare_workdays, hourly_demand, hourly_profile, weather_impact
from src.config import ROOT


def numeric_values(answer: str) -> list[float]:
    """Extract literal numbers for a modest diagnostic, not semantic proof."""
    matches = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])", answer)
    return [float(value.replace(",", "")) for value in matches]


def contains_numbers(answer: str, expected: list[tuple[float, float]]) -> bool:
    found = numeric_values(answer)
    return all(any(abs(value - number) <= tolerance for value in found)
               for number, tolerance in expected)


def evaluation_cases() -> list[dict]:
    workdays = compare_workdays()
    nonwork = next(row for row in workdays if row["workingday"] == 0)
    working = next(row for row in workdays if row["workingday"] == 1)
    hourly = sorted(hourly_demand(workingday=1), key=lambda row: row["avg_rentals"], reverse=True)[:3]
    morning = hourly_profile(8)
    weather = next(row for row in weather_impact() if row["weathersit"] == 4)

    return [
        {
            "id": "working_day_comparison",
            "question": "Compare average bike rentals on working and non-working days, including sample sizes and percentage difference.",
            "expected": [
                (nonwork["observations"], 0), (working["observations"], 0),
                (nonwork["avg_rentals"], 0.11), (working["avg_rentals"], 0.11),
                (working["percent_difference_vs_nonworking"], 0.11),
            ],
        },
        {
            "id": "busiest_working_hours",
            "question": "Which three hours have the highest average bike rentals on working days? Include hourly means.",
            "expected": [(row["hr"], 0) for row in hourly]
                        + [(row["avg_rentals"], 0.15) for row in hourly],
        },
        {
            "id": "hourly_reference",
            "question": "How many hourly records are there at 8 AM, and what is the average rental count for that hour?",
            "expected": [(morning["observations"], 0), (morning["avg_rentals"], 0.11)],
        },
        {
            "id": "rare_weather_sample",
            "question": "How many hourly records are in weather category 4? Explain why this is a small sample.",
            "expected": [(weather["observations"], 0)],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional live model-answer checks against SQLite facts")
    parser.add_argument("--live", action="store_true", help="Make four provider requests, which may consume quota")
    args = parser.parse_args()
    if not args.live:
        print("No API requests made. Use --live to evaluate four independent questions.")
        return 0

    results = []
    for case in evaluation_cases():
        answer = ask_agent(case["question"])
        result = {
            "id": case["id"], "question": case["question"],
            "numeric_evidence_present": contains_numbers(answer, case["expected"]),
            "answer": answer,
        }
        results.append(result)
        print(f"{case['id']}: {'PASS' if result['numeric_evidence_present'] else 'REVIEW'}")

    path = ROOT / "reports" / "agent_evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Heuristic numeric checks against SQLite; not a semantic correctness guarantee",
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {path}")
    print("Review answer wording, units, tool use, and limitations manually before publication.")
    return 0 if all(item["numeric_evidence_present"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
