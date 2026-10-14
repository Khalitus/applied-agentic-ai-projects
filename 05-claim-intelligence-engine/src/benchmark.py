import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from src.config import ROOT_DIR
from src.evaluate_retrieval import BASE_CLAIM
from src.predict import predict_claim
from src.retrieval import retrieve_evidence


REPORT_DIR = ROOT_DIR / "reports"


def measure(function, repeats=3):
    durations = []
    result = None

    for index in range(repeats):
        print(
            f"Run {index + 1}/{repeats}: {function.__name__}",
            flush=True,
        )

        start = perf_counter()
        result = function(BASE_CLAIM)
        elapsed = perf_counter() - start

        durations.append(elapsed)
        print(f"Elapsed: {elapsed:.3f}s", flush=True)

    return {
        "first_call_seconds": round(durations[0], 4),
        "warm_median_seconds": round(
            statistics.median(durations[1:]),
            4,
        ) if repeats > 1 else None,
        "runs_seconds": [
            round(value, 4) for value in durations
        ],
    }, result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        choices=["before", "after"],
        required=True,
    )

    args = parser.parse_args()

    prediction_timing, prediction = measure(
        predict_claim,
    )

    retrieval_timing, evidence = measure(
        retrieve_evidence,
    )

    report = {
        "label": args.label,
        "iterations": 3,
        "prediction": prediction_timing,
        "retrieval": retrieval_timing,
        "reference_output": {
            "predicted_amount": prediction["predicted_amount"],
            "selected_model": prediction["selected_model"],
            "historical_claim_ids": [case["claim_id"] for case in evidence["historical_cases"]],
            "policy_chunk_ids": [chunk["chunk_id"] for chunk in evidence["policies"]],
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    path = REPORT_DIR / f"performance_{args.label}.json"

    if path.exists():
        raise FileExistsError(f"Report already exists: {path}")

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
