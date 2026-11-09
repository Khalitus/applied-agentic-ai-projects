"""Live smoke test using the configured primary provider and its fallback."""

import argparse

from src.agent import ask_agent


QUESTIONS = (
    "Compare average bike rentals on working and non-working days, including sample sizes.",
    "Which three hours have the highest average demand on working days?",
    "How many rentals were observed at 8 AM in the historical dataset?",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test configured provider tool calling against local analytics")
    parser.add_argument("--all", action="store_true", help="Run all three example prompts")
    args = parser.parse_args()

    for question in QUESTIONS if args.all else QUESTIONS[:1]:
        print(f"\nQuestion: {question}\n")
        print(ask_agent(question))


if __name__ == "__main__":
    main()
