import sqlite3

import pandas as pd

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR
from src.database import create_database


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python -m src.clean_data first."
        )

    claims = pd.read_csv(DATA_PATH)

    for column in ["incident_date", "policy_start_date"]:
        claims[column] = (
            pd.to_datetime(claims[column])
            .dt.strftime("%Y-%m-%d")
        )

    create_database(claims)

    with sqlite3.connect(DATABASE_PATH) as connection:
        result = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS total_claims,
                COUNT(DISTINCT claim_id) AS unique_claims
            FROM claims
            """,
            connection,
        )

    print("\nDatabase created")
    print(f"Path: {DATABASE_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()