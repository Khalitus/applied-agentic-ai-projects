import sqlite3

import pandas as pd

from src.config import DATABASE_PATH


def create_database(claims: pd.DataFrame) -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if claims["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs found.")

    with sqlite3.connect(DATABASE_PATH) as connection:
        claims.to_sql(
            "claims",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute("""
            CREATE UNIQUE INDEX idx_claims_id
            ON claims(claim_id)
        """)

        connection.execute("""
            CREATE INDEX idx_claims_comparison
            ON claims(
                incident_type,
                damage_severity,
                vehicle_type
            )
        """)

        connection.execute("""
            CREATE INDEX idx_claims_date
            ON claims(incident_date)
        """)


def get_similar_claims(
    incident_type: str,
    vehicle_type: str,
    damage_severity: str,
) -> dict:
    query = """
        SELECT
            COUNT(*) AS claim_count,
            ROUND(AVG(claim_amount), 2) AS average_claim,
            ROUND(MIN(claim_amount), 2) AS minimum_claim,
            ROUND(MAX(claim_amount), 2) AS maximum_claim
        FROM claims
        WHERE incident_type = ?
          AND vehicle_type = ?
          AND damage_severity = ?
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        result = pd.read_sql_query(
            query,
            connection,
            params=[incident_type, vehicle_type, damage_severity],
        )

    return result.iloc[0].to_dict()
