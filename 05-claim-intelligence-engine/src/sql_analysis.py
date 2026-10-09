import sqlite3

import pandas as pd

from src.config import DATABASE_PATH


def run_query(query: str, params=()):
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=params,
        )

def incident_statistics():
    query = """
        SELECT
            incident_type,
            COUNT(*) AS claim_count,
            ROUND(AVG(claim_amount), 2) AS average_claim
        FROM claims
        GROUP BY incident_type
        ORDER BY claim_count DESC
    """

    return run_query(query)

def severity_statistics():
    query = """
        SELECT
            damage_severity,
            COUNT(*) AS claim_count,
            ROUND(AVG(claim_amount), 2) AS average_settlement,
            MIN(claim_amount) AS minimum_settlement,
            MAX(claim_amount) AS maximum_settlement

        FROM claims
        GROUP BY damage_severity
        ORDER BY average_settlement DESC
    """

    return run_query(query)

def monthly_statistics():
    query = """
        SELECT
            strftime('%Y-%m', incident_date) AS month,
            COUNT(*) AS claim_count,
            ROUND(SUM(claim_amount), 2) AS total_claim_amount

        FROM claims
        GROUP BY month
        ORDER BY month
    """

    return run_query(query)

def comparable_claims(
    incident_type: str,
    damage_severity: str,
    vehicle_type: str,
    repair_estimate: float,
    limit: int = 5,
):
    query = """
        SELECT
            claim_id,
            incident_date,
            incident_type,
            vehicle_type,
            damage_severity,
            repair_estimate,
            vehicle_value,
            prior_claims,
            claim_amount
        FROM claims
        WHERE incident_type = ?
          AND damage_severity = ?
          AND vehicle_type = ?
        ORDER BY ABS(repair_estimate - ?) ASC
        LIMIT ?
    """

    params = (
        incident_type,
        damage_severity,
        vehicle_type,
        repair_estimate,
        limit,
    )

    return run_query(query, params)

def main():
    print("\nIncident statistics")
    print(incident_statistics().to_string(index=False))

    print("\nSeverity statistics")
    print(severity_statistics().to_string(index=False))

    print("\nMonthly statistics")
    print(monthly_statistics().head(10).to_string(index=False))

    print("\nComparable historical cases")
    result = comparable_claims(
        incident_type="collision",
        damage_severity="moderate",
        vehicle_type="sedan",
        repair_estimate=7800,
        limit=5,
    )

    print(result.to_string(index=False))

    print("\nSecond historical comparison")

    result = comparable_claims(
        incident_type="theft",
        damage_severity="severe",
        vehicle_type="suv",
        repair_estimate=15000,
        limit=5,
    )

    print(result.to_string(index=False))

if __name__ == "__main__":
    main()