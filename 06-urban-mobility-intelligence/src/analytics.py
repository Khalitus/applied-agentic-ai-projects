"""Examples of safe, restricted SQL tools; add additional tools as exercises."""
from pathlib import Path

from src.config import DB_PATH
from src.db import connect


def hourly_profile(hour: int, path: Path = DB_PATH) -> dict:
    if not 0 <= int(hour) <= 23:
        raise ValueError("Hour must be 0–23")
    sql = """
        SELECT hr, COUNT(*) AS observations,
               ROUND(AVG(cnt), 2) AS avg_rentals,
               MIN(cnt) AS min_rentals,
               MAX(cnt) AS max_rentals
        FROM bike_rentals
        WHERE hr = ?
        GROUP BY hr
    """
    with connect(path) as connection:
        row = connection.execute(sql, (int(hour),)).fetchone()
    if row is None:
        return {"hour": hour, "observations": 0}
    return dict(row)


def compare_workdays(path: Path = DB_PATH) -> list[dict]:
    # TODO (Task 4): Group by workingday and compare average rental counts.
    raise NotImplementedError("Implement compare_workdays in Task 4")


def weather_impact(path: Path = DB_PATH) -> list[dict]:
    # TODO (Task 4): Group by weathersit and report sample sizes and demand.
    raise NotImplementedError("Implement weather_impact in Task 4")
