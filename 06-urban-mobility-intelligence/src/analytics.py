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
    sql = """
        SELECT
            workingday,
            COUNT(*) AS observations,
            ROUND(AVG(cnt), 2) AS avg_rentals,
            MIN(cnt) AS min_rentals,
            MAX(cnt) AS max_rentals
        FROM bike_rentals
        GROUP BY workingday
        ORDER BY workingday
    """

    with connect(path) as connection:
        rows = connection.execute(sql).fetchall()

    return [dict(row) for row in rows]




def weather_impact(path: Path = DB_PATH) -> list[dict]:
    sql = """
        SELECT
            weathersit,
            COUNT(*) AS observations,
            ROUND(AVG(cnt), 2) AS avg_rentals,
            MIN(cnt) AS min_rentals,
            MAX(cnt) AS max_rentals
        FROM bike_rentals
        GROUP BY weathersit
        ORDER BY weathersit
    """

    with connect(path) as connection:
        rows = connection.execute(sql).fetchall()

    return [dict(row) for row in rows]


def hourly_demand(
    workingday: int | None = None,
    path: Path = DB_PATH,
) -> list[dict]:

    if workingday is not None:
        if type(workingday) is not int or workingday not in (0, 1):
            raise ValueError("workingday must be 0, 1, or None")

    sql = """
        SELECT
            hr,
            COUNT(*) AS observations,
            ROUND(AVG(cnt), 2) AS avg_rentals
        FROM bike_rentals
    """

    params = []

    if workingday is not None:
        sql += " WHERE workingday = ?"
        params.append(workingday)

    sql += " GROUP BY hr ORDER BY hr"

    with connect(path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def historical_reference(
    hour: int,
    workingday: int | None = None,
    weathersit: int | None = None,
    path: Path = DB_PATH,
) -> dict:
    if type(hour) is not int or not 0 <= hour <= 23:
        raise ValueError("hour must be an integer between 0 and 23")

    if workingday is not None:
        if type(workingday) is not int or workingday not in (0, 1):
            raise ValueError("workingday must be 0, 1, or None")

    if weathersit is not None:
        if type(weathersit) is not int or weathersit not in (1, 2, 3, 4):
            raise ValueError("weathersit must be 1, 2, 3, 4, or None")

    sql = """
        SELECT
            COUNT(*) AS observations,
            ROUND(AVG(cnt), 2) AS avg_rentals,
            MIN(cnt) AS min_rentals,
            MAX(cnt) AS max_rentals
        FROM bike_rentals
        WHERE hr = ?
    """

    params = [hour]

    if workingday is not None:
        sql += " AND workingday = ?"
        params.append(workingday)

    if weathersit is not None:
        sql += " AND weathersit = ?"
        params.append(weathersit)

    with connect(path) as connection:
        row = connection.execute(sql, params).fetchone()

    return {"hour": hour, **dict(row)}