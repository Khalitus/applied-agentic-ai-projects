"""Historical campaign analytics using parameterized SQLite queries."""

import sqlite3
from contextlib import closing
from pathlib import Path

from src.config import DATABASE

SQL_OVERVIEW = """
SELECT COUNT(*) AS total_records,
       COALESCE(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 0) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent
FROM campaign_records;
"""

SQL_SEGMENT = """
SELECT {dimension} AS segment, COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
GROUP BY {dimension}
HAVING COUNT(*) >= ?
ORDER BY subscription_rate_percent DESC, total_records DESC, segment;
"""

SQL_CONTACT_SEGMENTS = """
SELECT job, poutcome, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS conversion_rate,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
WHERE balance > 0 AND pdays != -1
GROUP BY job, poutcome
HAVING COUNT(*) >= ?
ORDER BY conversion_rate DESC, total_contacts DESC, job, poutcome
LIMIT ?;
"""

DIMENSIONS = {"job": "job", "education": "education", "poutcome": "poutcome"}


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def query_history(segment="job", min_records=100, path=DATABASE):
    if segment not in DIMENSIONS:
        raise ValueError(f"Unsupported grouping dimension: {segment}")
    _positive_integer(min_records, "min_records")
    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        overview = dict(connection.execute(SQL_OVERVIEW).fetchone())
        sql = SQL_SEGMENT.format(dimension=DIMENSIONS[segment])
        rows = [dict(row) for row in connection.execute(sql, (min_records,))]
    return {"overview": overview, "grouped_by": segment, "segments": rows}


def query_contact_segments(min_records=50, limit=10, path=DATABASE):
    _positive_integer(min_records, "min_records")
    _positive_integer(limit, "limit")
    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            SQL_CONTACT_SEGMENTS, (min_records, limit)
        )]
