"""Trusted, parameterized SQLite analytics; no LLM-generated SQL execution."""

import sqlite3
from src.config import DATABASE


SQL_OVERVIEW = """
SELECT COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent
FROM campaign_records;
"""

SQL_SEGMENT = """
SELECT {dimension} AS segment, COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent
FROM campaign_records
GROUP BY {dimension}
HAVING COUNT(*) >= ?
ORDER BY subscription_rate_percent DESC, total_records DESC;
"""

DIMENSIONS = {"job": "job", "education": "education", "poutcome": "poutcome"}


def query_history(segment="job", min_records=100, path=DATABASE):
    if segment not in DIMENSIONS:
        raise ValueError(f"Unsupported grouping dimension: {segment}")
    if not isinstance(min_records, int) or min_records < 1:
        raise ValueError("min_records must be a positive integer")
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        overview = [dict(x) for x in connection.execute(SQL_OVERVIEW)][0]
        # Only pre-registered column identifiers may be interpolated.
        sql = SQL_SEGMENT.format(dimension=DIMENSIONS[segment])
        rows = [dict(x) for x in connection.execute(sql, (min_records,))]
    return {"overview": overview, "grouped_by": segment, "segments": rows}
