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


AGE_GROUP = """CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age < 30 THEN 'Under 30'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END"""

BALANCE_GROUP = """CASE
        WHEN balance IS NULL THEN 'Unknown'
        WHEN balance < 0 THEN 'Negative'
        WHEN balance = 0 THEN 'Zero'
        WHEN balance < 1000 THEN 'Low'
        WHEN balance < 5000 THEN 'Medium'
        ELSE 'High'
    END"""

SEGMENT_METRICS = """COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)
           AS conversion_rate,
       AVG(age) AS average_age,
       AVG(balance) AS average_balance"""

DISPLAY_METRICS = """total_contacts, subscriptions, non_subscriptions,
       ROUND(conversion_rate, 2) AS conversion_rate,
       ROUND(average_age, 2) AS average_age,
       ROUND(average_balance, 2) AS average_balance"""


def _single_segment_sql(group, expression):
    return f"""
WITH segments AS (
    SELECT age, balance, y,
    {expression} AS {group}
    FROM campaign_records
), segment_statistics AS (
    SELECT {group}, {SEGMENT_METRICS}
    FROM segments
    GROUP BY {group}
    HAVING COUNT(*) >= ?
)
SELECT {group}, {DISPLAY_METRICS}
FROM segment_statistics
ORDER BY conversion_rate DESC, total_contacts DESC, {group};
"""


SQL_AGE_SEGMENTS = _single_segment_sql("age_group", AGE_GROUP)
SQL_BALANCE_SEGMENTS = _single_segment_sql("balance_group", BALANCE_GROUP)

COMBINED_STATISTICS = f"""
WITH age_segments AS (
    SELECT age, balance, y,
    {AGE_GROUP} AS age_group
    FROM campaign_records
), combined_segments AS (
    SELECT age_group, age, balance, y,
    {BALANCE_GROUP} AS balance_group
    FROM age_segments
), segment_statistics AS (
    SELECT age_group, balance_group, {SEGMENT_METRICS}
    FROM combined_segments
    GROUP BY age_group, balance_group
    HAVING COUNT(*) >= ?
)"""

SQL_AGE_BALANCE_SEGMENTS = COMBINED_STATISTICS + f"""
SELECT age_group, balance_group, {DISPLAY_METRICS}
FROM segment_statistics
ORDER BY conversion_rate DESC, total_contacts DESC, age_group, balance_group;
"""

SQL_RANKED_SEGMENTS = COMBINED_STATISTICS + f""", ranked_segments AS (
    SELECT *, DENSE_RANK() OVER (
        PARTITION BY age_group ORDER BY conversion_rate DESC
    ) AS segment_rank
    FROM segment_statistics
)
SELECT age_group, balance_group, {DISPLAY_METRICS}, segment_rank
FROM ranked_segments
WHERE segment_rank <= ?
ORDER BY age_group, segment_rank, total_contacts DESC, balance_group;
"""


def _query_segments(sql, parameters, path):
    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, parameters)]


def query_age_segments(min_records=1, path=DATABASE):
    """Return historical metrics grouped by age range."""
    _positive_integer(min_records, "min_records")
    return _query_segments(SQL_AGE_SEGMENTS, (min_records,), path)


def query_balance_segments(min_records=1, path=DATABASE):
    """Return historical metrics grouped by account balance range."""
    _positive_integer(min_records, "min_records")
    return _query_segments(SQL_BALANCE_SEGMENTS, (min_records,), path)


def query_age_balance_segments(min_records=100, path=DATABASE):
    """Return qualifying age and balance combinations."""
    _positive_integer(min_records, "min_records")
    return _query_segments(SQL_AGE_BALANCE_SEGMENTS, (min_records,), path)


def query_ranked_segments(min_records=100, top_n=3, path=DATABASE):
    """Return the top distinct conversion ranks per age group, including ties."""
    _positive_integer(min_records, "min_records")
    _positive_integer(top_n, "top_n")
    return _query_segments(SQL_RANKED_SEGMENTS, (min_records, top_n), path)
