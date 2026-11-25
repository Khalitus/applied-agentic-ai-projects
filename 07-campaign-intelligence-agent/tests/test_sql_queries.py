import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.sql_queries import (
    query_age_balance_segments,
    query_age_segments,
    query_balance_segments,
    query_contact_segments,
    query_history,
    query_ranked_segments,
)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    rows = [
        ("admin.", "secondary", "success", "yes", 100, 10),
        ("admin.", "secondary", "success", "no", 300, 20),
        ("technician", "tertiary", "success", "yes", 400, 30),
        ("technician", "tertiary", "success", "yes", 600, 40),
        ("technician", "tertiary", "success", "no", 500, 50),
        ("student", "secondary", "unknown", "yes", 200, -1),
        ("admin.", "secondary", "success", "yes", 0, 10),
        ("admin.", "secondary", "success", "yes", -20, 10),
        ("retired", "primary", "failure", "yes", 100, 20),
    ]
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE campaign_records (
                job TEXT, education TEXT, poutcome TEXT, y TEXT,
                balance INTEGER, pdays INTEGER
            )
        """)
        connection.executemany("INSERT INTO campaign_records VALUES (?, ?, ?, ?, ?, ?)", rows)
    return path


@pytest.mark.parametrize("dimension", ["job", "education", "poutcome"])
def test_history(database, dimension):
    result = query_history(dimension, min_records=1, path=database)
    assert result["grouped_by"] == dimension
    assert result["overview"]["total_records"] == 9
    assert result["overview"]["subscriptions"] == 7
    assert sum(row["total_records"] for row in result["segments"]) == 9
    assert sum(row["non_subscriptions"] for row in result["segments"]) == 2
    if dimension == "job":
        admin = next(row for row in result["segments"] if row["segment"] == "admin.")
        assert admin["average_balance"] == 95.0
        assert admin["subscription_rate_percent"] == 75.0


def test_contact_segments_filters_and_inclusive_threshold(database):
    rows = query_contact_segments(min_records=2, path=database)
    assert [row["job"] for row in rows] == ["technician", "admin."]
    assert rows[0]["total_contacts"] == 3
    assert rows[0]["conversion_rate"] == 66.67
    assert rows[1]["total_contacts"] == 2
    assert rows[1]["subscriptions"] == 1
    assert rows[1]["conversion_rate"] == 50.0
    assert rows[1]["average_balance"] == 200.0
    assert query_contact_segments(min_records=2, limit=1, path=database) == rows[:1]


def test_contact_segments_sort_ties(database):
    with sqlite3.connect(database) as connection:
        connection.executemany("INSERT INTO campaign_records VALUES (?, ?, ?, ?, ?, ?)", [
            ("services", "primary", "failure", "yes", 100, 20),
            ("services", "primary", "failure", "yes", 100, 20),
        ])
    rows = query_contact_segments(min_records=1, path=database)
    assert [row["job"] for row in rows[:2]] == ["services", "retired"]


def test_history_threshold(database):
    assert len(query_history("job", min_records=4, path=database)["segments"]) == 1


def test_invalid_dimension(database):
    with pytest.raises(ValueError):
        query_history("job; DROP TABLE campaign_records", path=database)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_parameters(database, value):
    with pytest.raises(ValueError):
        query_history(min_records=value, path=database)
    with pytest.raises(ValueError):
        query_contact_segments(min_records=value, path=database)
    with pytest.raises(ValueError):
        query_contact_segments(limit=value, path=database)


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        query_history(path=path)
    assert not path.exists()


def _segmentation_database(path, rows):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE campaign_records (age INTEGER, balance INTEGER, y TEXT)
            """)
            connection.executemany("INSERT INTO campaign_records VALUES (?, ?, ?)", rows)
    return path


@pytest.fixture
def segmentation_database(tmp_path):
    return _segmentation_database(tmp_path / "segments.sqlite3", [
        (29, -1, "yes"), (30, 0, "no"), (44, 1, "yes"),
        (45, 999, "no"), (59, 1000, "yes"), (60, 4999, "no"),
        (61, 5000, "yes"), (None, None, "no"),
    ])


@pytest.mark.parametrize("function,group,expected", [
    (query_age_segments, "age_group", {
        "Under 30": 1, "30-44": 2, "45-59": 2, "60+": 2, "Unknown": 1,
    }),
    (query_balance_segments, "balance_group", {
        "Negative": 1, "Zero": 1, "Low": 2, "Medium": 2, "High": 1, "Unknown": 1,
    }),
])
def test_segmentation_boundaries(segmentation_database, function, group, expected):
    rows = function(path=segmentation_database)
    assert {r[group]: r["total_contacts"] for r in rows} == expected
    assert sum(r["subscriptions"] for r in rows) == 4
    assert sum(r["non_subscriptions"] for r in rows) == 4
    unknown = next(r for r in rows if r[group] == "Unknown")
    assert unknown["average_age"] is None
    assert unknown["average_balance"] is None
    assert all(r["total_contacts"] >= 2 for r in function(2, path=segmentation_database))


def test_combined_segmentation_and_metrics(tmp_path):
    path = _segmentation_database(tmp_path / "combined.sqlite3", [
        (30, 10, "yes"), (44, 30, "no"), (45, 999, "yes"),
        (31, 1000, "yes"), (32, 4999, "yes"), (60, -1, "no"),
    ])
    rows = query_age_balance_segments(2, path=path)
    assert [(r["age_group"], r["balance_group"]) for r in rows] == [
        ("30-44", "Medium"), ("30-44", "Low"),
    ]
    assert rows[1] == {
        "age_group": "30-44", "balance_group": "Low", "total_contacts": 2,
        "subscriptions": 1, "non_subscriptions": 1, "conversion_rate": 50.0,
        "average_age": 37.0, "average_balance": 20.0,
    }
    assert len(query_age_balance_segments(1, path=path)) == 4


def test_rank_partitions_ties_and_filtering(tmp_path):
    path = _segmentation_database(tmp_path / "ranks.sqlite3", [
        (25, -1, "yes"),  # Excluded before ranking.
        (25, 1, "yes"), (25, 1, "no"),
        (25, 1000, "yes"), (25, 1000, "no"),
        (25, 5000, "no"), (25, 5000, "no"),
        (35, 1, "no"), (35, 1, "no"),
        (35, 5000, "yes"), (35, 5000, "no"),
    ])
    rows = query_ranked_segments(2, 2, path=path)
    assert [(r["age_group"], r["balance_group"], r["segment_rank"]) for r in rows] == [
        ("30-44", "High", 1), ("30-44", "Low", 2),
        ("Under 30", "Low", 1), ("Under 30", "Medium", 1),
        ("Under 30", "High", 2),
    ]
    assert len(query_ranked_segments(2, 1, path=path)) == 3


def test_ranking_uses_unrounded_rates(tmp_path):
    rows = [(25, 1, "yes")] + [(25, 1, "no")] * 4999
    rows += [(25, 1000, "yes")] + [(25, 1000, "no")] * 5000
    path = _segmentation_database(tmp_path / "precision.sqlite3", rows)
    ranked = query_ranked_segments(1, 2, path=path)
    assert [r["conversion_rate"] for r in ranked] == [0.02, 0.02]
    assert [r["balance_group"] for r in ranked] == ["Low", "Medium"]
    assert [r["segment_rank"] for r in ranked] == [1, 2]


@pytest.mark.parametrize("function", [query_age_segments, query_balance_segments,
                                     query_age_balance_segments, query_ranked_segments])
def test_empty_and_missing_segmentation_database(tmp_path, function):
    path = _segmentation_database(tmp_path / "empty.sqlite3", [])
    assert function(path=path) == []
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        function(path=missing)
    assert not missing.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_segmentation_parameters(segmentation_database, value):
    for function in (query_age_segments, query_balance_segments,
                     query_age_balance_segments, query_ranked_segments):
        with pytest.raises(ValueError, match="min_records"):
            function(min_records=value, path=segmentation_database)
    with pytest.raises(ValueError, match="top_n"):
        query_ranked_segments(top_n=value, path=segmentation_database)


def test_standalone_segmentation_sql_matches_api(tmp_path):
    path = _segmentation_database(tmp_path / "standalone.sqlite3", [
        (25, 1, "yes"), (25, 1, "no"), (35, 1000, "yes"), (60, 5000, "no"),
    ] * 100)
    script = (Path(__file__).resolve().parents[1] / "sql" / "analysis.sql").read_text()
    statements = [part.strip() for part in script.split(';') if part.strip()][-4:]
    functions = (query_age_segments, query_balance_segments,
                 query_age_balance_segments, query_ranked_segments)
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        for statement, function in zip(statements, functions, strict=True):
            assert [dict(r) for r in connection.execute(statement)] == function(path=path)
