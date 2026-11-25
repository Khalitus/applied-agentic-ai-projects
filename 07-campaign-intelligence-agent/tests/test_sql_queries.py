import sqlite3

import pytest

from src.sql_queries import query_contact_segments, query_history


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
