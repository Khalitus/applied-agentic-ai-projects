import sqlite3
import pytest
from src.sql_queries import query_history


def test_sql_and_parameterized_grouping(tmp_path):
    db = tmp_path / "mini.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE campaign_records(job TEXT, education TEXT, poutcome TEXT, y TEXT)")
        conn.executemany("INSERT INTO campaign_records VALUES (?, ?, ?, ?)", [
            ("admin.", "secondary", "unknown", "yes"),
            ("admin.", "secondary", "unknown", "no"),
            ("technician", "tertiary", "success", "yes"),
        ])
    result = query_history(segment="job", min_records=1, path=db)
    assert result["overview"]["total_records"] == 3
    assert result["overview"]["subscriptions"] == 2
    assert len(result["segments"]) == 2
    with pytest.raises(ValueError):
        query_history(segment="job;DROP TABLE campaign_records", min_records=1, path=db)
