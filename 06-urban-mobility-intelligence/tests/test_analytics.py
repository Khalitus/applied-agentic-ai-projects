import pandas as pd

from src.analytics import hourly_profile
from src.db import connect


def test_hourly_profile_uses_bounded_query(tmp_path):
    db = tmp_path / "test.sqlite3"
    with connect(db) as connection:
        pd.DataFrame({"hr": [8, 8, 9], "cnt": [40, 60, 20]}).to_sql(
            "bike_rentals", connection, if_exists="replace", index=False
        )
    result = hourly_profile(8, db)
    assert result["observations"] == 2
    assert result["avg_rentals"] == 50.0
