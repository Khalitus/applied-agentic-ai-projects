
import pandas as pd
import pytest

from src.analytics import (
    compare_workdays,
    historical_reference,
    hourly_demand,
    hourly_profile,
    weather_impact,
)
from src.db import connect


@pytest.fixture
def sample_db(tmp_path):
    db = tmp_path / "analytics_test.sqlite3"

    sample = pd.DataFrame({
        "hr": [8, 8, 8, 17, 17, 17],
        "workingday": [1, 1, 0, 1, 0, 0],
        "weathersit": [1, 2, 1, 3, 2, 3],
        "cnt": [40, 60, 20, 80, 30, 10],
    })

    with connect(db) as connection:
        sample.to_sql(
            "bike_rentals",
            connection,
            if_exists="replace",
            index=False,
        )

    return db


def test_hourly_profile(sample_db):
    result = hourly_profile(8, sample_db)

    assert result["hr"] == 8
    assert result["observations"] == 3
    assert result["avg_rentals"] == 40.0


def test_compare_workdays(sample_db):
    results = compare_workdays(sample_db)

    assert len(results) == 2

    non_working, working = results

    assert non_working["workingday"] == 0
    assert non_working["observations"] == 3
    assert non_working["avg_rentals"] == 20.0

    assert working["workingday"] == 1
    assert working["observations"] == 3
    assert working["avg_rentals"] == 60.0


def test_weather_impact(sample_db):
    results = weather_impact(sample_db)

    category = next(row for row in results if row["weathersit"] == 1)

    assert category["observations"] == 2
    assert category["avg_rentals"] == 30.0
    assert category["min_rentals"] == 20
    assert category["max_rentals"] == 40


def test_hourly_demand(sample_db):
    results = hourly_demand(workingday=1, path=sample_db)

    hour_8 = next(row for row in results if row["hr"] == 8)

    assert hour_8["observations"] == 2
    assert hour_8["avg_rentals"] == 50.0


def test_historical_reference_valid(sample_db):
    result = historical_reference(
        hour=8,
        workingday=1,
        weathersit=1,
        path=sample_db,
    )

    assert result["hour"] == 8
    assert result["observations"] == 1
    assert result["avg_rentals"] == 40.0
    assert result["min_rentals"] == 40
    assert result["max_rentals"] == 40


def test_historical_reference_empty(sample_db):
    result = historical_reference(hour=3, path=sample_db)

    assert result["observations"] == 0
    assert result["avg_rentals"] is None
    assert result["min_rentals"] is None
    assert result["max_rentals"] is None


def test_hourly_demand_invalid_workingday(sample_db):
    with pytest.raises(ValueError):
        hourly_demand(workingday=25, path=sample_db)


def test_historical_reference_invalid_hour(sample_db):
    with pytest.raises(ValueError):
        historical_reference(hour=25, path=sample_db)


def test_historical_reference_invalid_weather(sample_db):
    with pytest.raises(ValueError):
        historical_reference(hour=8, weathersit=99, path=sample_db)