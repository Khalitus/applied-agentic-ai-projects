import sqlite3

import numpy as np
import pytest

import src.predict as prediction
from src.config import FEATURES


SCENARIO = {
    "date_text": "2012-08-15",
    "hour": 8,
    "holiday": False,
    "weather_code": 1,
    "temperature_c": 22,
    "humidity_pct": 60,
    "windspeed_kmh": 12,
}


class ModelStub:
    def __init__(self, value=471.7, names=FEATURES):
        self.value = value
        self.feature_names_in_ = np.asarray(names)

    def predict(self, features):
        return np.asarray([self.value])


@pytest.fixture
def model_stub(tmp_path, monkeypatch):
    path = tmp_path / "optimized.joblib"
    path.write_bytes(b"test-stub")
    monkeypatch.setattr(prediction, "OPTIMIZED_MODEL_PATH", path)
    monkeypatch.setattr(prediction.joblib, "load", lambda file: ModelStub())
    return path


@pytest.fixture
def historical_db(tmp_path, monkeypatch):
    path = tmp_path / "bike_demand.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE bike_rentals (hr INTEGER, workingday INTEGER, "
            "weathersit INTEGER, cnt INTEGER)"
        )
        connection.executemany(
            "INSERT INTO bike_rentals VALUES (?, ?, ?, ?)",
            [
                (8, 1, 1, 40),
                (8, 1, 1, 60),
                (8, 1, 2, 200),
                (8, 0, 1, 20),
                (17, 1, 1, 80),
            ],
        )
    monkeypatch.setattr(prediction, "DB_PATH", path)
    return path


def test_prediction_uses_optimized_model(model_stub):
    assert prediction.predict_demand(**SCENARIO) == 471.7


def test_estimate_scenario_matches_historical_conditions(model_stub, historical_db):
    result = prediction.estimate_scenario(**SCENARIO)

    assert result["estimated_rentals"] == 471.7
    assert result["model"] == "optimized_xgboost"
    assert result["scenario"]["workingday"] == 1
    assert result["historical_reference"]["observations"] == 2
    assert result["historical_reference"]["avg_rentals"] == 50.0
    assert result["reference_sample"] == "small_sample"


def test_empty_reference_does_not_invent_an_average(model_stub, historical_db):
    result = prediction.estimate_scenario(**{**SCENARIO, "hour": 3})

    assert result["reference_sample"] == "no_matches"
    assert result["historical_reference"]["observations"] == 0
    assert result["historical_reference"]["avg_rentals"] is None


def test_invalid_scenario_is_rejected_before_loading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(prediction, "OPTIMIZED_MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(ValueError, match="Hour"):
        prediction.predict_demand(**{**SCENARIO, "hour": 25})


def test_missing_model_has_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "OPTIMIZED_MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="python -m src.optimize"):
        prediction.predict_demand(**SCENARIO)


def test_model_feature_mismatch_is_rejected(model_stub, monkeypatch):
    monkeypatch.setattr(
        prediction.joblib,
        "load",
        lambda file: ModelStub(names=FEATURES[:-1]),
    )
    with pytest.raises(RuntimeError, match="feature contract mismatch"):
        prediction.predict_demand(**SCENARIO)


@pytest.mark.parametrize("value, expected", [(-7.0, 0.0), (471.74, 471.7)])
def test_prediction_is_nonnegative_and_rounded(model_stub, monkeypatch, value, expected):
    monkeypatch.setattr(prediction.joblib, "load", lambda file: ModelStub(value=value))
    assert prediction.predict_demand(**SCENARIO) == expected


def test_nonfinite_prediction_is_rejected(model_stub, monkeypatch):
    monkeypatch.setattr(prediction.joblib, "load", lambda file: ModelStub(value=np.nan))
    with pytest.raises(RuntimeError, match="invalid prediction"):
        prediction.predict_demand(**SCENARIO)


def test_missing_database_has_actionable_message(model_stub, tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "DB_PATH", tmp_path / "missing.sqlite3")
    with pytest.raises(FileNotFoundError, match="python -m src.db"):
        prediction.estimate_scenario(**SCENARIO)


def test_malformed_database_is_reported(model_stub, tmp_path, monkeypatch):
    path = tmp_path / "invalid.sqlite3"
    sqlite3.connect(path).close()
    monkeypatch.setattr(prediction, "DB_PATH", path)
    with pytest.raises(RuntimeError, match="database cannot be queried"):
        prediction.estimate_scenario(**SCENARIO)
