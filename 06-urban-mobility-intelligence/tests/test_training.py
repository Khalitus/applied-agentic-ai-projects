import json

import joblib
import numpy as np
import pandas as pd
import pytest

import src.train as training
from src.config import FEATURES


def test_baselines_use_training_data_only():
    train = pd.DataFrame({"hr": [8, 8, 17, 17], "cnt": [10, 30, 50, 70]})
    test = pd.DataFrame({"hr": [8, 17, 5], "cnt": [999, 999, 999]})

    overall, hourly = training.baseline_predictions(train, test)

    np.testing.assert_allclose(overall, [40, 40, 40])
    np.testing.assert_allclose(hourly, [20, 60, 40])


def test_regression_metrics_use_correct_units():
    result = training.regression_metrics([10, 20], [12, 16])

    assert result["mae"] == 3.0
    assert result["rmse"] == pytest.approx(np.sqrt(10), abs=0.001)


def test_chronological_split_rejects_invalid_fraction():
    frame = pd.DataFrame({"dteday": ["2012-01-01", "2012-01-02"], "hr": [8, 8]})

    with pytest.raises(ValueError):
        training.chronological_split(frame, train_fraction=1.0)


def test_training_saves_model_metrics_and_holdout(monkeypatch, tmp_path):
    days = pd.date_range("2012-01-01", periods=10)
    records = []
    for index, day in enumerate(days):
        for hour in (8, 17):
            weekday = (day.weekday() + 1) % 7
            workingday = int(weekday not in (0, 6))
            records.append({
                "instant": len(records) + 1,
                "dteday": day.strftime("%Y-%m-%d"),
                "mnth": day.month,
                "hr": hour,
                "holiday": 0,
                "weekday": weekday,
                "workingday": workingday,
                "weathersit": 1,
                "temp": 0.5,
                "hum": 0.6,
                "windspeed": 0.2,
                "cnt": 25 + 5 * index + (hour == 17) * 12,
            })

    monkeypatch.setattr(training, "load_hourly", lambda: pd.DataFrame(records))
    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(training, "REPORT_PATH", tmp_path / "model_metrics.json")
    monkeypatch.setattr(training, "PREDICTIONS_PATH", tmp_path / "holdout.csv")

    result = training.main()
    saved = json.loads(training.REPORT_PATH.read_text(encoding="utf-8"))
    holdout = pd.read_csv(training.PREDICTIONS_PATH)
    model = joblib.load(training.MODEL_PATH)

    assert saved == result
    assert result["train_rows"] == 16
    assert result["test_rows"] == 4
    assert result["train_end"] < result["test_start"]
    assert result["features"] == list(FEATURES)
    assert len(holdout) == 4
    assert (holdout["predicted_rentals"] >= 0).all()
    assert (holdout["absolute_error"] >= 0).all()
    assert list(model.feature_names_in_) == list(FEATURES)
