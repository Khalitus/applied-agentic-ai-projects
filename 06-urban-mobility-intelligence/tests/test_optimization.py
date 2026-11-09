import json

import joblib
import numpy as np
import pandas as pd
import pytest

import src.optimize as optimization
from src.config import FEATURES
from src.train import chronological_split


def test_selection_keeps_baseline_for_marginal_gain():
    results = [
        {"name": "task6_baseline", "mae": 100.0},
        {"name": "shallow_regularized", "mae": 99.8},
    ]
    assert optimization.choose_candidate(results) == "task6_baseline"


def test_selection_chooses_meaningful_validation_gain():
    results = [
        {"name": "task6_baseline", "mae": 100.0},
        {"name": "shallow_regularized", "mae": 95.0},
    ]
    assert optimization.choose_candidate(results) == "shallow_regularized"


def test_prediction_frame_and_error_profiles():
    data = pd.DataFrame({
        "instant": [1, 2, 3],
        "dteday": ["2012-01-01", "2012-01-01", "2012-01-02"],
        "hr": [8, 8, 17],
        "workingday": [1, 1, 0],
        "weathersit": [1, 2, 3],
        "cnt": [50, 150, 350],
    })
    rows = optimization.prediction_frame(data, np.array([60, 130, 300]))
    np.testing.assert_allclose(rows["absolute_error"], [10, 20, 50])
    np.testing.assert_allclose(rows["signed_error"], [10, -20, -50])
    bands = optimization.error_profile(rows, "demand_band")
    assert [item["group"] for item in bands] == [
        "low_0_99", "medium_100_299", "high_300_plus"
    ]
    assert sum(item["observations"] for item in bands) == 3
    assert bands[-1]["mae"] == 50.0


def test_inner_split_never_reaches_outer_holdout():
    df = pd.DataFrame({
        "dteday": [f"2012-01-{i:02d}" for i in range(1, 21) for _ in range(2)],
        "hr": [8, 17] * 20,
    })
    train, holdout = chronological_split(df)
    fit, validation = chronological_split(train)
    assert fit["dteday"].max() < validation["dteday"].min()
    assert validation["dteday"].max() < holdout["dteday"].min()


def test_selection_rejects_invalid_scores():
    with pytest.raises(ValueError):
        optimization.choose_candidate([{"name": "task6_baseline", "mae": float("nan")}])


def test_optimization_pipeline_saves_separate_artifacts(monkeypatch, tmp_path):
    records = []
    for idx, day in enumerate(pd.date_range("2012-01-01", periods=25)):
        for hour in (8, 12, 17):
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
                "weathersit": 1 + idx % 3,
                "temp": 0.3 + idx / 100,
                "hum": 0.5,
                "windspeed": 0.2,
                "cnt": 20 + idx * 4 + (hour == 17) * 25,
            })
    monkeypatch.setattr(optimization, "load_hourly", lambda: pd.DataFrame(records))
    monkeypatch.setattr(optimization, "OPTIMIZED_MODEL_PATH", tmp_path / "optimized.joblib")
    monkeypatch.setattr(optimization, "OPTIMIZATION_REPORT_PATH", tmp_path / "optimization.json")
    monkeypatch.setattr(optimization, "VALIDATION_PREDICTIONS_PATH", tmp_path / "validation.csv")
    monkeypatch.setattr(optimization, "OPTIMIZED_PREDICTIONS_PATH", tmp_path / "holdout.csv")
    monkeypatch.setattr(optimization, "REPORT_PATH", tmp_path / "missing_task6_metrics.json")
    monkeypatch.setattr(optimization, "CANDIDATES", {
        "task6_baseline": {
            "n_estimators": 8, "max_depth": 2, "learning_rate": 0.1,
        },
        "shallow_regularized": {
            "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
            "reg_lambda": 2,
        },
    })

    report = optimization.main()
    saved = json.loads(optimization.OPTIMIZATION_REPORT_PATH.read_text())
    validation_rows = pd.read_csv(optimization.VALIDATION_PREDICTIONS_PATH)
    holdout_rows = pd.read_csv(optimization.OPTIMIZED_PREDICTIONS_PATH)
    model = joblib.load(optimization.OPTIMIZED_MODEL_PATH)

    assert report == saved
    assert report["outer_split"]["training_rows"] == 60
    assert report["outer_split"]["holdout_rows"] == 15
    assert report["inner_split"]["fit_rows"] == 48
    assert report["inner_split"]["validation_rows"] == 12
    assert len(validation_rows) == 12
    assert len(holdout_rows) == 15
    assert (holdout_rows["predicted_rentals"] >= 0).all()
    assert list(model.feature_names_in_) == list(FEATURES)
    assert report["selected_candidate"] in optimization.CANDIDATES
    assert report["task6_holdout_reference"] is None
    assert sum(item["observations"] for item in report["validation_error_analysis"]["by_hour"]) == 12
