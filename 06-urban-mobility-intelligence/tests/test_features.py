import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.config import FEATURES
from src.features import scenario_features, training_features

def test_calendar_feature_mapping():
    # 2012-08-15 was Wednesday: UCI weekday=3 (Sunday=0).
    result = scenario_features("2012-08-15", 8, False, 1, 22, 60, 12)
    assert result.iloc[0]["weekday"] == 3
    assert result.iloc[0]["workingday"] == 1
    assert list(result.columns) == list(FEATURES)
    assert "casual" not in result.columns
    assert "registered" not in result.columns


def test_holiday_is_not_workingday():
    result = scenario_features("2012-08-15", 8, True, 1, 22, 60, 12)
    assert result.iloc[0]["workingday"] == 0


def test_commute_peak_on_working_day():
    result = scenario_features(
        "2012-08-15", 8, False, 1, 22, 60, 12
    )

    assert result.iloc[0]["commute_peak"] == 1
    assert list(result.columns) == list(FEATURES)


def test_commute_peak_is_zero_on_non_working_day():
    result = scenario_features(
        "2012-08-18", 8, False, 1, 22, 60, 12
    )

    assert result.iloc[0]["workingday"] == 0
    assert result.iloc[0]["commute_peak"] == 0


def test_training_scenario_feature_parity():
    raw = pd.DataFrame([{
        "mnth": 8,
        "hr": 8,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        "temp": 22 / 41,
        "hum": 0.60,
        "windspeed": 12 / 67,
    }])

    training = training_features(raw)
    scenario = scenario_features(
        "2012-08-15", 8, False, 1, 22, 60, 12
    )

    assert_frame_equal(training, scenario, check_exact=False)


@pytest.mark.parametrize(
    "scenario",
    [
        ("2012-08-15", 25, False, 1, 22, 60, 12),
        ("2012-08-15", 8, False, 9, 22, 60, 12),
        ("2012-08-15", 8, False, 1, 22, 120, 12),
        ("not-a-date", 8, False, 1, 22, 60, 12),
    ],
)
def test_invalid_scenario_inputs_raise_value_error(scenario):
    with pytest.raises(ValueError):
        scenario_features(*scenario)
