from src.config import FEATURES
from src.features import scenario_features


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
