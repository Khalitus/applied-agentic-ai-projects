from src.features import FEATURES, EXCLUDED


def test_no_current_contact_features_in_predictors():
    assert not set(FEATURES) & set(EXCLUDED)
    assert len(FEATURES) == len(set(FEATURES))
