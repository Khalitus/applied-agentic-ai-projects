import pandas as pd

from src.train import chronological_split


def test_chronological_split_keeps_dates_separate():
    frame = pd.DataFrame({
        "dteday": ["2012-01-01", "2012-01-01", "2012-01-02", "2012-01-03"],
        "hr": [8, 9, 8, 8],
    })

    train, test = chronological_split(frame, train_fraction=0.75)

    assert set(train["dteday"]).isdisjoint(set(test["dteday"]))
    assert train["dteday"].max() < test["dteday"].min()
