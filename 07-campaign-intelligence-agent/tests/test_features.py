import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import ml
from src.features import (
    CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, TARGET,
    build_preprocessor, prepare_features, split_dataset,
)


@pytest.fixture
def source_frame():
    rows = []
    for i in range(120):
        rows.append({
            "age": 20 + i % 50,
            "balance": i * 30 - 1000,
            "pdays": -1 if i % 3 == 0 else i,
            "previous": i % 5,
            "job": "student" if i % 2 else "management",
            "marital": "single" if i % 2 else "married",
            "education": "unknown" if i % 3 else "tertiary",
            "default": "no",
            "housing": "yes" if i % 2 else "no",
            "loan": "no" if i % 3 else "yes",
            "poutcome": "unknown" if i % 3 else "success",
            "y": "yes" if i % 3 == 0 and i < 102 else "no",
            "duration": i * 2,
            "campaign": 1,
            "contact": "cellular",
            "day": 1,
            "month": "may",
        })
    frame = pd.DataFrame(rows)
    frame.loc[84:, "job"] = "validation_only"
    frame.loc[84:, "balance"] += 5000
    return frame


def test_no_current_contact_features_in_predictors():
    assert not set(FEATURES) & set(EXCLUDED)
    assert len(FEATURES) == len(set(FEATURES))
    assert TARGET not in FEATURES
    assert len(NUMERIC) == 4 and len(CATEGORICAL) == 7


def test_prepare_features_preserves_contract_and_labels(source_frame):
    X, y = prepare_features(source_frame)
    assert X.columns.tolist() == FEATURES
    assert y.tolist() == source_frame[TARGET].map({"yes": 1, "no": 0}).tolist()
    assert pd.api.types.is_integer_dtype(y)
    assert X.index.equals(y.index)
    X.loc[0, "age"] = 99
    assert source_frame.loc[0, "age"] != 99


@pytest.mark.parametrize("column", ["age", "job", TARGET])
def test_missing_source_columns_rejected(source_frame, column):
    with pytest.raises(ValueError, match="Missing source columns"):
        prepare_features(source_frame.drop(columns=column))


def test_duplicate_columns_rejected(source_frame):
    with pytest.raises(ValueError, match="Duplicate source columns"):
        prepare_features(pd.concat([source_frame, source_frame[["age"]]], axis=1))


@pytest.mark.parametrize("label", ["unknown", "YES", None])
def test_invalid_target_rejected(source_frame, label):
    source_frame.loc[0, TARGET] = label
    with pytest.raises(ValueError, match="Unrecognized target label"):
        prepare_features(source_frame)


@pytest.mark.parametrize("column", ["age", "job"])
def test_missing_features_rejected(source_frame, column):
    source_frame.loc[0, column] = None
    with pytest.raises(ValueError, match="Missing prediction feature"):
        prepare_features(source_frame)


def test_non_numeric_features_rejected(source_frame):
    source_frame["age"] = source_frame["age"].astype(str)
    with pytest.raises(ValueError, match="numeric dtypes"):
        prepare_features(source_frame)


def test_non_finite_features_rejected(source_frame):
    source_frame["balance"] = source_frame["balance"].astype(float)
    source_frame.loc[0, "balance"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        prepare_features(source_frame)


def test_non_string_categories_rejected(source_frame):
    source_frame.loc[0, "job"] = 123
    with pytest.raises(ValueError, match="must contain strings"):
        prepare_features(source_frame)


def test_source_order_partitions(source_frame):
    source_frame.index = np.arange(120)[::-1]
    X, y = prepare_features(source_frame)
    parts = split_dataset(X, y)
    assert [len(a) for a, _ in parts] == [84, 18, 18]
    for a, b in parts:
        assert a.index.equals(b.index)
    assert pd.concat([a for a, _ in parts]).equals(X)
    assert pd.concat([b for _, b in parts]).equals(y)
    assert all(a.index.intersection(b.index).empty for i, (a, _) in enumerate(parts) for b, _ in parts[i+1:])


def test_split_rejects_misalignment_and_duplicate_indices(source_frame):
    X, y = prepare_features(source_frame)
    with pytest.raises(ValueError, match="aligned and unique"):
        split_dataset(X, y.iloc[::-1])
    X.index = y.index = [0] * len(X)
    with pytest.raises(ValueError, match="aligned and unique"):
        split_dataset(X, y)


def test_split_rejects_empty_partitions(source_frame):
    X, y = prepare_features(source_frame.iloc[:2])
    with pytest.raises(ValueError, match="too small"):
        split_dataset(X, y)


def test_preprocessing_training_parameters_and_unknown_categories(source_frame):
    X, y = prepare_features(source_frame)
    (train, _), (val, _), _ = split_dataset(X, y)
    pre = build_preprocessor()
    a = pre.fit_transform(train)
    b = pre.transform(val)
    scaler = pre.named_transformers_["numeric"]
    encoder = pre.named_transformers_["categorical"]
    assert scaler.n_samples_seen_ == len(train)
    np.testing.assert_allclose(scaler.mean_, train[NUMERIC].mean())
    np.testing.assert_allclose(scaler.var_, train[NUMERIC].var(ddof=0))
    np.testing.assert_allclose(a[:, :4].toarray().mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(a[:, :4].toarray().std(axis=0), 1)
    for col, categories in zip(CATEGORICAL, encoder.categories_):
        assert set(categories) == set(train[col])
    names = pre.get_feature_names_out()
    job = np.char.startswith(names.astype(str), "categorical__job_")
    assert b[:, job].nnz == 0
    assert np.all(np.asarray(a[:, 4:].sum(axis=1)) == 7)
    assert np.all(np.asarray(b[:, 4:].sum(axis=1)) == 6)
    assert "categorical__education_unknown" in names
    assert issparse(a) and issparse(b)
    assert a.shape[1] == b.shape[1] == len(set(names))
    assert np.isfinite(a.data).all() and np.isfinite(b.data).all()


def test_preprocessor_equivalence_and_single_row(source_frame):
    X, y = prepare_features(source_frame)
    (train, _), (val, _), _ = split_dataset(X, y)
    pre = build_preprocessor()
    actual = pre.fit_transform(train)
    num = StandardScaler().fit_transform(train[NUMERIC])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit_transform(train[CATEGORICAL])
    np.testing.assert_allclose(actual.toarray(), np.hstack([num, cat.toarray()]))
    np.testing.assert_allclose(pre.transform(val.iloc[[0]]).toarray(), pre.transform(val)[:1].toarray())
    np.testing.assert_allclose(pre.transform(train.assign(duration=999)).toarray(), actual.toarray())
    assert not hasattr(build_preprocessor(), "transformers_")


def test_preprocessor_serialization(source_frame, tmp_path):
    X, y = prepare_features(source_frame)
    (train, _), (val, _), _ = split_dataset(X, y)
    pre = build_preprocessor().fit(train)
    path = tmp_path / "preprocessing.joblib"
    joblib.dump(pre, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.get_feature_names_out(), pre.get_feature_names_out())
    np.testing.assert_allclose(restored.transform(val).toarray(), pre.transform(val).toarray())


def test_baseline_training_reserves_holdout(source_frame, tmp_path, monkeypatch):
    raw = tmp_path / "source.csv"
    source_frame.to_csv(raw, sep=";", index=False)
    monkeypatch.setattr(ml, "RAW_CSV", raw)
    monkeypatch.setattr(ml, "MODEL", tmp_path / "artifacts" / "model.joblib")
    monkeypatch.setattr(ml, "METRICS", tmp_path / "reports" / "metrics.json")
    evaluated = []
    original_evaluate = ml.evaluate

    def record_evaluation(model, rows, labels):
        evaluated.append(rows.index.tolist())
        return original_evaluate(model, rows, labels)

    monkeypatch.setattr(ml, "evaluate", record_evaluation)
    ml.main()
    assert evaluated == [list(range(84)), list(range(84, 102))]
    metrics = json.loads(ml.METRICS.read_text())
    assert "test" not in metrics
    assert metrics["holdout_status"] == "reserved; not evaluated"
    model = joblib.load(ml.MODEL)
    pre = model.named_steps["preprocessing"]
    assert pre.named_transformers_["numeric"].n_samples_seen_ == 84
    assert "validation_only" not in pre.named_transformers_["categorical"].categories_[0]
    assert model.predict_proba(source_frame.iloc[[84]][FEATURES]).shape == (1, 2)
