import json

from src import app, predict, retrieval


def test_selected_model_is_cached(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    model_path = model_dir / "dummy.joblib"
    model_path.write_bytes(b"placeholder")

    selection_path = model_dir / "model_selection.json"
    selection_path.write_text(
        json.dumps({
            "selected_model": "engineered",
            "feature_mode": "engineered",
            "model_path": "models/dummy.joblib",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(predict, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(predict, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        predict,
        "SELECTION_PATH",
        selection_path,
    )

    calls = []
    fake_model = object()

    def fake_joblib_load(path):
        calls.append(path)
        return fake_model

    monkeypatch.setattr(
        predict.joblib,
        "load",
        fake_joblib_load,
    )

    predict.load_selected_model.cache_clear()

    try:
        first_model, first_selection = (
            predict.load_selected_model()
        )

        second_model, second_selection = (
            predict.load_selected_model()
        )

        assert first_model is second_model
        assert first_selection == second_selection
        assert len(calls) == 1

    finally:
        predict.load_selected_model.cache_clear()


def test_chroma_collections_are_cached(tmp_path, monkeypatch):
    class FakeCollection:
        def count(self):
            return 1

    class FakeClient:
        def get_collection(self, name, embedding_function):
            return FakeCollection()

    client_calls = []

    def fake_persistent_client(path):
        client_calls.append(path)
        return FakeClient()

    monkeypatch.setattr(
        retrieval.chromadb,
        "PersistentClient",
        fake_persistent_client,
    )
    monkeypatch.setattr(
        retrieval,
        "get_embedding_function",
        lambda: object(),
    )
    monkeypatch.setattr(retrieval, "VECTOR_STORE_DIR", tmp_path)

    retrieval.load_collections.cache_clear()

    try:
        first = retrieval.load_collections()
        second = retrieval.load_collections()

        assert first is second
        assert len(client_calls) == 1
    finally:
        retrieval.load_collections.cache_clear()


def test_ui_uses_training_categories():
    expected = predict.get_training_categories()

    for field, values in expected.items():
        assert app.CATEGORY_CHOICES[field] == sorted(values)
