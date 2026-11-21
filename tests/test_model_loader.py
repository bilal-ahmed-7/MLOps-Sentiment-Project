"""Tests for loading the serving model (file path and auto fallback)."""
import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.serving import model_loader
from src.serving.model_loader import load_from_file, load_model_bundle


@pytest.fixture
def saved_model(tmp_path):
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(["good great", "bad awful", "good fine", "bad poor"], [1, 0, 1, 0])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    (tmp_path / "metrics.json").write_text(json.dumps({"val_f1": 0.91, "mlflow_run_id": "abc"}))
    return path


def test_load_from_file_returns_working_model_and_metadata(saved_model):
    bundle = load_from_file(str(saved_model))
    assert bundle.source == "file" and bundle.version == "local-file"
    assert bundle.metadata == {"val_f1": 0.91, "source_run_id": "abc"}
    assert bundle.model.predict(["good great"])[0] == 1


def test_model_source_file_env(saved_model, monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "file")
    monkeypatch.setenv("MODEL_PATH", str(saved_model))
    assert load_model_bundle().source == "file"


def test_auto_falls_back_to_file_when_registry_fails(saved_model, monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "auto")
    monkeypatch.setenv("MODEL_PATH", str(saved_model))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///does-not-exist-dir/none.db")

    def broken(*args, **kwargs):
        raise RuntimeError("registry down")

    monkeypatch.setattr(model_loader, "load_from_registry", broken)
    assert load_model_bundle().source == "file"


def test_invalid_model_source_raises(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE", "banana")
    with pytest.raises(ValueError):
        load_model_bundle()
