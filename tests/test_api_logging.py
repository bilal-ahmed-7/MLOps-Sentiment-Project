"""The API logs numeric features and ids for monitoring, and never the review text."""
import json

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.text_features import FEATURE_NAMES
from src.serving.app import create_app
from src.serving.model_loader import ModelBundle

POS = ["great wonderful loved it", "amazing brilliant acting"] * 10
NEG = ["awful boring waste", "terrible dull plot"] * 10


@pytest.fixture(scope="module")
def bundle():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(POS + NEG, [1] * len(POS) + [0] * len(NEG))
    return ModelBundle(model=pipe, version="3", source="test")


def test_prediction_records_have_features_but_no_text(bundle, tmp_path):
    log_file = tmp_path / "predictions.jsonl"
    client = create_app(bundle=bundle, load=False, prediction_log_path=str(log_file)).test_client()
    secret = "SECRET great wonderful film"
    response = client.post("/predict", json={"texts": [secret, "awful boring waste"]})
    request_id = response.headers["X-Request-ID"]

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [r["prediction_id"] for r in records] == [f"{request_id}-0", f"{request_id}-1"]
    assert {r["label"] for r in records} == {"positive", "negative"}
    assert all(r["model_version"] == "3" and set(FEATURE_NAMES) <= set(r) for r in records)
    assert "SECRET" not in log_file.read_text()


def test_invalid_requests_are_not_logged(bundle, tmp_path):
    log_file = tmp_path / "predictions.jsonl"
    client = create_app(bundle=bundle, load=False, prediction_log_path=str(log_file)).test_client()
    client.post("/predict", json={"text": ""})
    assert not log_file.exists()


def test_logging_is_off_by_default(bundle, tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTION_LOG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    client = create_app(bundle=bundle, load=False).test_client()
    assert client.post("/predict", json={"text": "great film"}).status_code == 200
    assert not list(tmp_path.iterdir())
