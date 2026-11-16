"""TODO: implement tests/test_api.py"""
"""API tests: valid requests, malformed input, error responses, metrics and privacy."""
import logging

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.serving.app import MAX_BATCH, MAX_TEXT_CHARS, create_app
from src.serving.model_loader import ModelBundle

POS = ["great wonderful loved it", "amazing brilliant acting", "fantastic moving film"] * 10
NEG = ["awful boring waste", "terrible dull plot", "hated horrible movie"] * 10


@pytest.fixture(scope="module")
def model():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    return pipe.fit(POS + NEG, [1] * len(POS) + [0] * len(NEG))


@pytest.fixture
def client(model):
    bundle = ModelBundle(model=model, version="7", source="test", metadata={"val_f1": "0.9"})
    return create_app(bundle=bundle, load=False).test_client()


@pytest.fixture
def client_no_model():
    return create_app(bundle=None, load=False).test_client()


# ---- /health and /model-info -------------------------------------------------
def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True, "model_version": "7"}


def test_health_unavailable_without_model(client_no_model):
    assert client_no_model.get("/health").status_code == 503


def test_model_info(client):
    body = client.get("/model-info").get_json()
    assert body["model_version"] == "7" and body["source"] == "test"
    assert body["labels"] == {"0": "negative", "1": "positive"}
    assert body["metadata"]["val_f1"] == "0.9"


def test_model_info_without_model(client_no_model):
    assert client_no_model.get("/model-info").status_code == 503


# ---- /predict: valid ------------------------------------------------------------
def test_predict_single_text(client):
    response = client.post("/predict", json={"text": "great wonderful loved it"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"] == "7"
    prediction = body["predictions"][0]
    assert prediction["label"] == "positive" and prediction["class_id"] == 1
    assert 0.5 <= prediction["confidence"] <= 1.0


def test_predict_batch(client):
    response = client.post("/predict", json={"texts": ["great film", "awful boring waste"]})
    labels = [p["label"] for p in response.get_json()["predictions"]]
    assert labels == ["positive", "negative"]


def test_predict_applies_same_cleaning_as_training(client):
    plain = client.post("/predict", json={"text": "great wonderful film"}).get_json()
    html = client.post("/predict", json={"text": "great<br /><br />wonderful   film"}).get_json()
    assert plain["predictions"] == html["predictions"]


def test_response_has_request_id_header(client):
    assert client.get("/health").headers["X-Request-ID"]


# ---- /predict: invalid input ----------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "<br /><br />"},
        {"text": 123},
        {"text": None},
        {"texts": []},
        {"texts": "not a list"},
        {"texts": ["ok text", 5]},
        {"text": "a", "texts": ["b"]},
        {"text": "x" * (MAX_TEXT_CHARS + 1)},
        {"texts": ["ok text"] * (MAX_BATCH + 1)},
        ["a list is not an object"],
    ],
)
def test_predict_rejects_invalid_payload(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_malformed_json(client):
    response = client.post("/predict", data="{not json", content_type="application/json")
    assert response.status_code == 400


def test_predict_rejects_wrong_content_type(client):
    response = client.post("/predict", data="great film", content_type="text/plain")
    assert response.status_code == 415


def test_predict_rejects_oversized_body(client):
    response = client.post("/predict", data="x" * 1_100_000, content_type="application/json")
    assert response.status_code == 413


def test_predict_without_model_returns_503(client_no_model):
    assert client_no_model.post("/predict", json={"text": "great film"}).status_code == 503


# ---- errors are JSON ------------------------------------------------------------
def test_unknown_route_returns_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404 and "error" in response.get_json()


def test_wrong_method_returns_json_405(client):
    response = client.get("/predict")
    assert response.status_code == 405 and "error" in response.get_json()


# ---- metrics and privacy --------------------------------------------------------
def test_metrics_endpoint_counts_requests_and_predictions(client):
    client.post("/predict", json={"text": "great wonderful loved it"})
    client.post("/predict", json={"text": ""})
    text = client.get("/metrics").get_data(as_text=True)
    assert 'api_requests_total{endpoint="/predict",method="POST",status="200"} 1.0' in text
    assert 'api_requests_total{endpoint="/predict",method="POST",status="400"} 1.0' in text
    assert 'predictions_total{label="positive",model_version="7"} 1.0' in text
    assert "input_validation_failures_total 1.0" in text
    assert "model_loaded 1.0" in text


def test_review_text_is_never_logged(client, caplog):
    secret = "SUPERSECRET-review-text-12345"
    with caplog.at_level(logging.DEBUG):
        client.post("/predict", json={"text": secret})
        client.post("/predict", json={"texts": [secret, 5]})  # invalid request path too
    assert secret not in caplog.text