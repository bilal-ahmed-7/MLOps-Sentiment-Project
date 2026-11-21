"""Flask REST API for sentiment prediction.

Endpoints:
    GET  /health      liveness/readiness (503 if no model is loaded)
    GET  /model-info  model version and metadata
    POST /predict     {"text": "..."} or {"texts": ["...", "..."]}
    GET  /metrics     Prometheus metrics

Run locally (development server only):
    python -m src.serving.app
Production (Docker): gunicorn "src.serving.app:create_app()"
Review text is never logged; only request id, status, latency and lengths are.
"""
import logging
import os
import time
import uuid

from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from werkzeug.exceptions import HTTPException

from src.features.text_cleaning import clean_text
from src.serving.model_loader import ModelBundle, load_model_bundle

logger = logging.getLogger("sentiment_api")

LABELS = {0: "negative", 1: "positive"}  # must match label_map in configs/pipeline_config.yaml
MAX_TEXT_CHARS = 10_000
MAX_BATCH = 32
MAX_BODY_BYTES = 1_000_000
MODEL_NAME = "sentiment-classifier"


class ValidationError(Exception):
    """The request body is not acceptable (maps to HTTP 400)."""


def parse_texts(payload) -> list[str]:
    if not isinstance(payload, dict):
        raise ValidationError("JSON body must be an object")
    if "text" in payload and "texts" in payload:
        raise ValidationError("Provide either 'text' or 'texts', not both")
    if "text" in payload:
        texts = [payload["text"]]
    elif "texts" in payload:
        texts = payload["texts"]
        if not isinstance(texts, list):
            raise ValidationError("'texts' must be a list of strings")
    else:
        raise ValidationError("Missing 'text' (string) or 'texts' (list of strings)")
    if not texts:
        raise ValidationError("'texts' must not be empty")
    if len(texts) > MAX_BATCH:
        raise ValidationError(f"At most {MAX_BATCH} texts per request")
    for item in texts:
        if not isinstance(item, str):
            raise ValidationError("Every text must be a string")
        if len(item) > MAX_TEXT_CHARS:
            raise ValidationError(f"Each text must be at most {MAX_TEXT_CHARS} characters")
        if not clean_text(item):
            raise ValidationError("Text must not be empty")
    return texts


def create_app(bundle: ModelBundle | None = None, load: bool = True) -> Flask:
    """App factory. Pass `bundle` (and load=False) in tests; otherwise the model is loaded at start."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

    if bundle is None and load:
        try:
            bundle = load_model_bundle()
            logger.info("Loaded model version=%s source=%s", bundle.version, bundle.source)
        except Exception:
            logger.exception("Could not load a model; /predict will return 503")
    holder = {"bundle": bundle}

    registry = CollectorRegistry()  # one per app, so tests can create many apps
    requests_total = Counter("api_requests_total", "API requests", ["endpoint", "method", "status"], registry=registry)
    latency = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)
    predictions_total = Counter("predictions_total", "Predictions made", ["label", "model_version"], registry=registry)
    invalid_total = Counter("input_validation_failures_total", "Rejected prediction requests", registry=registry)
    text_length = Histogram(
        "input_text_length_chars", "Length of submitted texts",
        buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000), registry=registry,
    )
    model_loaded = Gauge("model_loaded", "1 if a model is loaded", registry=registry)
    model_loaded.set(1 if bundle else 0)

    @app.before_request
    def start_timer():
        g.start = time.perf_counter()
        g.request_id = uuid.uuid4().hex

    @app.after_request
    def record(response):
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        elapsed = time.perf_counter() - g.get("start", time.perf_counter())
        latency.labels(endpoint).observe(elapsed)
        requests_total.labels(endpoint, request.method, str(response.status_code)).inc()
        response.headers["X-Request-ID"] = g.get("request_id", "")
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            g.get("request_id"), request.method, request.path, response.status_code, elapsed * 1000,
        )
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("Unhandled error")  # traceback only; the request body is never logged
        return jsonify(error="Internal server error"), 500

    @app.get("/health")
    def health():
        current = holder["bundle"]
        if current is None:
            return jsonify(status="unavailable", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True, model_version=current.version)

    @app.get("/model-info")
    def model_info():
        current = holder["bundle"]
        if current is None:
            return jsonify(error="No model loaded"), 503
        return jsonify(
            model_name=MODEL_NAME,
            model_version=current.version,
            source=current.source,
            labels={str(k): v for k, v in LABELS.items()},
            loaded_at=current.loaded_at,
            metadata=current.metadata,
        )

    @app.post("/predict")
    def predict():
        current = holder["bundle"]
        if current is None:
            return jsonify(error="No model loaded"), 503
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json"), 415
        payload = request.get_json(silent=True)
        if payload is None:
            invalid_total.inc()
            return jsonify(error="Request body must be valid JSON"), 400
        try:
            texts = parse_texts(payload)
        except ValidationError as exc:
            invalid_total.inc()
            return jsonify(error=str(exc)), 400

        for item in texts:
            text_length.observe(len(item))
        cleaned = [clean_text(item) for item in texts]  # same cleaning as training
        model = current.model
        class_ids = model.predict(cleaned)
        if hasattr(model, "predict_proba"):
            confidences = model.predict_proba(cleaned).max(axis=1)
        else:
            confidences = [None] * len(cleaned)

        results = []
        for class_id, confidence in zip(class_ids, confidences):
            label = LABELS[int(class_id)]
            predictions_total.labels(label, current.version).inc()
            results.append({
                "label": label,
                "class_id": int(class_id),
                "confidence": None if confidence is None else round(float(confidence), 4),
            })
        return jsonify(model_version=current.version, predictions=results)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    # Development server only; production uses gunicorn (see the Docker step).
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
