"""Validates the monitoring configuration files and their consistency with the API."""
import json
from pathlib import Path

import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.serving.app import create_app
from src.serving.model_loader import ModelBundle

ROOT = Path(__file__).resolve().parents[1]
load_yaml = lambda rel: yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))  # noqa: E731
EXPECTED_ALERTS = {
    "HighErrorRate", "HighLatencyP95", "ApiDown", "ModelNotLoaded",
    "HighValidationFailureRate", "InputDataDrift",
}
API_METRICS = [
    "api_requests_total", "api_request_latency_seconds_bucket", "predictions_total",
    "input_validation_failures_total", "input_text_length_chars_sum", "model_loaded",
]


def test_prometheus_config_scrapes_the_api_and_loads_the_rules():
    cfg = load_yaml("monitoring/prometheus/prometheus.yml")
    jobs = {job["job_name"]: job for job in cfg["scrape_configs"]}
    assert jobs["sentiment-api"]["static_configs"][0]["targets"] == ["api:8000"]
    assert "pushgateway" in jobs and jobs["pushgateway"]["honor_labels"] is True
    assert cfg["rule_files"] == ["/etc/prometheus/alert_rules.yml"]


def test_alert_rules_are_complete():
    groups = load_yaml("monitoring/prometheus/alert_rules.yml")["groups"]
    rules = [rule for group in groups for rule in group["rules"]]
    assert {r["alert"] for r in rules} == EXPECTED_ALERTS
    for rule in rules:
        assert rule["expr"].strip() and rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"] and rule["annotations"]["description"]


def test_grafana_dashboard_is_valid_and_uses_the_provisioned_datasource():
    dashboard = json.loads((ROOT / "monitoring/grafana/dashboards/sentiment-api.json").read_text())
    uid = load_yaml("monitoring/grafana/provisioning/datasources/prometheus.yml")["datasources"][0]["uid"]
    assert dashboard["panels"] and len({p["id"] for p in dashboard["panels"]}) == len(dashboard["panels"])
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == uid and set(panel["gridPos"]) == {"x", "y", "w", "h"}
        assert all(t["expr"] and t["datasource"]["uid"] == uid for t in panel["targets"])


def test_compose_exposes_ports_on_localhost_only_and_takes_secrets_from_env():
    services = load_yaml("docker-compose.yml")["services"]
    assert {"api", "prometheus", "pushgateway", "grafana"} <= set(services)
    for name, service in services.items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on all interfaces"
    assert "${GRAFANA_ADMIN_PASSWORD" in services["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"]


@pytest.fixture(scope="module")
def metrics_text():
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(["good great", "bad awful"] * 5, [1, 0] * 5)
    client = create_app(bundle=ModelBundle(model=pipe, version="1", source="test"), load=False).test_client()
    client.post("/predict", json={"text": "good great"})
    client.post("/predict", json={"text": ""})
    return client.get("/metrics").get_data(as_text=True)


@pytest.mark.parametrize("metric", API_METRICS)
def test_metrics_used_by_rules_and_dashboard_are_exported_by_the_api(metric, metrics_text):
    assert metric in metrics_text
    used_in = (ROOT / "monitoring/prometheus/alert_rules.yml").read_text() + (
        ROOT / "monitoring/grafana/dashboards/sentiment-api.json"
    ).read_text()
    assert metric.replace("_bucket", "").replace("_sum", "") in used_in
