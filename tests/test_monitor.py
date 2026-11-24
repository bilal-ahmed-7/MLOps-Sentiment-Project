"""Tests for drift detection, summaries and labeled performance."""
import numpy as np
import pandas as pd
import pytest

from src.features.text_features import FEATURE_NAMES, extract_features
from src.monitoring import monitor
from src.monitoring.monitor import (
    detect_drift, labeled_performance, load_predictions, prediction_summary,
)
from src.monitoring.simulate_traffic import drifted_texts

RNG = np.random.default_rng(0)


def frame(n, scale=1.0):
    return pd.DataFrame({name: RNG.normal(100 * scale, 10, n) for name in FEATURE_NAMES})


def test_no_drift_when_distributions_match():
    report = detect_drift(frame(2000), frame(300))
    assert report["status"] == "ok" and report["dataset_drift"] is False
    assert report["drift_share"] <= 0.3


def test_drift_when_distribution_shifts():
    report = detect_drift(frame(2000), frame(300, scale=3.0))
    assert report["dataset_drift"] is True and report["drift_share"] == 1.0
    assert all(item["drifted"] for item in report["features"].values())


def test_not_enough_rows_gives_no_verdict():
    report = detect_drift(frame(2000), frame(5), min_rows=30)
    assert report == {"status": "insufficient_data", "current_rows": 5, "min_rows": 30}


def test_thresholds_are_respected():
    shifted = frame(300, scale=3.0)
    assert detect_drift(frame(2000), shifted, share_threshold=1.0)["dataset_drift"] is False


def test_simulated_drift_texts_really_look_different_from_reviews():
    reviews = pd.DataFrame([extract_features("This was a long and thoughtful review of a film. " * 20)] * 100)
    tweets = pd.DataFrame([extract_features(t) for t in drifted_texts(100)])
    assert detect_drift(reviews.sample(100), tweets)["features"]["char_length"]["drifted"]


def test_load_predictions_handles_missing_and_empty_files(tmp_path):
    assert load_predictions(str(tmp_path / "missing.jsonl")).empty
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert load_predictions(str(empty)).empty


def test_prediction_summary():
    preds = pd.DataFrame({"label": ["positive", "negative", "positive"], "confidence": [0.9, 0.8, 0.7],
                          "model_version": [1, 1, 2]})
    summary = prediction_summary(preds)
    assert summary["positive_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert summary["model_versions"] == ["1", "2"]


def test_labeled_performance_with_ground_truth():
    preds = pd.DataFrame({
        "prediction_id": ["a", "b", "c", "d"],
        "label": ["positive", "positive", "negative", "negative"],
    })
    truth = pd.DataFrame({"prediction_id": ["a", "b", "c"], "label": ["positive", "negative", "negative"]})
    perf = labeled_performance(preds, truth)
    assert perf["labeled_rows"] == 3 and perf["accuracy"] == pytest.approx(0.6667, abs=1e-3)


def test_labeled_performance_without_matches():
    preds = pd.DataFrame({"prediction_id": ["a"], "label": ["positive"]})
    truth = pd.DataFrame({"prediction_id": ["zzz"], "label": ["positive"]})
    assert labeled_performance(preds, truth) == {"labeled_rows": 0}


def test_push_failure_does_not_crash(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("gateway down")

    monkeypatch.setattr(monitor, "push_to_gateway", boom)
    assert monitor.push_metrics({"status": "insufficient_data", "current_rows": 3}, "http://127.0.0.1:1") is False
