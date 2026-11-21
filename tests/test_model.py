"""Model tests: metrics, quality gate, output shape, valid predictions, regression."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.models.evaluate import compute_metrics, passes_quality_gate
from src.models.train import build_model, load_params

POS = ["great wonderful loved it", "amazing brilliant acting", "fantastic and moving film", "loved every minute"] * 10
NEG = ["awful boring waste", "terrible and dull plot", "hated this horrible movie", "worst film ever"] * 10
TEXTS = POS + NEG
LABELS = [1] * len(POS) + [0] * len(NEG)


def test_compute_metrics_perfect_and_worst():
    assert compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])["f1"] == 1.0
    assert compute_metrics([1, 1, 0, 0], [0, 0, 1, 1])["accuracy"] == 0.0


def test_quality_gate_threshold():
    assert passes_quality_gate({"f1": 0.9}, 0.85)
    assert passes_quality_gate({"f1": 0.85}, 0.85)
    assert not passes_quality_gate({"f1": 0.7}, 0.85)


def test_model_output_shape_and_valid_labels():
    model = build_model(load_params()).fit(TEXTS, LABELS)
    preds = model.predict(TEXTS[:7])
    assert preds.shape == (7,)
    assert set(preds) <= {0, 1}


def test_model_probabilities_are_valid():
    model = build_model(load_params()).fit(TEXTS, LABELS)
    proba = model.predict_proba(["great film", "awful film"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_learns_obvious_signal():
    model = build_model({**load_params(), "tfidf": {"max_features": 1000, "ngram_range": [1, 1], "min_df": 1}})
    model.fit(TEXTS, LABELS)
    assert compute_metrics(LABELS, model.predict(TEXTS))["f1"] > 0.95


@pytest.mark.skipif(not Path("artifacts/metrics.json").exists(), reason="no trained model yet (run `dvc repro`)")
def test_regression_trained_model_meets_threshold():
    """Detects deterioration in model quality after retraining."""
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    assert metrics["val_f1"] >= load_params()["quality_gate"]["min_f1"]
