"""Tests for the Hugging Face wrapper, using a stub in place of a real transformers pipeline."""
import numpy as np
import pytest

from src.models.hf_model import HFSentimentClassifier, label_to_id


def stub_pipeline(texts, **kwargs):
    """Mimics transformers' text-classification output with top_k=None (all labels, best first)."""
    assert kwargs["truncation"] is True and kwargs["max_length"] > 0
    out = []
    for text in texts:
        if "good" in text:
            out.append([{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}])
        else:
            out.append([{"label": "NEGATIVE", "score": 0.8}, {"label": "POSITIVE", "score": 0.2}])
    return out


@pytest.fixture
def model():
    return HFSentimentClassifier("stub", predictor=stub_pipeline)


@pytest.mark.parametrize("label, expected", [
    ("POSITIVE", 1), ("negative", 0), ("LABEL_1", 1), ("LABEL_0", 0), (" Pos ", 1), (1, 1), (0, 0),
])
def test_label_to_id(label, expected):
    assert label_to_id(label) == expected


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        label_to_id("NEUTRAL")


def test_predict_proba_shape_and_order(model):
    proba = model.predict_proba(["good film", "bad film"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] == 0.9 and proba[1, 0] == 0.8  # column 0 = negative, 1 = positive


def test_predict_returns_class_ids(model):
    assert model.predict(["good film", "bad film"]).tolist() == [1, 0]


def test_label_order_from_the_pipeline_does_not_matter():
    def reversed_order(texts, **kwargs):
        return [[{"label": "NEGATIVE", "score": 0.3}, {"label": "POSITIVE", "score": 0.7}] for _ in texts]

    assert HFSentimentClassifier("stub", predictor=reversed_order).predict(["x"]).tolist() == [1]
