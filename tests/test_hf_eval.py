"""Tests for the Hugging Face evaluation workflow (stub model, temporary MLflow store)."""
import mlflow
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.hf_eval import load_hf_params, run_hf_evaluation, subsample
from src.models.hf_model import HFSentimentClassifier
from src.models.train import load_params

POS = ["good great film", "good wonderful acting", "good moving story"] * 20
NEG = ["bad awful film", "bad boring plot", "bad terrible waste"] * 20


def stub_pipeline(texts, **kwargs):
    return [
        [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}] if "good" in t
        else [{"label": "NEGATIVE", "score": 0.9}, {"label": "POSITIVE", "score": 0.1}]
        for t in texts
    ]


@pytest.fixture
def setup(tmp_path):
    for name in ("train", "val", "test"):
        frame = pd.DataFrame({"text": POS + NEG, "label": [1] * len(POS) + [0] * len(NEG)})
        frame.insert(0, "review_id", range(len(frame)))
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp_id = mlflow.create_experiment("hf-test", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_id=exp_id)
    cfg = {"data": {"processed_dir": str(tmp_path), "raw_path": "data/raw/imdb_reviews.csv"}}
    baseline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    baseline.fit(POS + NEG, [1] * len(POS) + [0] * len(NEG))
    return cfg, baseline


def test_hf_params_file_has_required_keys():
    hf = load_hf_params()
    assert {"model_name", "max_length", "batch_size", "eval_max_rows"} <= set(hf)


def test_subsample_limits_rows_and_is_reproducible():
    frame = pd.DataFrame({"text": list("abcdefghij"), "label": [0, 1] * 5})
    assert len(subsample(frame, 4)) == 4
    assert subsample(frame, 4).equals(subsample(frame, 4))
    assert len(subsample(frame, None)) == 10 and len(subsample(frame, 100)) == 10


def test_evaluation_logs_a_run_with_both_models(setup):
    cfg, baseline = setup
    hf = {"model_name": "stub-model", "revision": None, "max_length": 64, "batch_size": 8, "eval_max_rows": 50}
    model = HFSentimentClassifier("stub-model", predictor=stub_pipeline)
    report = run_hf_evaluation(cfg, load_params(), hf, model, baseline)

    assert report["n_val"] == 50 and report["n_test"] == 50
    assert report["hf"]["f1"] == 1.0 and report["baseline"]["f1"] == 1.0
    assert report["passed_quality_gate"] is True
    run = mlflow.get_run(report["mlflow_run_id"])
    assert run.data.params["hf_model_name"] == "stub-model"
    assert {"val_f1", "test_f1", "baseline_val_f1", "inference_ms_per_review"} <= set(run.data.metrics)
    assert run.data.tags["model_family"] == "huggingface"


def test_evaluation_without_baseline(setup):
    cfg, _ = setup
    hf = {"model_name": "stub-model", "revision": None, "max_length": 64, "batch_size": 8, "eval_max_rows": None}
    model = HFSentimentClassifier("stub-model", predictor=stub_pipeline)
    report = run_hf_evaluation(cfg, load_params(), hf, model, baseline=None)
    assert "baseline" not in report and report["n_val"] == len(POS + NEG)


def test_failing_model_does_not_pass_the_gate(setup):
    cfg, _ = setup

    def always_wrong(texts, **kwargs):
        return [[{"label": "NEGATIVE", "score": 0.9}, {"label": "POSITIVE", "score": 0.1}] for _ in texts]

    hf = {"model_name": "bad", "revision": None, "max_length": 64, "batch_size": 8, "eval_max_rows": 30}
    report = run_hf_evaluation(cfg, load_params(), hf, HFSentimentClassifier("bad", predictor=always_wrong))
    assert report["passed_quality_gate"] is False
