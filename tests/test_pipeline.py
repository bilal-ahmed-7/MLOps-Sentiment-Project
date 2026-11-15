"""TODO: implement tests/test_pipeline.py"""
"""Integration test: processed data -> training -> MLflow run with metrics logged."""
import mlflow
import pandas as pd

from src.models.train import load_params, run_experiment, setup_tracking

POS = ["great wonderful loved it", "amazing brilliant acting", "fantastic moving film"] * 20
NEG = ["awful boring waste", "terrible dull plot", "hated horrible movie"] * 20


def _write_split(path):
    df = pd.DataFrame({"text": POS + NEG, "label": [1] * len(POS) + [0] * len(NEG)})
    df.insert(0, "review_id", range(len(df)))
    df.to_csv(path, index=False)


def test_training_run_is_logged_to_mlflow(tmp_path, monkeypatch):
    for name in ("train", "val", "test"):
        _write_split(tmp_path / f"{name}.csv")
    cfg = {"data": {"processed_dir": str(tmp_path), "raw_path": "data/raw/imdb_reviews.csv"}}
    params = {**load_params(), "tfidf": {"max_features": 500, "ngram_range": [1, 1], "min_df": 1}}

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    setup_tracking(params)
    _, metrics, run_id = run_experiment(cfg, params, run_name="integration-test")

    run = mlflow.get_run(run_id)
    assert "val_f1" in run.data.metrics and "training_seconds" in run.data.metrics
    assert run.data.params["tfidf_max_features"] == "500"
    assert "git_commit" in run.data.tags and "dvc_data_md5" in run.data.tags
    assert metrics["val_f1"] > 0.9