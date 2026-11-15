"""TODO: implement src/models/train.py"""
"""Train the TF-IDF + Logistic Regression model and track it with MLflow.

Run from the project root:
    python -m src.models.train

Tracking location comes from the MLFLOW_TRACKING_URI environment variable
(e.g. your DagsHub MLflow URL). If it is not set, a local SQLite file is used.
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.ingestion import load_config
from src.models.evaluate import compute_metrics, confusion_matrix_figure, passes_quality_gate

logger = logging.getLogger(__name__)
DEFAULT_PARAMS = Path("configs/model_params.yaml")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def load_params(path: Path = DEFAULT_PARAMS) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    return tuple(pd.read_csv(d / f"{name}.csv") for name in ("train", "val", "test"))


def build_model(params: dict) -> Pipeline:
    t, lr = params["tfidf"], params["logreg"]
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                max_features=t["max_features"],
                ngram_range=tuple(t["ngram_range"]),
                min_df=t["min_df"],
                lowercase=True,
            )),
            ("clf", LogisticRegression(C=lr["C"], max_iter=lr["max_iter"])),
        ]
    )


def _run_git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_git_info() -> dict:
    dirty = _run_git("status", "--porcelain") not in ("", "unknown")
    return {"git_commit": _run_git("rev-parse", "HEAD"), "git_dirty": str(dirty)}


def get_data_version(cfg: dict) -> str:
    """MD5 of the raw dataset, read from its DVC pointer file."""
    pointer = Path(cfg["data"]["raw_path"] + ".dvc")
    try:
        return yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
    except (OSError, KeyError, IndexError, TypeError):
        return "unknown"


def setup_tracking(params: dict) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(params["experiment_name"])


def flatten_params(params: dict) -> dict:
    flat = {"model_type": params["model_type"]}
    for section in ("tfidf", "logreg"):
        for key, value in params[section].items():
            flat[f"{section}_{key}"] = str(value) if isinstance(value, list) else value
    return flat


def run_experiment(cfg: dict, params: dict, run_name: str | None = None):
    """Train once, log everything to MLflow. Returns (model, metrics, run_id)."""
    train, val, test = load_splits(cfg)
    model = build_model(params)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**get_git_info(), "dvc_data_md5": get_data_version(cfg)})
        mlflow.log_params(flatten_params(params))
        mlflow.log_param("n_train", len(train))

        start = time.time()
        model.fit(train["text"], train["label"])
        mlflow.log_metric("training_seconds", round(time.time() - start, 2))

        val_pred = model.predict(val["text"])
        test_pred = model.predict(test["text"])
        metrics = {
            **{f"val_{k}": v for k, v in compute_metrics(val["label"], val_pred).items()},
            **{f"test_{k}": v for k, v in compute_metrics(test["label"], test_pred).items()},
        }
        mlflow.log_metrics(metrics)
        mlflow.log_figure(confusion_matrix_figure(test["label"], test_pred), "confusion_matrix_test.png")
        mlflow.sklearn.log_model(model, name="model")
        logger.info("Run %s | %s", run.info.run_id, metrics)
        return model, metrics, run.info.run_id


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg, params = load_config(), load_params()
    setup_tracking(params)
    model, metrics, run_id = run_experiment(cfg, params, run_name="baseline-tfidf-logreg")

    gate_ok = passes_quality_gate(
        {"f1": metrics["val_f1"]}, params["quality_gate"]["min_f1"]
    )
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(
        json.dumps({**metrics, "mlflow_run_id": run_id, "passed_quality_gate": gate_ok}, indent=2),
        encoding="utf-8",
    )
    if not gate_ok:
        logger.error("Quality gate FAILED: val_f1=%s < %s", metrics["val_f1"], params["quality_gate"]["min_f1"])
        return 1  # a failing model must not proceed
    joblib.dump(model, "artifacts/model.joblib")
    logger.info("Quality gate passed. Model saved to artifacts/model.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())