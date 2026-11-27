"""Evaluate a pretrained Hugging Face sentiment model and compare it with the TF-IDF baseline.

Needs the extra packages:  pip install transformers torch
Run from the project root:
    python -m src.models.hf_eval

Both models are scored on the SAME rows of the validation and test sets, so the comparison is
fair. The result is logged as an MLflow run in the same experiment as the baseline, and a short
summary is written to artifacts/hf_comparison.json.
"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import mlflow
import yaml

from src.data.ingestion import load_config
from src.models.evaluate import compute_metrics, passes_quality_gate
from src.models.hf_model import HFSentimentClassifier
from src.models.train import get_data_version, get_git_info, load_params, load_splits, setup_tracking

logger = logging.getLogger(__name__)
HF_PARAMS = Path("configs/hf_params.yaml")
BASELINE_MODEL = Path("artifacts/model.joblib")
REPORT_FILE = Path("artifacts/hf_comparison.json")


def load_hf_params(path: Path = HF_PARAMS) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["huggingface"]


def subsample(frame, max_rows, seed: int = 42):
    if not max_rows or max_rows >= len(frame):
        return frame
    return frame.sample(n=max_rows, random_state=seed)


def score(model, frame) -> tuple[dict, float]:
    """Metrics and the seconds spent predicting."""
    start = time.perf_counter()
    predictions = model.predict(frame["text"].tolist())
    return compute_metrics(frame["label"], predictions), time.perf_counter() - start


def run_hf_evaluation(cfg: dict, params: dict, hf: dict, hf_model, baseline=None,
                      run_name: str = "hf-pretrained-eval") -> dict:
    """Score the Hugging Face model (and the baseline) on the same rows; log everything to MLflow."""
    _, val, test = load_splits(cfg)
    val, test = subsample(val, hf.get("eval_max_rows")), subsample(test, hf.get("eval_max_rows"))
    n_rows = len(val) + len(test)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**get_git_info(), "dvc_data_md5": get_data_version(cfg), "model_family": "huggingface"})
        mlflow.log_params({
            "model_type": "huggingface_pretrained",
            "hf_model_name": hf["model_name"],
            "hf_revision": str(hf.get("revision")),
            "hf_max_length": hf["max_length"],
            "hf_batch_size": hf["batch_size"],
            "n_eval_val": len(val),
            "n_eval_test": len(test),
        })
        hf_val, hf_val_seconds = score(hf_model, val)
        hf_test, hf_test_seconds = score(hf_model, test)
        ms_per_review = round(1000 * (hf_val_seconds + hf_test_seconds) / n_rows, 2)
        metrics = {
            **{f"val_{k}": v for k, v in hf_val.items()},
            **{f"test_{k}": v for k, v in hf_test.items()},
            "inference_ms_per_review": ms_per_review,
        }
        report = {
            "hf_model": hf["model_name"], "n_val": len(val), "n_test": len(test),
            "hf": {**hf_val, "test_f1": hf_test["f1"], "ms_per_review": ms_per_review},
        }
        if baseline is not None:
            base_val, base_val_seconds = score(baseline, val)
            base_test, base_test_seconds = score(baseline, test)
            base_ms = round(1000 * (base_val_seconds + base_test_seconds) / n_rows, 2)
            metrics.update({"baseline_val_f1": base_val["f1"], "baseline_test_f1": base_test["f1"],
                            "baseline_ms_per_review": base_ms})
            report["baseline"] = {**base_val, "test_f1": base_test["f1"], "ms_per_review": base_ms}
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "hf_comparison.json")

    report["mlflow_run_id"] = run.info.run_id
    report["passed_quality_gate"] = passes_quality_gate({"f1": hf_val["f1"]}, params["quality_gate"]["min_f1"])
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        hf = load_hf_params()
        model = HFSentimentClassifier(hf["model_name"], hf.get("revision"), hf["max_length"], hf["batch_size"])
    except ImportError:
        logger.error("transformers/torch are not installed. Run: pip install transformers torch")
        return 1

    cfg, params = load_config(), load_params()
    setup_tracking(params)
    baseline = joblib.load(BASELINE_MODEL) if BASELINE_MODEL.exists() else None
    if baseline is None:
        logger.warning("No baseline model at %s; scoring the Hugging Face model only.", BASELINE_MODEL)

    report = run_hf_evaluation(cfg, params, hf, model, baseline)
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nScored {report['n_val']} validation and {report['n_test']} test reviews (same rows for both models)")
    for name in ("baseline", "hf"):
        if name in report:
            r = report[name]
            label = "TF-IDF baseline" if name == "baseline" else report["hf_model"]
            print(f"  {label:48s} val F1 {r['f1']:.4f} | test F1 {r['test_f1']:.4f} | {r['ms_per_review']} ms/review")
    print(f"Quality gate (val F1 >= {params['quality_gate']['min_f1']}): "
          f"{'PASSED' if report['passed_quality_gate'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
