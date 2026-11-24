"""Data-drift and performance monitoring.

Compares numeric text features of recent predictions (logs/predictions.jsonl) with a
reference sample of the training data, using a two-sample Kolmogorov-Smirnov test per feature.
The dataset counts as drifted when the share of drifted features exceeds a threshold.

Run from the project root:
    python -m src.monitoring.monitor
    python -m src.monitoring.monitor --pushgateway http://localhost:9091 --fail-on-drift
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score

from src.features.text_features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)
LABEL_TO_ID = {"negative": 0, "positive": 1}

P_THRESHOLD = 0.05      # a feature drifts when the KS-test p-value is below this
SHARE_THRESHOLD = 0.30  # dataset drift when MORE than this share of features drifted
MIN_ROWS = 30           # too few recent predictions to say anything


def load_reference(processed_dir: str = "data/processed", n: int = 5000, seed: int = 42) -> pd.DataFrame:
    train = pd.read_csv(Path(processed_dir) / "train.csv")
    sample = train.sample(n=min(n, len(train)), random_state=seed)
    return pd.DataFrame([extract_features(text) for text in sample["text"]])


def load_predictions(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists() or file.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(file, lines=True)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features=FEATURE_NAMES,
    p_threshold: float = P_THRESHOLD,
    share_threshold: float = SHARE_THRESHOLD,
    min_rows: int = MIN_ROWS,
) -> dict:
    if len(current) < min_rows:
        return {"status": "insufficient_data", "current_rows": len(current), "min_rows": min_rows}
    per_feature = {}
    for name in features:
        result = ks_2samp(reference[name], current[name])
        per_feature[name] = {
            "ks_statistic": round(float(result.statistic), 4),
            "p_value": float(result.pvalue),
            "drifted": bool(result.pvalue < p_threshold),
        }
    share = sum(item["drifted"] for item in per_feature.values()) / len(per_feature)
    return {
        "status": "ok",
        "current_rows": int(len(current)),
        "reference_rows": int(len(reference)),
        "drift_share": round(share, 3),
        "dataset_drift": bool(share > share_threshold),
        "features": per_feature,
    }


def prediction_summary(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {}
    return {
        "positive_rate": round(float((predictions["label"] == "positive").mean()), 4),
        "mean_confidence": round(float(predictions["confidence"].mean()), 4),
        "model_versions": sorted(predictions["model_version"].astype(str).unique().tolist()),
    }


def labeled_performance(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Accuracy/F1 on predictions whose ground truth is known (labels need prediction_id, label)."""
    merged = predictions.merge(labels, on="prediction_id", suffixes=("_pred", "_true"))
    if merged.empty:
        return {"labeled_rows": 0}

    def to_id(series):
        return series.map(lambda v: LABEL_TO_ID.get(str(v).lower(), v)).astype(int)

    y_true, y_pred = to_id(merged["label_true"]), to_id(merged["label_pred"])
    return {
        "labeled_rows": int(len(merged)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def push_metrics(report: dict, url: str) -> bool:
    """Publish drift results to a Prometheus Pushgateway so Grafana and alert rules can use them."""
    registry = CollectorRegistry()
    Gauge("drift_monitor_last_run_timestamp_seconds", "Last drift check", registry=registry).set(time.time())
    Gauge("drift_monitor_current_rows", "Predictions compared", registry=registry).set(report.get("current_rows", 0))
    if report.get("status") == "ok":
        Gauge("input_drift_share", "Share of drifted features", registry=registry).set(report["drift_share"])
        Gauge("input_drift_detected", "1 if dataset drift detected", registry=registry).set(
            1 if report["dataset_drift"] else 0
        )
        ks = Gauge("input_drift_ks_statistic", "KS statistic per feature", ["feature"], registry=registry)
        for name, item in report["features"].items():
            ks.labels(name).set(item["ks_statistic"])
    perf = report.get("performance", {})
    if "f1" in perf:
        Gauge("model_labeled_f1", "F1 on labeled predictions", registry=registry).set(perf["f1"])
        Gauge("model_labeled_accuracy", "Accuracy on labeled predictions", registry=registry).set(perf["accuracy"])
    try:
        push_to_gateway(url, job="drift_monitor", registry=registry)
        return True
    except Exception as exc:  # network errors must not crash the monitor
        logger.error("Could not push metrics to %s (%s)", url, type(exc).__name__)
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", default="logs/predictions.jsonl")
    parser.add_argument("--labels", help="CSV with columns prediction_id,label (ground truth)")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="artifacts/drift_report.json")
    parser.add_argument("--pushgateway", default=os.getenv("PUSHGATEWAY_URL"))
    parser.add_argument("--p-threshold", type=float, default=P_THRESHOLD)
    parser.add_argument("--share-threshold", type=float, default=SHARE_THRESHOLD)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--fail-on-drift", action="store_true", help="exit with code 2 when drift is detected")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    reference = load_reference(args.processed_dir)
    report = detect_drift(
        reference, predictions, p_threshold=args.p_threshold,
        share_threshold=args.share_threshold, min_rows=args.min_rows,
    )
    report["predictions"] = prediction_summary(predictions)
    if args.labels and not predictions.empty:
        report["performance"] = labeled_performance(predictions, pd.read_csv(args.labels))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["status"] == "insufficient_data":
        logger.info("Only %d predictions logged (need %d); no drift verdict.", report["current_rows"], args.min_rows)
    else:
        for name, item in report["features"].items():
            logger.info("%-18s KS=%.3f p=%.3g %s", name, item["ks_statistic"], item["p_value"],
                        "DRIFT" if item["drifted"] else "ok")
        logger.info("Drift share %.2f -> dataset drift: %s", report["drift_share"], report["dataset_drift"])
    if args.pushgateway:
        logger.info("Pushed to Pushgateway: %s", push_metrics(report, args.pushgateway))
    return 2 if args.fail_on_drift and report.get("dataset_drift") else 0


if __name__ == "__main__":
    sys.exit(main())
