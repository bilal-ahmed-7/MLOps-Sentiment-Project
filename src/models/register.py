"""Model registry with a gated promotion workflow and rollback.

Aliases used in the MLflow Model Registry:
    champion  - the version currently approved for serving
    previous  - the version that was champion before the last promotion (rollback target)

Usage (from the project root, with the same MLFLOW_TRACKING_URI used for training):
    python -m src.models.register             # register the last trained model; promote only if it passes
    python -m src.models.register --rollback  # make `previous` the champion again
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.train import load_params, setup_tracking

logger = logging.getLogger(__name__)
MODEL_NAME = "sentiment-classifier"
CHAMPION, PREVIOUS = "champion", "previous"


def should_promote(candidate_f1: float, champion_f1: float | None, min_f1: float) -> bool:
    """Promote only if the candidate passes the gate AND beats the current champion."""
    if candidate_f1 < min_f1:
        return False
    return champion_f1 is None or candidate_f1 > champion_f1


def get_by_alias(client: MlflowClient, alias: str) -> ModelVersion | None:
    """Return the model version that holds `alias`, or None if the alias is not set."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def promote(client: MlflowClient, version: str) -> None:
    """Make `version` the champion; the old champion (if any) becomes `previous`."""
    current = get_by_alias(client, CHAMPION)
    if current is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, current.version)
        client.set_model_version_tag(MODEL_NAME, current.version, "status", "previous")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    client.set_model_version_tag(MODEL_NAME, version, "status", "champion")


def rollback(client: MlflowClient) -> dict:
    """Swap champion and previous. Raises if there is nothing to roll back to."""
    previous, current = get_by_alias(client, PREVIOUS), get_by_alias(client, CHAMPION)
    if previous is None:
        raise RuntimeError("No previous version to roll back to.")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous.version)
    client.set_model_version_tag(MODEL_NAME, previous.version, "status", "champion")
    if current is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, current.version)
        client.set_model_version_tag(MODEL_NAME, current.version, "status", "rolled_back")
    return {"decision": "rolled_back", "champion_version": previous.version}


def register_and_gate(run_id: str, val_f1: float, min_f1: float, client: MlflowClient | None = None) -> dict:
    """Register a trained run as a new model version and promote it only if eligible."""
    client = client or MlflowClient()
    if val_f1 < min_f1:  # failing models are not registered at all
        return {"decision": "rejected_quality_gate", "val_f1": val_f1, "min_f1": min_f1}

    for existing in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if existing.tags.get("source_run_id") == run_id:  # re-running must not create duplicates
            return {"decision": "already_registered", "version": existing.version, "val_f1": val_f1}

    champion = get_by_alias(client, CHAMPION)
    champion_f1 = float(champion.tags["val_f1"]) if champion and "val_f1" in champion.tags else None

    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "val_f1", str(val_f1))
    client.set_model_version_tag(MODEL_NAME, mv.version, "source_run_id", run_id)

    if should_promote(val_f1, champion_f1, min_f1):
        promote(client, mv.version)
        decision = "promoted"
    else:
        client.set_model_version_tag(MODEL_NAME, mv.version, "status", "rejected_not_better")
        decision = "rejected_not_better"
    return {
        "decision": decision,
        "version": mv.version,
        "val_f1": val_f1,
        "previous_champion_f1": champion_f1,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true", help="restore the previous champion")
    args = parser.parse_args()

    params = load_params()
    setup_tracking(params)
    client = MlflowClient()

    if args.rollback:
        try:
            result = rollback(client)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    else:
        metrics_file = Path("artifacts/metrics.json")
        if not metrics_file.exists():
            logger.error("artifacts/metrics.json not found. Run `dvc repro` first.")
            return 1
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        result = register_and_gate(
            metrics["mlflow_run_id"], metrics["val_f1"], params["quality_gate"]["min_f1"], client
        )

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/registry_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Registry result: %s", result)
    return 1 if result["decision"] == "rejected_quality_gate" else 0


if __name__ == "__main__":
    sys.exit(main())
