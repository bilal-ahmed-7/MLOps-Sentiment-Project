"""Load the model to serve: MLflow registry `champion` first, saved file as fallback.

Environment variables:
    MODEL_SOURCE         auto (default) | registry | file
    MODEL_PATH           path of the fallback model file (default artifacts/model.joblib)
    MLFLOW_TRACKING_URI  where the registry lives (default: local sqlite:///mlflow.db)
    MODEL_NAME / MODEL_ALIAS  registry model name and alias (defaults: sentiment-classifier / champion)

In "auto" mode the registry is tried only when MLFLOW_TRACKING_URI is set or a local
mlflow.db exists; if loading from it fails, the saved file is used instead.
Only load model files you produced yourself: joblib/pickle files can run code on load.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = "artifacts/model.joblib"
DEFAULT_MODEL_NAME = "sentiment-classifier"
DEFAULT_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass
class ModelBundle:
    model: Any
    version: str
    source: str  # "registry" or "file"
    metadata: dict = field(default_factory=dict)
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_from_file(path: str = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load a joblib model (trusted files only) plus val_f1/run id from metrics.json next to it."""
    model = joblib.load(path)
    metadata = {}
    metrics_file = Path(path).parent / "metrics.json"
    if metrics_file.exists():
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))
        metadata = {"val_f1": saved.get("val_f1"), "source_run_id": saved.get("mlflow_run_id")}
    return ModelBundle(model=model, version="local-file", source="file", metadata=metadata)


def load_from_registry(
    name: str = DEFAULT_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    tracking_uri: str | None = None,
) -> ModelBundle:
    """Load the model that holds `alias` in the MLflow Model Registry."""
    import mlflow  # imported here so the file-only path does not need mlflow installed
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    metadata = {
        "val_f1": version.tags.get("val_f1"),
        "source_run_id": version.tags.get("source_run_id"),
        "alias": alias,
    }
    return ModelBundle(model=model, version=str(version.version), source="registry", metadata=metadata)


def load_model_bundle() -> ModelBundle:
    """Pick the model source from MODEL_SOURCE (auto/registry/file) and return the loaded bundle."""
    source = os.getenv("MODEL_SOURCE", "auto").lower()
    path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = os.getenv("MODEL_ALIAS", DEFAULT_ALIAS)

    if source == "file":
        return load_from_file(path)
    if source == "registry":
        return load_from_registry(name, alias)
    if source != "auto":
        raise ValueError(f"MODEL_SOURCE must be auto, registry or file (got {source!r})")

    if os.getenv("MLFLOW_TRACKING_URI") or Path("mlflow.db").exists():
        try:
            return load_from_registry(name, alias)
        except Exception as exc:  # registry unreachable, alias missing, etc.
            logger.warning("Registry load failed (%s); falling back to %s", type(exc).__name__, path)
    return load_from_file(path)
