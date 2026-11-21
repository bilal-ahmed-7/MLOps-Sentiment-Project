"""Tests for gated promotion and rollback in the MLflow Model Registry."""
import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.models.register import (
    CHAMPION, MODEL_NAME, PREVIOUS, get_by_alias, register_and_gate, rollback, should_promote,
)


def test_should_promote_rules():
    assert should_promote(0.90, None, 0.85)        # first model that passes the gate
    assert should_promote(0.92, 0.90, 0.85)        # better than champion
    assert not should_promote(0.90, 0.90, 0.85)    # equal is not better
    assert not should_promote(0.88, 0.90, 0.85)    # worse
    assert not should_promote(0.80, None, 0.85)    # below the gate


@pytest.fixture
def client(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp_id = mlflow.create_experiment("registry-test", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment(experiment_id=exp_id)
    return MlflowClient()


def _logged_run() -> str:
    model = LogisticRegression().fit(np.array([[0.0], [1.0], [0.0], [1.0]]), [0, 1, 0, 1])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model")
    return run.info.run_id


def test_promotion_rejection_and_rollback_lifecycle(client):
    r1 = register_and_gate(_logged_run(), 0.90, 0.85, client)
    assert r1["decision"] == "promoted"
    assert get_by_alias(client, CHAMPION).version == r1["version"]

    r2 = register_and_gate(_logged_run(), 0.88, 0.85, client)  # worse than champion
    assert r2["decision"] == "rejected_not_better"
    assert get_by_alias(client, CHAMPION).version == r1["version"]

    r3 = register_and_gate(_logged_run(), 0.93, 0.85, client)  # better
    assert r3["decision"] == "promoted"
    assert get_by_alias(client, CHAMPION).version == r3["version"]
    assert get_by_alias(client, PREVIOUS).version == r1["version"]

    assert rollback(client)["champion_version"] == r1["version"]
    assert get_by_alias(client, CHAMPION).version == r1["version"]


def test_model_below_quality_gate_is_not_registered(client):
    result = register_and_gate(_logged_run(), 0.70, 0.85, client)
    assert result["decision"] == "rejected_quality_gate"
    assert client.search_model_versions(f"name='{MODEL_NAME}'") == []


def test_rollback_without_previous_raises(client):
    register_and_gate(_logged_run(), 0.90, 0.85, client)
    with pytest.raises(RuntimeError):
        rollback(client)


def test_registering_same_run_twice_does_not_duplicate(client):
    run_id = _logged_run()
    first = register_and_gate(run_id, 0.90, 0.85, client)
    again = register_and_gate(run_id, 0.90, 0.85, client)
    assert again["decision"] == "already_registered"
    assert again["version"] == first["version"]
    assert len(client.search_model_versions(f"name='{MODEL_NAME}'")) == 1
