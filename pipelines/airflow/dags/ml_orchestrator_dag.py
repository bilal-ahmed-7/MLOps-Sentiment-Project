"""Schedule drift-gated Kubeflow retraining in the free local development stack.

Airflow detects data drift and submits a compiled KFP package. KFP performs the
data/model work and produces its decision artifact. This DAG never deploys a
model: BentoML health checks are required before any serving promotion.
"""
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_DIR = Path("/opt/airflow/project")
PACKAGE = PROJECT_DIR / "artifacts/sentiment_retraining.yaml"
KFP_HOST = os.getenv("KFP_HOST", "http://host.docker.internal:8081")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")


@dag(
    dag_id="sentiment_retraining_orchestrator",
    description="Check drift and submit a gated KFP retraining run.",
    schedule="0 3 * * 1",  # Monday 03:00 local Docker host time
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "kubeflow", "retraining"],
)
def sentiment_retraining_orchestrator():
    @task.short_circuit
    def detect_drift() -> bool:
        """Continue only if the monitor has enough observations and detects drift."""
        subprocess.run(
            ["python", "-m", "src.monitoring.monitor", "--output", "artifacts/drift_report.json"],
            cwd=PROJECT_DIR,
            check=True,
        )
        import json

        report = json.loads((PROJECT_DIR / "artifacts/drift_report.json").read_text(encoding="utf-8"))
        return report.get("status") == "ok" and bool(report.get("dataset_drift"))

    @task
    def compile_pipeline() -> str:
        subprocess.run(
            ["python", "-m", "pipelines.kubeflow.pipeline_definition", "--output", str(PACKAGE)],
            cwd=PROJECT_DIR,
            check=True,
        )
        return str(PACKAGE)

    @task
    def submit_and_wait(package_path: str) -> str:
        import kfp

        client = kfp.Client(host=KFP_HOST)
        run = client.create_run_from_pipeline_package(
            package_path,
            arguments={"model_family": "baseline", "mlflow_tracking_uri": MLFLOW_TRACKING_URI},
            experiment_name="sentiment-retraining",
        )
        completed = client.wait_for_run_completion(run.run_id, timeout=60 * 60)
        if completed.state != "SUCCEEDED":
            raise RuntimeError(f"Kubeflow run {run.run_id} finished with state {completed.state}")
        return run.run_id

    drift = detect_drift()
    package = compile_pipeline()
    submitted = submit_and_wait(package)
    drift >> submitted


sentiment_retraining_orchestrator()
