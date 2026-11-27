"""Compile the standalone Kubeflow pipeline used by local Airflow.

Run: python -m pipelines.kubeflow.pipeline_definition --output artifacts/sentiment_retraining.yaml
"""
import argparse
from pathlib import Path

from kfp import compiler, dsl, kubernetes

from pipelines.kubeflow.components.evaluate_component import evaluate_component
from pipelines.kubeflow.components.preprocess_component import preprocess_component
from pipelines.kubeflow.components.train_component import train_component

PIPELINE_NAME = "sentiment-retraining"
DVC_SECRET = "dagshub-credentials"


def add_dvc_secret(task):
    """Inject DVC credentials into the only task that pulls raw data."""
    kubernetes.use_secret_as_env(
        task,
        secret_name=DVC_SECRET,
        secret_key_to_env={"DAGSHUB_USER": "DAGSHUB_USER", "DAGSHUB_TOKEN": "DAGSHUB_TOKEN"},
    )
    return task


@dsl.pipeline(name=PIPELINE_NAME, description="Reproducible train/evaluate pipeline with a gated decision artifact.")
def sentiment_retraining_pipeline(
    model_family: str = "baseline",
    champion_val_f1: float = 0.9034,
    min_f1: float = 0.85,
    mlflow_tracking_uri: str = "http://host.docker.internal:5000",
):
    preprocess = add_dvc_secret(preprocess_component())
    train = train_component(splits=preprocess.outputs["splits"], model_family=model_family)
    train.set_env_variable("MLFLOW_TRACKING_URI", mlflow_tracking_uri)
    train.set_caching_options(enable_caching=False)
    evaluate_component(
        candidate=train.outputs["model_report"],
        champion_val_f1=champion_val_f1,
        min_f1=min_f1,
    ).set_caching_options(enable_caching=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the Kubeflow retraining pipeline.")
    parser.add_argument("--output", default="artifacts/sentiment_retraining.yaml")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(sentiment_retraining_pipeline, str(output))
    print(f"Compiled {PIPELINE_NAME} to {output}")


if __name__ == "__main__":
    main()
