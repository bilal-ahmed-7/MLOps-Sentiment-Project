# Codex change log

This file records changes made by Codex so the project hand-off remains auditable.

## 2026-11-27 — Hugging Face fine-tuning challenger

- Added `src/models/hf_finetune.py`: GPU-friendly DistilBERT fine-tuning on the existing DVC-managed train/validation/test splits, MLflow logging, fair baseline comparison, saved Hugging Face artifact, and a machine-readable promotion recommendation.
- Extended `configs/hf_params.yaml` with a reproducible `finetuning` configuration section.
- Added `datasets` and `accelerate` to `requirements.txt`, required by the Hugging Face Trainer workflow.
- Added `tests/test_hf_finetune.py` for configuration and deterministic data-subset helpers; these tests do not require a GPU or download a model.
- Updated `.gitignore` so generated model and checkpoint directories are not committed.
- Documented Colab/Kaggle GPU setup and commands in `README.md`.

### Operational decision

The fine-tuned transformer is evaluated and logged, but it does not automatically replace the Flask champion. The Flask loader currently supports sklearn MLflow models only. The report's `eligible_for_bentoml_promotion` flag is the safe hand-off to the upcoming BentoML serving stage.

## 2026-11-27 — Free local Airflow and standalone Kubeflow Pipelines

- Replaced the KFP component and pipeline placeholders with a three-stage `preprocess → train → evaluate` workflow. It passes DVC-produced splits as a KFP artifact, logs candidate metrics to MLflow, and writes a gated recommendation without changing production serving.
- Added a local KFP task image (`docker/Dockerfile.ml`), an isolated Airflow image (`docker/Dockerfile.airflow`), KFP orchestration requirements, and `docker-compose.airflow.yml`.
- Added a weekly, drift-gated Airflow DAG that compiles, submits, and waits for the KFP baseline retraining pipeline. No-drift runs skip submission; no DAG path deploys a model.
- Added local-installation instructions, Kubernetes Secret setup, port-forwarding, and safe stop commands to `README.md`.
- Updated Docker ignore rules so the pipeline image receives DVC metadata and the raw-data pointer but never the dataset, DVC local credentials, or model artifacts.
- Pinned the KFP SDK and Kubernetes extension to `2.18.2`, matching the standalone backend manifest, rather than mixing independently-versioned packages.
- Moved standard-library imports inside KFP component functions so compiled lightweight Python components run correctly from the custom pipeline image.
