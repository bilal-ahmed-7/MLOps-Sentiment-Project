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
