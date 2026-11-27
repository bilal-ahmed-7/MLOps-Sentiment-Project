# MLOps Sentiment Analysis Project

Sentiment analysis and text classification (IMDB movie reviews) with an end-to-end MLOps workflow:
data validation, a reproducible DVC pipeline, MLflow tracking and a gated model registry, a
Flask prediction API with Prometheus metrics, Docker, and GitHub Actions CI.

- GitHub: `bilal-ahmed-7/MLOps-Sentiment-Project`
- DVC remote and MLflow on DagsHub: `bilal-ahmed-7/Mlops-sentiment-project`
  (the capitalisation differs between the two names; use each one exactly as written)

## Project status

| Area | Status |
| --- | --- |
| Data ingestion + validation (Pandera), preprocessing, stratified 70/15/15 split | Done |
| Training (TF-IDF + Logistic Regression), tuning (8-run grid), MLflow tracking | Done |
| MLflow Model Registry: quality gate, champion/previous aliases, promotion, rollback | Done |
| Flask API (`/health`, `/model-info`, `/predict`, `/metrics`) | Done |
| Docker image, Docker Compose (API + MLflow + PostgreSQL) | Done |
| GitHub Actions CI (lint, tests, security reports, image build and publish) | Done |
| Prometheus/Grafana dashboards and alerts, drift checks | Planned (placeholders in `monitoring/`, `src/monitoring/`) |
| Airflow DAG, Kubeflow pipeline, retraining loop | Planned (placeholders in `pipelines/`) |
| Kubernetes manifests, AWS-style deployment | Planned (placeholders in `k8s/`, `.github/workflows/deploy.yml`) |
| Feast feature store, BentoML service, Hugging Face (DistilBERT) model | Planned (placeholders in `feature_store/`, `src/serving/`) |
| Final report | Skeleton in `REPORT.md` |

Files marked "Placeholder" in their first line are intentionally empty for now.

## Repository layout

```text
configs/        pipeline_config.yaml (data, labels, split) and model_params.yaml (model, gate, tuning grid)
data/           raw/ and processed/ (managed by DVC, not committed to Git)
src/data/       ingestion and Pandera validation
src/features/   text cleaning (shared by training and serving) and preprocessing
src/models/     train, evaluate, tune, register (MLflow registry)
src/serving/    Flask API and model loader
tests/          pytest suite (data, preprocessing, model, MLflow, registry, API, loader, consistency)
docker/         Dockerfile.api, Dockerfile.mlflow (Dockerfile.ml is a placeholder)
artifacts/      metrics.json and tuning outputs (Git); model.joblib (DVC)
.github/        CI workflow and a manual deploy placeholder
```

## 1. Setup (Windows PowerShell, Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS activate with `source .venv/bin/activate`. `requirements.txt` includes PyTorch from the
CPU index; it is not needed for the tests or the API.
`requirements-ci.txt` is the smaller set used by CI and `requirements-api.txt` is what the API image installs.
scikit-learn, numpy, scipy and joblib are pinned to the same versions in all three files because the saved
model is a pickle (a test checks this).

## 2. Get the data and the trained model (DVC + DagsHub)

Store your DagsHub credentials locally. They go into `.dvc/config.local`, which is git-ignored.
Never commit it and never paste a token into a tracked file:

```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
dvc pull
```

`dvc pull` downloads the raw dataset (`data/raw/imdb_reviews.csv`) and the DVC-tracked outputs, including
`artifacts/model.joblib`.

## 3. Reproduce the pipeline

```powershell
dvc repro                      # preprocess -> train (see dvc.yaml)
python -m src.models.tune      # optional: 8-run grid, winner chosen by VALIDATION F1
python -m src.models.register  # register the last trained model; promote only if it passes the gate and beats the champion
python -m src.models.register --rollback   # make the previous champion the champion again
```

- Quality gate: validation F1 must be at least `quality_gate.min_f1` in `configs/model_params.yaml`.
- Tracking location comes from `MLFLOW_TRACKING_URI`; if it is not set, a local `mlflow.db` (SQLite) is used.
- After `tune`, copy the best values from `artifacts/best_params.yaml` into `configs/model_params.yaml`, then run `dvc repro`.

## 4. Run the tests and the linter

```powershell
ruff check src tests
python -m pytest
```

One test is skipped when `data/raw/imdb_reviews.csv` is not present (run `dvc pull` to include it).
No test needs a network connection or the trained model.

## 5. Fine-tune the Hugging Face challenger (free GPU)

Use a free **Google Colab** (GPU runtime) or **Kaggle Notebook** (GPU accelerator). Clone this repository
there, configure DVC credentials as in section 2, then run from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
dvc pull
python -m src.models.hf_finetune --smoke  # optional: 1 epoch / 1,000 rows confirms the GPU setup
python -m src.models.hf_finetune
```

For Colab, select **Runtime → Change runtime type → T4 GPU** first. For Kaggle, select
**Settings → Accelerator → GPU** and enable Internet only long enough to download the base model and Python
packages. `nvidia-smi` should show a GPU before the run. The defaults in `configs/hf_params.yaml` are intended
for a free T4/P100 session; change only that file if you need a smaller batch size or a faster subset run.

The command uses the project's existing 70/15/15 splits, logs a `hf-distilbert-finetune` run to MLflow, saves
the reusable Hugging Face files under `artifacts/hf_finetuned/`, and writes
`artifacts/hf_finetune_report.json`. The report says whether it passes the quality gate and beats the baseline
on validation F1. It deliberately does **not** change the live Flask champion: Flask currently serves sklearn
models, while the fine-tuned transformer will be introduced through the planned BentoML service. This prevents
a registry promotion from producing an API that cannot load its champion.

## 6. Prediction API

Local development server (binds to `127.0.0.1` only):

```powershell
python -m src.serving.app
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | `200` when a model is loaded, `503` otherwise |
| GET | `/model-info` | model name, version, source and metadata |
| POST | `/predict` | body `{"text": "..."}` or `{"texts": ["...", "..."]}` |
| GET | `/metrics` | Prometheus metrics |

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType "application/json" -Body '{"text":"This movie was excellent"}'
```

The response contains `model_version` and a `predictions` list with `label`, `class_id` and `confidence`.
Limits: at most 10,000 characters per text, 32 texts per request, 1 MB request body. Errors are JSON.
Review text is never written to the logs.

Which model is served is controlled by environment variables (see `src/serving/model_loader.py`):
`MODEL_SOURCE` (`auto`, `registry` or `file`), `MODEL_PATH`, `MODEL_NAME`, `MODEL_ALIAS`, `MLFLOW_TRACKING_URI`.
In `auto` mode the registry `champion` is tried first and `artifacts/model.joblib` is the fallback.

## 7. Docker

The API image copies `artifacts/model.joblib`, so get it first (`dvc pull` or `dvc repro`).
Optionally copy `.env.example` to `.env` to set `MLFLOW_DB_PASSWORD`.

```powershell
docker build -f docker/Dockerfile.api -t sentiment-api:local .
docker compose up --build
```

This starts the API (<http://localhost:8000/health>), an MLflow tracking server (<http://localhost:5000>)
and its PostgreSQL backend. The API container serves the saved model file (`MODEL_SOURCE=file`);
it does not read the registry. To train against the containerised MLflow:

```powershell
docker compose up -d mlflow-db mlflow
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m src.models.train
```

All published ports are bound to `127.0.0.1`. PostgreSQL is reachable only inside the Compose network.
If `MLFLOW_DB_PASSWORD` is not set, a local-only default password is used; never reuse it in a deployed
environment. Stop with `docker compose down`. Add `-v` only if you want to delete the MLflow database
and artifact volumes.

## 8. Continuous integration

`.github/workflows/ci.yml` runs on pull requests to `main` and on pushes to `main`:

1. **quality**: `ruff`, `pytest`, then `pip-audit` and `bandit` (report only).
2. **docker** (only if `quality` passed): `dvc pull` of the model, image build, container smoke test,
   Trivy scan (report only), and publish to GitHub Container Registry on pushes to `main`.

The `docker` job needs the repository secrets `DAGSHUB_USER` and `DAGSHUB_TOKEN`.
`.github/workflows/deploy.yml` is a manual placeholder until the deployment step exists.

## 9. Security notes

- Secrets live only in `.env`, `.dvc/config.local` or repository secrets, all of which are git-ignored or external.
- The model file is a joblib/pickle file. Unpickling can run code, so only load files you produced yourself.
- The API container runs as a non-root user; in Docker Compose it also drops all Linux capabilities and sets `no-new-privileges`.

## Dataset

Reviews come from the Kaggle "IMDB Dataset of 50K Movie Reviews" (`IMDB Dataset.csv`), a copy of the
Large Movie Review Dataset released by Stanford. When using it, cite:

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. (2011).
> Learning Word Vectors for Sentiment Analysis. *Proceedings of the 49th Annual Meeting of the Association
> for Computational Linguistics: Human Language Technologies*, pp. 142-150.

The dataset is not committed to Git (DVC remote only). It has no standard open-source licence attached;
check the dataset page's terms before redistributing it.
