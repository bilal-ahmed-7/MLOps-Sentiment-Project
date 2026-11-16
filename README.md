# MLOps Sentiment Analysis Project

Sentiment analysis and text classification with an end-to-end MLOps pipeline.

## Local container stack

The API image includes the trained model, which is generated locally and is not
committed to Git. Create an environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Make sure the configured dataset is available (use `dvc pull` if it is stored
in the configured DVC remote), then preprocess it:

```powershell
dvc pull data/raw/imdb_reviews.csv
python -m src.features.preprocessing
```

Start the local MLflow tracking server and train against it:

```powershell
docker compose up -d mlflow-db mlflow
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m src.models.train
```

Build and start the API, MLflow tracking server, and PostgreSQL backend:

```powershell
docker compose up --build
```

- API health: <http://localhost:8000/health>
- API metrics: <http://localhost:8000/metrics>
- MLflow UI: <http://localhost:5000>

Try a prediction from PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType "application/json" -Body '{"text":"This movie was excellent"}'
```

Both web services bind to loopback and are intended for local development.
The PostgreSQL service is only reachable inside the Compose network. Set
`MLFLOW_DB_PASSWORD` in the environment before starting Compose to override the
local-only default password; use URL-safe characters. Do not use the default
credentials in a deployed environment.

Stop the services with `docker compose down`. The named database and artifact
volumes are retained between runs.
