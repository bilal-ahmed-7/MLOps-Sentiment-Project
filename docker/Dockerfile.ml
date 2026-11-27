FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# DVC needs git to identify and use the configured remote.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ configs/
COPY dvc.yaml dvc.lock ./
COPY .dvc/ .dvc/
COPY data/raw/imdb_reviews.csv.dvc data/raw/imdb_reviews.csv.dvc
COPY src/ src/
COPY pipelines/ pipelines/

# Credentials are injected at runtime through a Kubernetes Secret and are never
# baked into this image.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser
