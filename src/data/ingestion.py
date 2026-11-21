"""Data ingestion: load the raw dataset described in configs/pipeline_config.yaml.

Run from the project root:
    python -m src.data.ingestion
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.validation import DataValidationError, quality_report, validate

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/pipeline_config.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(cfg: dict) -> pd.DataFrame:
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `dvc pull` or download the dataset."
        )
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows from %s", len(df), raw_path)
    return df


def ingest(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load and validate the raw data. Used by later pipeline stages."""
    cfg = load_config(config_path)
    df = load_raw_data(cfg)
    return validate(df, cfg)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        cfg = load_config()
        df = validate(load_raw_data(cfg), cfg)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        return 1  # non-zero exit lets DVC/Airflow/CI fail the pipeline
    logger.info("Validation passed. Report: %s", quality_report(df, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
