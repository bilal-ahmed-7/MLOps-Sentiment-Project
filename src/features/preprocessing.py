"""Text cleaning, de-duplication, label encoding and train/val/test split.

Run from the project root:
    python -m src.features.preprocessing
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingestion import ingest, load_config
from src.data.validation import DataValidationError
from src.features.text_cleaning import clean_text

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a clean frame with columns: review_id, text, label."""
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    min_len = cfg["validation"]["min_text_length"]

    out = df[[text_col, label_col]].copy()
    out[text_col] = out[text_col].map(clean_text)
    out = out[out[text_col].str.len() >= min_len]
    # De-duplicate BEFORE splitting so the same review can't land in both train and test
    out = out.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    out["label"] = out[label_col].map(cfg["data"]["label_map"]).astype(int)
    out = out.rename(columns={text_col: "text"})
    out.insert(0, "review_id", range(len(out)))  # stable entity key (used by Feast later)
    return out[["review_id", "text", "label"]]


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/val/test split; fractions are relative to all rows."""
    p = cfg["preprocessing"]
    seed = p["random_seed"]
    train_val, test = train_test_split(
        df, test_size=p["test_size"], stratify=df["label"], random_state=seed
    )
    val_fraction = p["val_size"] / (1 - p["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["label"], random_state=seed
    )
    return tuple(d.sort_values("review_id").reset_index(drop=True) for d in (train, val, test))


def save_splits(train, val, test, cfg: dict) -> None:
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out_dir / f"{name}.csv", index=False)
        logger.info("Saved %s: %d rows", name, len(part))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        cfg = load_config()
        raw = ingest()  # loads AND validates; stops the pipeline if data is bad
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        return 1
    clean = preprocess(raw, cfg)
    logger.info("Rows: %d raw -> %d after cleaning and de-duplication", len(raw), len(clean))
    save_splits(*split_data(clean, cfg), cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())