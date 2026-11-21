"""Data tests: schema, nulls, labels, text quality, and ingestion."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import load_config, load_raw_data
from src.data.validation import DataValidationError, quality_report, validate

CFG = {
    "data": {
        "raw_path": "data/raw/imdb_reviews.csv",
        "text_column": "review",
        "label_column": "sentiment",
        "label_map": {"negative": 0, "positive": 1},
    },
    "validation": {"min_text_length": 5, "expected_min_rows": 3},
}


def make_df(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "review": ["Great movie, loved it", "Terrible and boring", "Not bad at all"],
            "sentiment": ["positive", "negative", "positive"],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_valid_data_passes():
    df = make_df()
    assert validate(df, CFG).equals(df)


def test_null_text_fails():
    with pytest.raises(DataValidationError):
        validate(make_df(review=["ok review", None, "fine movie"]), CFG)


def test_unknown_label_fails():
    with pytest.raises(DataValidationError):
        validate(make_df(sentiment=["positive", "neutral", "negative"]), CFG)


def test_too_short_text_fails():
    with pytest.raises(DataValidationError):
        validate(make_df(review=["ok review", "   ", "fine movie"]), CFG)


def test_missing_column_fails():
    with pytest.raises(DataValidationError):
        validate(make_df().drop(columns=["sentiment"]), CFG)


def test_too_few_rows_fails():
    with pytest.raises(DataValidationError):
        validate(make_df().iloc[:2], CFG)


def test_quality_report_counts_duplicates():
    df = make_df(review=["same text here", "same text here", "other text here"])
    report = quality_report(df, CFG)
    assert report["n_rows"] == 3
    assert report["n_duplicate_texts"] == 1


def test_missing_raw_file_raises_clear_error():
    cfg = {**CFG, "data": {**CFG["data"], "raw_path": "data/raw/does_not_exist.csv"}}
    with pytest.raises(FileNotFoundError):
        load_raw_data(cfg)


@pytest.mark.skipif(
    not Path("data/raw/imdb_reviews.csv").exists(),
    reason="real dataset not present (run `dvc pull`)",
)
def test_real_dataset_passes_validation():
    cfg = load_config()
    assert len(validate(load_raw_data(cfg), cfg)) >= cfg["validation"]["expected_min_rows"]
