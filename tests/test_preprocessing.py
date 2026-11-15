"""Unit tests for text cleaning, preprocessing and splitting."""
import pandas as pd
import pytest

from src.features.preprocessing import clean_text, preprocess, split_data

CFG = {
    "data": {
        "text_column": "review",
        "label_column": "sentiment",
        "label_map": {"negative": 0, "positive": 1},
    },
    "validation": {"min_text_length": 5},
    "preprocessing": {"random_seed": 42, "test_size": 0.2, "val_size": 0.2},
}


def make_raw(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review": [f"Review number {i} is here" for i in range(n)],
            "sentiment": ["positive" if i % 2 == 0 else "negative" for i in range(n)],
        }
    )


def test_clean_text_removes_html_and_extra_whitespace():
    assert clean_text("Great<br /><br />movie   indeed ") == "Great movie indeed"


def test_clean_text_keeps_case_and_punctuation():
    assert clean_text("Wow!!! LOVED it.") == "Wow!!! LOVED it."


def test_preprocess_columns_and_numeric_labels():
    out = preprocess(make_raw(10), CFG)
    assert list(out.columns) == ["review_id", "text", "label"]
    assert set(out["label"]) == {0, 1}


def test_preprocess_drops_duplicates_and_assigns_unique_ids():
    raw = make_raw(10)
    raw = pd.concat([raw, raw.iloc[:3]], ignore_index=True)  # 3 duplicates
    out = preprocess(raw, CFG)
    assert len(out) == 10
    assert out["review_id"].is_unique


def test_preprocess_drops_text_that_is_empty_after_cleaning():
    raw = make_raw(10)
    raw.loc[0, "review"] = "<br /><br />"
    assert len(preprocess(raw, CFG)) == 9


def test_split_sizes_add_up_and_do_not_overlap():
    df = preprocess(make_raw(100), CFG)
    train, val, test = split_data(df, CFG)
    assert (len(train), len(val), len(test)) == (60, 20, 20)
    ids = [set(part["review_id"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_stratified():
    df = preprocess(make_raw(100), CFG)
    for part in split_data(df, CFG):
        assert part["label"].mean() == pytest.approx(0.5, abs=0.05)


def test_split_is_deterministic():
    df = preprocess(make_raw(100), CFG)
    first = split_data(df, CFG)
    second = split_data(df, CFG)
    for a, b in zip(first, second):
        assert a.equals(b)