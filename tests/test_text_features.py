"""Tests for the shared numeric text features."""
import pytest

from src.features.text_features import FEATURE_NAMES, extract_features


def test_feature_names_match_output_keys():
    assert set(extract_features("Some text")) == set(FEATURE_NAMES)


def test_basic_values():
    feats = extract_features("Great film!! Loved IT?")
    assert feats["char_length"] == 22.0
    assert feats["word_count"] == 4.0
    assert feats["exclamation_count"] == 2.0
    assert feats["question_count"] == 1.0
    assert feats["uppercase_ratio"] == pytest.approx(4 / 16, abs=1e-3)  # G, L, I, T out of 16 letters


def test_empty_text_does_not_divide_by_zero():
    feats = extract_features("")
    assert feats["word_count"] == 0.0 and feats["avg_word_length"] == 0.0 and feats["uppercase_ratio"] == 0.0
