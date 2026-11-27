"""Unit tests for fine-tuning helpers; no GPU or Hugging Face download is required."""
import pandas as pd
import pytest

from src.models.hf_finetune import limited, load_finetune_params


def test_finetuning_config_has_required_settings():
    params = load_finetune_params()
    assert {"base_model", "max_length", "train_batch_size", "num_train_epochs"} <= set(params)


def test_limited_is_reproducible_and_keeps_both_labels():
    frame = pd.DataFrame({"text": [f"review {i}" for i in range(20)], "label": [0] * 10 + [1] * 10})
    first, second = limited(frame, 8, 42), limited(frame, 8, 42)
    assert len(first) == 8 and first.equals(second)
    assert set(first["label"]) == {0, 1}


def test_limited_rejects_too_small_subset():
    frame = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError, match="at least 2"):
        limited(frame, 1, 42)
