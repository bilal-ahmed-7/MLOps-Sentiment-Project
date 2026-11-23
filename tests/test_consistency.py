"""Guard: constants duplicated between config files and code must not drift apart."""
from pathlib import Path

import yaml

from src.models import register
from src.serving import app as api
from src.serving import model_loader

ROOT = Path(__file__).resolve().parent.parent


def _load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_api_labels_match_label_map_in_pipeline_config():
    label_map = _load("configs/pipeline_config.yaml")["data"]["label_map"]
    assert api.LABELS == {class_id: name for name, class_id in label_map.items()}


def test_registered_model_name_is_the_same_everywhere():
    assert api.MODEL_NAME == register.MODEL_NAME == model_loader.DEFAULT_MODEL_NAME


def test_registry_aliases_match_between_registry_and_loader():
    assert register.CHAMPION == model_loader.DEFAULT_ALIAS


def test_pipeline_config_has_every_key_the_code_reads():
    cfg = _load("configs/pipeline_config.yaml")
    for key in ("raw_path", "processed_dir", "text_column", "label_column", "label_map"):
        assert key in cfg["data"], f"data.{key} missing"
    for key in ("min_text_length", "expected_min_rows"):
        assert key in cfg["validation"], f"validation.{key} missing"
    for key in ("random_seed", "test_size", "val_size"):
        assert key in cfg["preprocessing"], f"preprocessing.{key} missing"


def test_model_params_has_every_key_the_code_reads():
    params = _load("configs/model_params.yaml")
    assert {"experiment_name", "model_type", "tfidf", "logreg", "quality_gate", "tuning"} <= set(params)
    assert {"max_features", "ngram_range", "min_df"} <= set(params["tfidf"])
    assert {"C", "max_iter"} <= set(params["logreg"])
    assert "min_f1" in params["quality_gate"]
    assert "grid" in params["tuning"]
