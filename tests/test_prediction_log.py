"""Tests for the privacy-friendly prediction logger."""
import json

from src.monitoring.prediction_log import PredictionLogger


def test_writes_one_json_line_per_record(tmp_path):
    path = tmp_path / "logs" / "predictions.jsonl"  # parent folder is created on demand
    PredictionLogger(str(path)).log([{"prediction_id": "a-0"}, {"prediction_id": "a-1"}])
    lines = path.read_text().splitlines()
    assert [json.loads(line)["prediction_id"] for line in lines] == ["a-0", "a-1"]


def test_appends_across_calls(tmp_path):
    logger = PredictionLogger(str(tmp_path / "p.jsonl"))
    logger.log([{"n": 1}])
    logger.log([{"n": 2}])
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 2


def test_rotates_when_file_is_too_large(tmp_path):
    path = tmp_path / "p.jsonl"
    logger = PredictionLogger(str(path), max_bytes=50)
    logger.log([{"padding": "x" * 100}])
    logger.log([{"n": 2}])
    assert (tmp_path / "p.jsonl.1").exists()
    assert json.loads(path.read_text().splitlines()[0]) == {"n": 2}


def test_unwritable_path_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    PredictionLogger(str(blocker / "sub" / "p.jsonl")).log([{"n": 1}])  # parent is a file -> OSError, swallowed
