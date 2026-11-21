"""Tests for the tuning helpers (no training involved)."""
from src.models.train import load_params
from src.models.tune import build_trials, pick_best


def test_build_trials_covers_full_grid():
    params = load_params()
    trials = build_trials(params)
    grid = params["tuning"]["grid"]
    expected = len(grid["logreg.C"]) * len(grid["tfidf.ngram_range"])
    assert len(trials) == expected
    combos = {(t["logreg"]["C"], tuple(t["tfidf"]["ngram_range"])) for t in trials}
    assert len(combos) == expected  # all distinct


def test_build_trials_does_not_mutate_base_params():
    params = load_params()
    original_c = params["logreg"]["C"]
    build_trials(params)
    assert params["logreg"]["C"] == original_c


def test_pick_best_uses_validation_f1_not_test_f1():
    results = [
        {"run_id": "a", "metrics": {"val_f1": 0.90, "test_f1": 0.99}},
        {"run_id": "b", "metrics": {"val_f1": 0.92, "test_f1": 0.80}},
    ]
    assert pick_best(results)["run_id"] == "b"
