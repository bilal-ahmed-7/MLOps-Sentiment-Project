"""Hyperparameter tuning: a grid search where every trial is logged to MLflow.

Run from the project root:
    python -m src.models.tune

Models are compared on VALIDATION F1 only; the test set is reported but never
used to choose the winner.
"""
import copy
import itertools
import logging
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.data.ingestion import load_config
from src.models.train import flatten_params, load_params, run_experiment, setup_tracking

logger = logging.getLogger(__name__)


def build_trials(params: dict) -> list[dict]:
    """Expand params['tuning']['grid'] into one full params dict per combination."""
    grid = params["tuning"]["grid"]
    keys = list(grid)
    trials = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        trial = copy.deepcopy(params)
        for key, value in zip(keys, combo):
            section, name = key.split(".", 1)
            trial[section][name] = value
        trials.append(trial)
    return trials


def pick_best(results: list[dict]) -> dict:
    """Return the result with the highest validation F1 (first one wins a tie)."""
    return max(results, key=lambda r: r["metrics"]["val_f1"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg, params = load_config(), load_params()
    setup_tracking(params)
    client = mlflow.MlflowClient()
    session = time.strftime("%Y%m%d-%H%M%S")

    trials = build_trials(params)
    logger.info("Running %d trials (session %s)", len(trials), session)
    results = []
    for i, trial in enumerate(trials, start=1):
        _, metrics, run_id = run_experiment(cfg, trial, run_name=f"tune-{session}-{i:02d}")
        client.set_tag(run_id, "tuning_session", session)
        results.append({"run_id": run_id, "params": trial, "metrics": metrics})

    best = pick_best(results)
    client.set_tag(best["run_id"], "selected", "true")

    rows = [
        {"run_id": r["run_id"], **{k: v for k, v in flatten_params(r["params"]).items()
                                  if k in ("logreg_C", "tfidf_ngram_range")},
         **{k: r["metrics"][k] for k in ("val_f1", "val_accuracy", "test_f1")}}
        for r in results
    ]
    summary = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    Path("artifacts").mkdir(exist_ok=True)
    summary.to_csv("artifacts/tuning_summary.csv", index=False)

    best_out = {
        "selected_run_id": best["run_id"],
        "val_f1": best["metrics"]["val_f1"],
        "tfidf": best["params"]["tfidf"],
        "logreg": best["params"]["logreg"],
    }
    Path("artifacts/best_params.yaml").write_text(yaml.safe_dump(best_out, sort_keys=False), encoding="utf-8")

    print("\n" + summary.to_string(index=False))
    print(f"\nBest run: {best['run_id']}  (val_f1={best['metrics']['val_f1']})")
    print("Copy the tfidf/logreg values from artifacts/best_params.yaml into configs/model_params.yaml,")
    print("then run `dvc repro` to train and save the final model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())