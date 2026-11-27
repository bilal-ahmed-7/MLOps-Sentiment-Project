"""Fine-tune a Hugging Face sentiment challenger on this project's IMDB splits.

Use a GPU runtime (Colab or Kaggle) for this workflow::

    python -m pip install -r requirements.txt
    dvc pull
    python -m src.models.hf_finetune

The training, validation and test CSVs are the same fixed, stratified splits used
by the TF-IDF champion.  This makes the resulting comparison fair.  The script
does *not* change the Flask champion: that service currently loads sklearn MLflow
models only.  It records whether the transformer is eligible to replace it once
the BentoML serving path is introduced.
"""
import argparse
import inspect
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from src.data.ingestion import load_config
from src.models.evaluate import passes_quality_gate
from src.models.hf_eval import score
from src.models.hf_model import HFSentimentClassifier
from src.models.train import get_data_version, get_git_info, load_params, load_splits, setup_tracking

logger = logging.getLogger(__name__)
HF_PARAMS = Path("configs/hf_params.yaml")
BASELINE_MODEL = Path("artifacts/model.joblib")
MODEL_DIR = Path("artifacts/hf_finetuned")
CHECKPOINT_DIR = Path("artifacts/hf_training_checkpoints")
REPORT_FILE = Path("artifacts/hf_finetune_report.json")


def load_finetune_params(path: Path = HF_PARAMS) -> dict:
    """Read and minimally validate the fine-tuning section of the HF config."""
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)["finetuning"]
    required = {"base_model", "max_length", "train_batch_size", "eval_batch_size",
                "learning_rate", "weight_decay", "num_train_epochs", "seed", "train_max_rows"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing finetuning settings: {', '.join(sorted(missing))}")
    if params["max_length"] <= 0 or params["train_batch_size"] <= 0 or params["eval_batch_size"] <= 0:
        raise ValueError("max_length and batch sizes must be positive")
    return params


def limited(frame, max_rows, seed: int):
    """Return a deterministic, stratified training subset when requested."""
    if max_rows is None or max_rows >= len(frame):
        return frame
    if max_rows < 2:
        raise ValueError("train_max_rows must be at least 2")
    # pandas' groupby sample keeps both sentiment classes represented.
    fraction = max_rows / len(frame)
    pieces = []
    for _, group in frame.groupby("label", group_keys=False):
        n = max(1, round(len(group) * fraction))
        pieces.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(pieces).sample(frac=1, random_state=seed).head(max_rows)


def trainer_arguments(config: dict, output_dir: Path):
    """Build TrainingArguments across the supported transformers API spellings."""
    import torch
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config["train_batch_size"],
        "per_device_eval_batch_size": config["eval_batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "num_train_epochs": config["num_train_epochs"],
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_strategy": "no",
        "report_to": [],
        "seed": config["seed"],
        "fp16": torch.cuda.is_available(),
    }
    argument_names = inspect.signature(TrainingArguments).parameters
    name = "eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"
    kwargs[name] = "no"  # final scoring below is the sole validation decision.
    return TrainingArguments(**kwargs)


def as_dataset(frame, tokenizer, max_length):
    from datasets import Dataset

    dataset = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True)


def train_and_evaluate(cfg: dict, params: dict, hp: dict) -> dict:
    """Fine-tune, log a portable MLflow model, and compare fairly to the baseline."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, set_seed)
    import transformers

    train, val, test = load_splits(cfg)
    train = limited(train, hp["train_max_rows"], hp["seed"])
    tokenizer = AutoTokenizer.from_pretrained(hp["base_model"], revision=hp.get("revision"))
    model = AutoModelForSequenceClassification.from_pretrained(
        hp["base_model"], revision=hp.get("revision"), num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    set_seed(hp["seed"])
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    trainer = Trainer(
        model=model,
        args=trainer_arguments(hp, CHECKPOINT_DIR),
        train_dataset=as_dataset(train, tokenizer, hp["max_length"]),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    with mlflow.start_run(run_name="hf-distilbert-finetune") as run:
        mlflow.set_tags({
            **get_git_info(),
            "dvc_data_md5": get_data_version(cfg),
            "model_family": "huggingface_finetuned",
        })
        mlflow.log_params({
            "model_type": "huggingface_finetuned",
            **hp,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        })
        started = time.perf_counter()
        trainer.train()
        mlflow.log_metric("training_seconds", round(time.perf_counter() - started, 2))

        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        trainer.save_model(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        candidate = HFSentimentClassifier(str(MODEL_DIR), max_length=hp["max_length"], batch_size=hp["eval_batch_size"])
        val_metrics, val_seconds = score(candidate, val)
        test_metrics, test_seconds = score(candidate, test)
        n_scored = len(val) + len(test)
        candidate_ms = round(1000 * (val_seconds + test_seconds) / n_scored, 2)
        metrics = {**{f"val_{k}": v for k, v in val_metrics.items()},
                   **{f"test_{k}": v for k, v in test_metrics.items()},
                   "inference_ms_per_review": candidate_ms}
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(MODEL_DIR), "hf_finetuned_files")
        # This is a real MLflow transformers model artifact, ready to register/serve via BentoML later.
        mlflow.transformers.log_model(transformers_model=model, artifact_path="model", tokenizer=tokenizer)

        report = {"base_model": hp["base_model"], "n_train": len(train), "n_val": len(val), "n_test": len(test),
                  "device": "cuda" if torch.cuda.is_available() else "cpu",
                  "transformers_version": transformers.__version__, "torch_version": torch.__version__,
                  "candidate": {**val_metrics, "test_f1": test_metrics["f1"], "ms_per_review": candidate_ms}}
        if BASELINE_MODEL.exists():
            baseline = joblib.load(BASELINE_MODEL)
            baseline_val, base_val_seconds = score(baseline, val)
            baseline_test, base_test_seconds = score(baseline, test)
            base_ms = round(1000 * (base_val_seconds + base_test_seconds) / n_scored, 2)
            mlflow.log_metrics({"baseline_val_f1": baseline_val["f1"], "baseline_test_f1": baseline_test["f1"],
                                "baseline_ms_per_review": base_ms})
            report["baseline"] = {**baseline_val, "test_f1": baseline_test["f1"], "ms_per_review": base_ms}

    report["mlflow_run_id"] = run.info.run_id
    report["passed_quality_gate"] = passes_quality_gate(val_metrics, params["quality_gate"]["min_f1"])
    report["beats_baseline"] = "baseline" in report and val_metrics["f1"] > report["baseline"]["f1"]
    report["eligible_for_bentoml_promotion"] = report["passed_quality_gate"] and report["beats_baseline"]
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face challenger on the project splits.")
    parser.add_argument("--smoke", action="store_true", help="train on 1,000 rows for a quick GPU setup check")
    args = parser.parse_args()
    try:
        hp = load_finetune_params()
        if args.smoke:
            hp["train_max_rows"] = 1000
            hp["num_train_epochs"] = 1
        cfg, params = load_config(), load_params()
        setup_tracking(params)
        report = train_and_evaluate(cfg, params, hp)
    except ImportError as exc:
        logger.error("Missing fine-tuning dependency: %s. Run: python -m pip install -r requirements.txt", exc)
        return 1
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
