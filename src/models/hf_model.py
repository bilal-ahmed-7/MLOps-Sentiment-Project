"""A Hugging Face text-classification model behind an sklearn-like interface.

predict(texts) and predict_proba(texts) work like the TF-IDF pipeline, so evaluation,
the quality gate and (later) the API can treat both model types the same way.
Class ids follow the project convention: 0 = negative, 1 = positive.
"""
import numpy as np

POSITIVE_LABELS = {"positive", "pos", "label_1", "1"}
NEGATIVE_LABELS = {"negative", "neg", "label_0", "0"}


def label_to_id(label: str) -> int:
    name = str(label).strip().lower()
    if name in POSITIVE_LABELS:
        return 1
    if name in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unknown sentiment label from the model: {label!r}")


class HFSentimentClassifier:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        predictor=None,
    ):
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self._predictor = predictor or self._load_pipeline()

    def _load_pipeline(self):
        from transformers import pipeline  # imported here: only needed when a real model is used

        return pipeline("text-classification", model=self.model_name, revision=self.revision, top_k=None)

    def predict_proba(self, texts) -> np.ndarray:
        texts = list(texts)
        outputs = self._predictor(
            texts, truncation=True, max_length=self.max_length, batch_size=self.batch_size
        )
        proba = np.zeros((len(texts), 2))
        for row, scores in enumerate(outputs):
            for item in scores:
                proba[row, label_to_id(item["label"])] = item["score"]
        return proba

    def predict(self, texts) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)
