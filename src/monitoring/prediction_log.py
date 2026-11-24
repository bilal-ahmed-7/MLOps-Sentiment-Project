"""Privacy-friendly prediction log: one JSON line per prediction, WITHOUT the review text.

Each record holds an id, timestamp, model version, label, confidence and numeric text
features. It feeds drift checks and (with ground-truth labels) performance tracking.
Logging problems never break an API request.
"""
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


class PredictionLogger:
    def __init__(self, path: str, max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._warned = False

    def log(self, records: list[dict]) -> None:
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                    self.path.replace(self.path.with_name(self.path.name + ".1"))  # keep one backup
                with self.path.open("a", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            if not self._warned:  # warn once, not on every request
                logger.warning("Could not write the prediction log at %s", self.path)
                self._warned = True
