"""Append-only JSONL training log (one dict per line), easy to load with pandas.read_json(lines=True).
Named logger.py (not logging.py) so it never shadows the standard library module."""
import json
import time
from pathlib import Path

from common.io import _to_builtin


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, record: dict, echo: bool = False) -> None:
        record = {"time_s": round(time.time() - self.start, 1), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_to_builtin) + "\n")
        if echo:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
