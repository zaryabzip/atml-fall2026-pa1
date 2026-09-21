"""Paths and result files.

PA1_DATA_ROOT  where datasets live (default: <project>/data). On Kaggle, e.g. /kaggle/working/data.
PA1_OUT_ROOT   where checkpoints go (default: <project>). On Kaggle, e.g. /kaggle/working.
Small result files (JSON/CSV) always go to <project>/taskN/results so they can be committed.
"""
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("PA1_DATA_ROOT", PROJECT_ROOT / "data"))


def checkpoint_dir(task: str, run_name: str) -> Path:
    path = Path(os.environ.get("PA1_OUT_ROOT", PROJECT_ROOT)) / "checkpoints" / task / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(task: str, run_name: str) -> Path:
    path = PROJECT_ROOT / task / "results" / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path):
    with open(path) as f:
        return json.load(f)
