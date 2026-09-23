"""Where a saved checkpoint lives, WITHOUT creating any directory.

common.io.checkpoint_dir() also does mkdir, which is right for training but wrong for read-only checks:
asking whether a mistyped run exists would leave an empty folder behind."""
import os
from pathlib import Path

from common.io import PROJECT_ROOT


def checkpoint_file(task: str, run_name: str) -> Path:
    root = Path(os.environ.get("PA1_OUT_ROOT", PROJECT_ROOT))
    return root / "checkpoints" / task / run_name / "best.pt"
