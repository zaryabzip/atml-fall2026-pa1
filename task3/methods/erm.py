"""ERM = the Task 2 Source-only checkpoint, loaded unchanged."""
from common.io import checkpoint_dir
from shared.pacs_models import load_pacs_checkpoint


def load_erm(device):
    path = checkpoint_dir("task2", "source_only") / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Train Task 2 source_only first.")
    return load_pacs_checkpoint(path, device)
