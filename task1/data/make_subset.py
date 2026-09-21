"""Stratified 80/20 train/val split of the official STL-10 train set and a class-balanced
500-image test subset, both with seed 6304. Saves the image indices.

    python -m task1.data.make_subset
"""
import numpy as np

from common.io import save_json
from common.metrics import stratified_split
from common.seed import SEED
from task1.data.stl10 import SPLIT_FILE, STL10_CLASSES, load_stl10


def main(subset_size: int = 500, val_fraction: float = 0.2, seed: int = SEED):
    train_labels = np.asarray(load_stl10("train").labels)
    train_idx, val_idx = stratified_split(train_labels, val_fraction, seed)

    test_labels = np.asarray(load_stl10("test").labels)
    rng = np.random.default_rng(seed)
    per_class = subset_size // len(STL10_CLASSES)
    chosen, shortfall = [], {}
    for c, name in enumerate(STL10_CLASSES):
        idx = np.flatnonzero(test_labels == c)
        take = min(per_class, len(idx))
        if take < per_class:
            shortfall[name] = int(take)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())

    save_json({
        "seed": seed,
        "classes": STL10_CLASSES,
        "train": train_idx, "val": val_idx,
        "eval_subset": sorted(chosen),
        "eval_subset_shortfall": shortfall,  # classes with fewer than subset_size/10 images (document it)
    }, SPLIT_FILE)
    print(f"train={len(train_idx)} val={len(val_idx)} eval_subset={len(chosen)} shortfall={shortfall}")
    print(f"Saved {SPLIT_FILE}")


if __name__ == "__main__":
    main()
