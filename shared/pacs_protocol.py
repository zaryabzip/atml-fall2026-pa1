"""The PACS protocol shared by Tasks 2 and 3: sources, target, splits, transforms, BatchNorm policy.

Sketch is never written into the split file. It is listed at runtime only through
`target_dataset`, which refuses any purpose other than Task 2 adaptation or a final evaluation.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms as T

from common.device import make_loader
from common.io import PROJECT_ROOT, load_json, save_json
from common.metrics import stratified_split
from common.seed import SEED
from shared.pacs import PACSDataset, find_pacs_root, list_domain

SOURCES = ("photo", "art_painting", "cartoon")
TARGET = "sketch"
VAL_FRACTION = 0.2
SPLIT_FILE = PROJECT_ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"

# Normalization of torchvision ResNet18_Weights.IMAGENET1K_V1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ALLOWED_TARGET_PURPOSES = {"task2_adaptation", "task2_final_eval", "task3_final_eval"}


def train_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---- Splits ----

def make_splits(pacs_root, out_path=SPLIT_FILE, seed: int = SEED) -> dict:
    pacs_root = find_pacs_root(pacs_root)
    split = {"seed": seed, "val_fraction": VAL_FRACTION, "sources": {}}
    for domain in SOURCES:
        samples = list_domain(pacs_root, domain)
        labels = [label for _, label in samples]
        train_idx, val_idx = stratified_split(labels, VAL_FRACTION, seed)
        split["sources"][domain] = {
            "train": [samples[i] for i in train_idx],
            "val": [samples[i] for i in val_idx],
        }
    save_json(split, out_path)
    return split


def load_splits(path=SPLIT_FILE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m shared.make_pacs_splits --pacs-root <dir>")
    return load_json(path)


def source_datasets(pacs_root, part: str, transform, splits=None) -> dict:
    """{domain: PACSDataset} for part in {'train', 'val'}."""
    pacs_root = find_pacs_root(pacs_root)
    splits = splits or load_splits()
    return {d: PACSDataset(pacs_root, splits["sources"][d][part], d, transform) for d in SOURCES}


def target_dataset(pacs_root, transform, *, purpose: str) -> PACSDataset:
    if purpose not in ALLOWED_TARGET_PURPOSES:
        raise PermissionError(f"Sketch access not allowed for purpose '{purpose}'")
    pacs_root = find_pacs_root(pacs_root)
    return PACSDataset(pacs_root, list_domain(pacs_root, TARGET), TARGET, transform)


# ---- Batching ----

def cycle(loader):
    """Endless iterator; each pass re-shuffles if the loader shuffles."""
    while True:
        yield from loader


def balanced_source_batches(datasets: dict, per_domain: int, device, seed: int, num_workers=None):
    """Yields {'x', 'y', 'domain'} with `per_domain` examples from each source domain."""
    iterators = [
        cycle(make_loader(ds, per_domain, device, shuffle=True, drop_last=True, seed=seed + i, num_workers=num_workers))
        for i, ds in enumerate(datasets.values())
    ]
    while True:
        parts = [next(it) for it in iterators]
        yield {
            "x": torch.cat([p[0] for p in parts]),
            "y": torch.cat([p[1] for p in parts]),
            "domain": torch.cat([p[2] for p in parts]),
        }


# ---- BatchNorm policy ----

def set_train_mode_frozen_bn(model: nn.Module) -> nn.Module:
    """model.train(), then put only BatchNorm layers in eval mode so running stats stay at ImageNet
    values. BN gamma/beta still receive gradients."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model
