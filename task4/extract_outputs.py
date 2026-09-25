"""Save penultimate features + logits once, so every score uses identical outputs.

    python -m task4.extract_outputs --run vanilla                     # CIFAR-10 train/val/test only
    python -m task4.extract_outputs --run vanilla --include-unknowns  # ONLY after all models + scores are fixed
Writes task4/cache/<run>/<part>.npz with features, logits, labels.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from common.device import get_device, make_loader
from common.io import PROJECT_ROOT, checkpoint_dir
from task4.data.cifar import cifar10_test, cifar10_train, cifar10_val, cifar100_unknowns, eval_transform
from task4.models.resnet_cifar import CifarResNet18


@torch.no_grad()
def extract(model, loader, device) -> dict:
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        f, z = model(x.to(device, non_blocking=True), return_features=True)
        feats.append(f.float().cpu())
        logits.append(z.float().cpu())
        labels.append(y)
    return {"features": torch.cat(feats).numpy(), "logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--include-unknowns", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N examples per part (preflight checks).")
    parser.add_argument("--cache-root", default=None, help="Default: task4/cache")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(checkpoint_dir("task4", args.run) / "best.pt", map_location=device, weights_only=False)
    model = CifarResNet18(num_classes=ckpt["num_outputs"])
    model.load_state_dict(ckpt["model"])
    model.to(device)

    parts = {
        "train": cifar10_train(eval_transform()),  # unaugmented, for Mahalanobis statistics
        "val": cifar10_val(),
        "test": cifar10_test(),
    }
    if args.include_unknowns:
        parts["near"] = cifar100_unknowns("near")
        parts["far"] = cifar100_unknowns("far")

    out_dir = (Path(args.cache_root) if args.cache_root else PROJECT_ROOT / "task4" / "cache") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in parts.items():
        if args.limit:
            ds = Subset(ds, range(min(args.limit, len(ds))))
        # num_workers=0: loading in the main process. With worker processes, the loader of the previous part
        # could hang forever while shutting its workers down. CIFAR is in memory, so this costs little.
        arrays = extract(model, make_loader(ds, args.batch_size, device, num_workers=0), device)
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        print(f"{name}: {arrays['logits'].shape[0]} examples -> {out_dir / name}.npz")


if __name__ == "__main__":
    main()
