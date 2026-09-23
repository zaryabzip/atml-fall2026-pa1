"""Prediction and source-validation evaluation (the checkpoint-selection rule for Tasks 2 and 3)."""
import numpy as np
import torch

from common.device import autocast, make_loader
from common.metrics import classification_metrics
from shared.pacs import NUM_CLASSES


@torch.no_grad()
def predict(model, loader, device, amp: bool = False, return_features: bool = False) -> dict:
    """Returns numpy arrays: y, pred, logits, and features if requested."""
    model.eval()
    ys, logits_all, feats = [], [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device, amp):
            features, logits = model(x, return_features=True)
        ys.append(y)
        logits_all.append(logits.float().cpu())
        if return_features:
            feats.append(features.float().cpu())
    logits_all = torch.cat(logits_all)
    out = {"y": torch.cat(ys).numpy(), "logits": logits_all.numpy(), "pred": logits_all.argmax(1).numpy()}
    if return_features:
        out["features"] = torch.cat(feats).numpy()
    return out


def val_loaders(datasets: dict, batch_size: int, device, num_workers=None) -> dict:
    return {d: make_loader(ds, batch_size, device, num_workers=num_workers) for d, ds in datasets.items()}


def evaluate_sources(model, loaders: dict, device, amp: bool = False) -> dict:
    per_domain = {}
    for domain, loader in loaders.items():
        p = predict(model, loader, device, amp)
        per_domain[domain] = classification_metrics(p["y"], p["pred"], NUM_CLASSES)
    accs = [m["accuracy"] for m in per_domain.values()]
    f1s = [m["macro_f1"] for m in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_accuracy": float(np.min(accs)),
        "worst_macro_f1": float(np.min(f1s)),
    }
