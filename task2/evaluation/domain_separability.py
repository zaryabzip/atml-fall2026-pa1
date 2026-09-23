"""Source-vs-target separability (Task 2 alignment diagnostic)."""
import numpy as np

from common.metrics import linear_probe_accuracy
from shared.pacs_eval import predict


def domain_separability(model, source_val_loaders, target_loader, device, seed=6304) -> float:
    """Held-out accuracy of a linear probe trained to tell source-domain features apart from
    target-domain (Sketch) features. 50% = chance (indistinguishable); 100% = trivially separable.

    Pooling choice: the 3 source-validation sets (photo/art_painting/cartoon) are pooled together
    into one big "source" pile before subsampling -- we are not balancing evenly across the 3
    source domains individually, just matching the total source count to the total target count."""
    # Extract frozen features for every source-validation image, pooling all 3 domains together.
    source_feats = []
    for domain, loader in source_val_loaders.items():
        out = predict(model, loader, device, return_features=True)
        source_feats.append(out["features"])
    source_feats = np.concatenate(source_feats, axis=0)  # (n_source_total, 512)

    # Extract frozen features for every Sketch (target) image.
    target_out = predict(model, target_loader, device, return_features=True)
    target_feats = target_out["features"]  # (n_target_total, 512)

    # Subsample down to equal counts on both sides, seeded for reproducibility.
    n = min(len(source_feats), len(target_feats))
    rng = np.random.default_rng(seed)
    source_idx = rng.choice(len(source_feats), size=n, replace=False)
    target_idx = rng.choice(len(target_feats), size=n, replace=False)
    source_feats = source_feats[source_idx]
    target_feats = target_feats[target_idx]

    # Train/evaluate a linear probe to distinguish source (0) from target (1).
    features = np.concatenate([source_feats, target_feats], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return linear_probe_accuracy(features, labels, seed, test_fraction=0.3, C=1.0)
