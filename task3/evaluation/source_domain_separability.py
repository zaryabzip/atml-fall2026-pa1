"""Photo vs Art vs Cartoon separability from source-VALIDATION features only (chance = 33.3%)."""
import numpy as np

from common.metrics import linear_probe_accuracy
from shared.pacs_eval import predict


def source_domain_separability(model, source_val_loaders, device, seed=6304) -> float:
    """Held-out accuracy of a linear probe that predicts WHICH source domain (Photo, Art Painting or
    Cartoon) an image came from, using the model's frozen 512-d features. A lower score means the
    source domains are harder to tell apart in feature space. It says nothing by itself about class
    information or Sketch performance. Uses no Sketch image."""
    # Extract frozen features for every source-validation image, one domain at a time.
    features_by_domain = []
    for domain, loader in source_val_loaders.items():
        out = predict(model, loader, device, return_features=True)
        features_by_domain.append(out["features"])

    # Subsample every domain down to the smallest domain's size, so the probe can't win by
    # just predicting the biggest domain. Seeded for reproducibility.
    n = min(len(f) for f in features_by_domain)
    rng = np.random.default_rng(seed)
    kept = [f[rng.choice(len(f), size=n, replace=False)] for f in features_by_domain]

    # Labels 0, 1, 2 follow the loader order; the probe is a 70/30 multinomial logistic regression.
    features = np.concatenate(kept, axis=0)
    labels = np.concatenate([np.full(n, i) for i in range(len(kept))])
    return linear_probe_accuracy(features, labels, seed, test_fraction=0.3, C=1.0)
