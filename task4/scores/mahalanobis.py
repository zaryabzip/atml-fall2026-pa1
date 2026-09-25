import numpy as np


def fit_class_gaussians(train_features: np.ndarray, train_labels: np.ndarray, num_classes: int = 10):
    """Class means mu_c and ONE shared DIAGONAL covariance from UNAUGMENTED CIFAR-10 train features.
    The shared variance is the per-dimension variance of the features around their own class mean,
    pooled over all classes, plus 1e-6. Returns (means [C, D], variances [D])."""
    feats = train_features.astype(np.float64)
    means = np.stack([feats[train_labels == c].mean(axis=0) for c in range(num_classes)])
    centered = feats - means[train_labels]
    variances = centered.var(axis=0, ddof=0) + 1e-6
    return means, variances


def mahalanobis(features: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """u = min_c sum_d (f_d - mu_cd)^2 / var_d."""
    feats = features.astype(np.float64)
    dists = (((feats[:, None, :] - means[None, :, :]) ** 2) / variances).sum(axis=2)  # [N, C]
    return dists.min(axis=1)
