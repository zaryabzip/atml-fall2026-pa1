import numpy as np
from scipy.special import logsumexp


def energy(logits: np.ndarray) -> np.ndarray:
    """u = -logsumexp_k z_k."""
    return -logsumexp(logits, axis=1)
