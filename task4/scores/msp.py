import numpy as np
from scipy.special import softmax


def msp(logits: np.ndarray) -> np.ndarray:
    """u = 1 - max_k softmax(z)_k (scipy's softmax is numerically stable). Larger = more novel."""
    return 1.0 - softmax(logits, axis=1).max(axis=1)
