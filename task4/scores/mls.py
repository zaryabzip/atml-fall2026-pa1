import numpy as np


def mls(logits: np.ndarray) -> np.ndarray:
    """u = -max_k z_k. For PROSER pass only the 10 known-class logits."""
    return -logits.max(axis=1)
