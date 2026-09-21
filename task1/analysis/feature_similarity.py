"""Representation stability I_T = mean cosine(f(x), f(T(x)))."""
import numpy as np


def cosine_stability(clean_feats, transformed_feats) -> float:
    """Rows are paired (same image). Mean of row-wise cosine similarity."""
    clean = np.asarray(clean_feats, dtype=np.float64)
    transformed = np.asarray(transformed_feats, dtype=np.float64)

    # Per-image cosine similarity: dot product divided by the product of the two vectors' lengths,
    # clipped away from zero so a degenerate all-zero row can't cause a divide-by-zero.
    dot_products = (clean * transformed).sum(axis=1)
    denominators = np.linalg.norm(clean, axis=1) * np.linalg.norm(transformed, axis=1)
    denominators = np.clip(denominators, 1e-12, None)
    cosine_per_row = dot_products / denominators

    return float(cosine_per_row.mean())
