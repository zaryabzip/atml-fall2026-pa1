import numpy as np


def proser_score(logits_with_dummies: np.ndarray, num_known: int = 10) -> np.ndarray:
    """PROSER placeholder score (Zhou et al. 2021, Sec. 4.3): u = max dummy logit - max known logit.

    The paper appends the strongest dummy logit to the known logits, adds a bias chosen so that 95% of the
    validation set is still predicted as known, and calls x unknown when the biased dummy logit wins. That is a
    threshold on (max dummy - max known), so the bias is the same thing as our threshold at the 95th percentile
    of u on CIFAR-10 val. The reference code's score, softmax([known, max dummy] / 1024) dummy prob minus the top
    known prob, ranks inputs (almost exactly) like this logit difference. Larger = more novel."""
    known = logits_with_dummies[:, :num_known]
    dummies = logits_with_dummies[:, num_known:]
    return dummies.max(axis=1) - known.max(axis=1)
