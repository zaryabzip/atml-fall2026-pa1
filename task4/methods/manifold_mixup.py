import torch


def mix_different_classes(h: torch.Tensor, y: torch.Tensor, beta_param: float, generator=None):
    """Pair each hidden state h_i with h_perm(i) from a random permutation of the batch, keep only pairs
    whose labels differ (y_i != y_j), and mix them with ONE lambda ~ Beta(beta_param, beta_param) for the
    whole batch (as in the PROSER reference code). Returns the mixed states (at most len(h) rows)."""
    perm = torch.randperm(h.shape[0], device=h.device, generator=generator)
    keep = y != y[perm]
    lam = torch.distributions.Beta(beta_param, beta_param).sample().item()
    return lam * h[keep] + (1.0 - lam) * h[perm][keep]
