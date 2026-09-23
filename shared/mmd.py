"""Multi-kernel MMD. Task 2 (DAN) and Task 3 (DAN-DG) must use this exact same function."""
import torch


def multi_kernel_mmd2(x: torch.Tensor, y: torch.Tensor, multipliers=(0.5, 1.0, 2.0), unbiased: bool = True) -> torch.Tensor:
    """Squared MMD between feature batches x (n, 512) and y (m, 512) with a sum of RBF kernels.

    Design choices, since the PDF leaves them open:
      - the kernel bandwidth (the median pairwise squared distance) is DETACHED: it just picks a
        sensible length scale for this batch, it isn't something we want gradients flowing through.
      - the UNBIASED estimator is the default: the diagonal (self-similarity) terms of K_xx and K_yy are
        dropped, so two batches from the same distribution score about 0 (a single value can be slightly
        negative). The biased estimator (`unbiased=False`) keeps the diagonal; for two batches from the SAME
        distribution its value is about 2(3 - k)/n for n samples per side, where k is the mean off-diagonal
        kernel similarity. With small batches (Task 3 has 8 per domain) minimizing it rewards raising every
        pairwise similarity, i.e. collapsing the features. That is what happened with the biased estimator:
        DAN-DG at lambda_DG = 1 collapsed to constant features on SOURCE validation, so both tasks were
        switched to the unbiased estimator before any Task 3 Sketch evaluation.
    """
    # Always compute in float32, even under fp16 autocast: exp() is numerically unstable in fp16.
    x = x.float()
    y = y.float()

    # Stack x and y together and compute every pair's squared distance in one go.
    combined = torch.cat([x, y], dim=0)  # (n + m, 512)
    n_total = combined.shape[0]
    pairwise_sq_dist = torch.cdist(combined, combined) ** 2  # (n+m, n+m)

    # Pick a bandwidth scale from the "typical" squared distance in this batch (the median,
    # excluding the always-zero diagonal), clamped away from zero for degenerate batches.
    off_diagonal = ~torch.eye(n_total, dtype=torch.bool, device=combined.device)
    median_sq_dist = pairwise_sq_dist[off_diagonal].median().detach()
    median_sq_dist = median_sq_dist.clamp(min=1e-8)

    # Sum three RBF kernels, one per bandwidth multiplier (e.g. 0.5x, 1x, 2x the median distance).
    kernel_sum = torch.zeros_like(pairwise_sq_dist)
    for multiplier in multipliers:
        bandwidth = multiplier * median_sq_dist
        kernel_sum = kernel_sum + torch.exp(-pairwise_sq_dist / bandwidth)

    # Split the combined kernel matrix into its x-x, y-y, and x-y blocks, then combine into MMD^2.
    n_x = x.shape[0]
    k_xx = kernel_sum[:n_x, :n_x]
    k_yy = kernel_sum[n_x:, n_x:]
    k_xy = kernel_sum[:n_x, n_x:]
    if unbiased:
        # Drop the diagonal (self-similarity) terms of K_xx and K_yy and average over the n(n-1) / m(m-1) pairs.
        n_y = y.shape[0]
        term_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n_x * (n_x - 1))
        term_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (n_y * (n_y - 1))
        return term_xx + term_yy - 2 * k_xy.mean()

    mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd2
