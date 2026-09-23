"""Gradient-reversal layer and the DANN schedule (shared by DANN and CDAN)."""
import math

import torch


class GradReverse(torch.autograd.Function):
    """Forward: identity (does nothing to the values). Backward: multiply the incoming gradient
    by -alpha. This is what makes the backbone get trained to CONFUSE the domain discriminator:
    the discriminator's own gradient, once it reaches the backbone, points the opposite way."""

    @staticmethod
    def forward(ctx, x, alpha):
        # Remember alpha for backward(), and pass the values through completely unchanged.
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Flip the sign and scale by alpha. Returning None for alpha's own gradient: it's a
        # schedule value, not a learned parameter.
        reversed_grad = -ctx.alpha * grad_output
        return reversed_grad, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


def dann_alpha(progress: float, max_alpha: float = 1.0, gamma: float = 10.0) -> float:
    """progress in [0, 1]. Starts at 0 (no reversal, let the network learn to classify first) and
    ramps up smoothly toward max_alpha as training proceeds."""
    return max_alpha * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
