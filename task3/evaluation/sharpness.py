"""Common local sharpness proxy: delta = L(theta + eps) - L(theta), eps = 0.05 * g / ||g||_2."""
import numpy as np
import torch
import torch.nn.functional as F


def fixed_validation_batch(source_val_datasets: dict, per_domain: int = 32, seed: int = 6304):
    """A seeded sample of `per_domain` validation images from each source domain, stacked into one
    (x, y) batch (3 x 32 = 96 images). Returns (x, y, indices) where `indices` maps each domain name
    to the validation-set indices that were used, so the exact batch can be saved and reproduced."""
    rng = np.random.default_rng(seed)
    images, labels, indices = [], [], {}
    for domain, dataset in source_val_datasets.items():
        chosen = sorted(int(i) for i in rng.choice(len(dataset), size=per_domain, replace=False))
        indices[domain] = chosen
        for i in chosen:
            image, label, _ = dataset[i]
            images.append(image)
            labels.append(label)
    return torch.stack(images), torch.tensor(labels), indices


def sharpness_proxy(model, x, y, rho: float = 0.05) -> float:
    """How much the validation cross-entropy rises after ONE normalized gradient-ascent step of
    length `rho` in weight space. Same fixed batch and the same rho for every model. This is a local
    stability diagnostic under that specific perturbation, not a statement about the whole loss
    landscape. The weights are restored exactly afterwards."""
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.detach().clone() for p in params]

    # Loss and gradient at the current weights, in full fp32 (no autocast).
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    grad_norm = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None))

    # Step rho along the normalized gradient, measure the loss there, then put the weights back.
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is not None:
                p.add_(g * (rho / (grad_norm + 1e-12)))
        perturbed_loss = F.cross_entropy(model(x), y)
        for p, original in zip(params, originals):
            p.copy_(original)
    model.zero_grad(set_to_none=True)
    model.train(was_training)  # leave the model in the mode the caller had it in

    return (perturbed_loss - loss.detach()).item()
