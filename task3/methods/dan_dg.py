import itertools

import torch
import torch.nn.functional as F

from shared.mmd import multi_kernel_mmd2  # same function as Task 2 DAN
from shared.pacs import DOMAINS
from shared.pacs_train import PACSMethod


class DANDG(PACSMethod):
    """L = CE(all sources) + (lambda_dg / 3) * sum over pairs (P,A), (P,C), (A,C) of MMD^2. Never sees Sketch."""
    name = "dan_dg"
    uses_target = False

    def compute_loss(self, source, target, progress):
        # One forward pass over the domain-balanced source batch (8 images from each source domain).
        features, logits = self.model(source["x"], return_features=True)
        cls_loss = F.cross_entropy(logits, source["y"])

        # Group the features by which source domain each example came from.
        domain_ids = torch.unique(source["domain"]).tolist()
        features_by_domain = {d: features[source["domain"] == d] for d in domain_ids}

        # MMD^2 for each unordered pair of source domains. Each pair gets its own kernel bandwidths,
        # computed from that pair's combined batch inside multi_kernel_mmd2.
        multipliers = self.cfg["method"]["kernel_multipliers"]
        unbiased = bool(self.cfg["method"].get("mmd_unbiased", True))  # same default as Task 2 DAN
        pair_mmds = {}
        for a, b in itertools.combinations(domain_ids, 2):
            pair_mmds[(a, b)] = multi_kernel_mmd2(features_by_domain[a], features_by_domain[b], multipliers, unbiased)

        # Average the pair penalties, weight by lambda_dg, and add to the classification loss.
        mean_mmd = torch.stack(list(pair_mmds.values())).mean()
        lambda_dg = float(self.cfg["method"]["lambda_dg"])  # float(): an override like 1e-1 is parsed as a string
        loss = cls_loss + lambda_dg * mean_mmd

        # Log the mean MMD (used by the training-curve plot) and each pair's own MMD.
        logs = {"cls_loss": cls_loss.detach(), "mmd": mean_mmd.detach()}
        for (a, b), value in pair_mmds.items():
            logs[f"mmd_{DOMAINS[a]}_{DOMAINS[b]}"] = value.detach()
        return loss, logs
