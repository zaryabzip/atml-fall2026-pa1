import torch
import torch.nn.functional as F

from shared.mmd import multi_kernel_mmd2  # use this exact function
from shared.pacs_train import PACSMethod


class DAN(PACSMethod):
    """L = CE(source) + lambda_mmd * MMD^2(source features, target features) on the 512-d feature."""
    name = "dan"
    uses_target = True

    def compute_loss(self, source, target, progress):
        # One forward pass on source+target together, then split the results back apart.
        combined_x = torch.cat([source["x"], target["x"]], dim=0)
        combined_features, combined_logits = self.model(combined_x, return_features=True)

        n_source = source["x"].shape[0]
        source_logits = combined_logits[:n_source]
        source_features = combined_features[:n_source]
        target_features = combined_features[n_source:]

        # Classification loss on the labeled source examples only, plus the MMD alignment term
        # pulling the source and target feature distributions closer together.
        cls_loss = F.cross_entropy(source_logits, source["y"])
        lambda_mmd = float(self.cfg["method"]["lambda_mmd"])  # float(): an override like 1e-1 is parsed as a string
        kernel_multipliers = self.cfg["method"]["kernel_multipliers"]
        unbiased = bool(self.cfg["method"].get("mmd_unbiased", True))  # same default as Task 3 DAN-DG
        mmd = multi_kernel_mmd2(source_features, target_features, kernel_multipliers, unbiased)
        loss = cls_loss + lambda_mmd * mmd

        return loss, {"cls_loss": cls_loss.detach(), "mmd": mmd.detach()}
