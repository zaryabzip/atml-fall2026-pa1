import torch
import torch.nn.functional as F

from shared.pacs_train import PACSMethod
from task2.models.domain_discriminator import DomainDiscriminator
from task2.models.grl import dann_alpha, grad_reverse


class DANN(PACSMethod):
    """Binary domain discriminator on the 512-d feature behind a gradient-reversal layer."""
    name = "dann"
    uses_target = True

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        method_cfg = cfg["method"]
        self.discriminator = DomainDiscriminator(512, method_cfg["disc_hidden"], method_cfg["disc_dropout"])

    def extra_modules(self):
        return self.discriminator

    def compute_loss(self, source, target, progress):
        # How strongly to reverse the discriminator's gradient right now: ramps 0 -> max_grl.
        method_cfg = self.cfg["method"]
        alpha = dann_alpha(progress, method_cfg["max_grl"], method_cfg["grl_gamma"])

        # One forward pass on source+target together, then split the results back apart.
        combined_x = torch.cat([source["x"], target["x"]], dim=0)
        combined_features, combined_logits = self.model(combined_x, return_features=True)
        n_source = source["x"].shape[0]
        n_target = target["x"].shape[0]

        # Classification loss on the labeled source examples only.
        source_logits = combined_logits[:n_source]
        cls_loss = F.cross_entropy(source_logits, source["y"])

        # Domain loss: the discriminator tries to tell source (label 0) from target (label 1),
        # while the gradient-reversal layer trains the backbone to make that job harder.
        reversed_features = grad_reverse(combined_features, alpha)
        if method_cfg.get("disc_input_norm"):
            # Per-example LayerNorm (no learned scale): the discriminator sees each feature at a fixed size, so the
            # backbone can no longer "win" the domain game by just making features bigger.
            reversed_features = F.layer_norm(reversed_features, reversed_features.shape[1:])
        domain_logits = self.discriminator(reversed_features)
        domain_labels = torch.cat([
            torch.zeros(n_source, dtype=torch.long, device=combined_features.device),
            torch.ones(n_target, dtype=torch.long, device=combined_features.device),
        ])
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()

        weight = method_cfg["domain_loss_weight"]
        loss = cls_loss + weight * domain_loss
        return loss, {
            "cls_loss": cls_loss.detach(),
            "domain_loss": domain_loss.detach(),
            "domain_acc": domain_acc.detach(),
            "alpha": torch.tensor(alpha),  # so the schedule is visible in the training log
        }
