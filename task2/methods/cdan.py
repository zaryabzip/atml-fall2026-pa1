import torch
import torch.nn.functional as F

from shared.pacs import NUM_CLASSES
from shared.pacs_train import PACSMethod
from task2.models.domain_discriminator import DomainDiscriminator
from task2.models.grl import dann_alpha, grad_reverse


class CDAN(PACSMethod):
    """Like DANN, but the discriminator sees g = vec(f outer p): 512 * 7 = 3584 inputs."""
    name = "cdan"
    uses_target = True

    def __init__(self, cfg, model, device):
        super().__init__(cfg, model, device)
        method_cfg = cfg["method"]
        in_dim = 512 * NUM_CLASSES  # the flattened outer product's size
        self.discriminator = DomainDiscriminator(in_dim, method_cfg["disc_hidden"], method_cfg["disc_dropout"])

    def extra_modules(self):
        return self.discriminator

    def compute_loss(self, source, target, progress):
        # How strongly to reverse the discriminator's gradient right now: same schedule as DANN.
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

        # Class-conditional domain loss: the discriminator sees feature x predicted-class-probs
        # (not detached, per the PDF), so it can align "source dogs" with "target dogs" instead of
        # just aligning the two domains overall.
        probs = F.softmax(combined_logits, dim=1)  # (N, 7)
        joint = torch.bmm(probs.unsqueeze(2), combined_features.unsqueeze(1)).flatten(1)  # (N, 3584)
        reversed_joint = grad_reverse(joint, alpha)
        if method_cfg.get("disc_input_norm"):
            # Same fix as DANN: fixed-size discriminator input, so the backbone can't win by inflating features.
            reversed_joint = F.layer_norm(reversed_joint, reversed_joint.shape[1:])
        domain_logits = self.discriminator(reversed_joint)
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
            "alpha": torch.tensor(alpha),
        }
