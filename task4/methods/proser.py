import torch
import torch.nn.functional as F
from torch import nn

from common.io import checkpoint_dir
from task4.methods.base import CifarMethod
from task4.methods.manifold_mixup import mix_different_classes
from task4.models.resnet_cifar import CifarResNet18

NUM_KNOWN = 10


def with_placeholder(logits: torch.Tensor) -> torch.Tensor:
    """[10 known logits, max over dummy logits]: the K+1-way output h_hat of the paper."""
    return torch.cat([logits[:, :NUM_KNOWN], logits[:, NUM_KNOWN:].max(dim=1, keepdim=True).values], dim=1)


class PROSER(CifarMethod):
    """Classifier placeholders (beta = 1) + data placeholders via manifold mixup after layer2 (gamma = 0.1).
    Zhou et al. (2021). No CIFAR-100 image is ever used.

    Each batch is split in two halves:
      first half:  CE(h_hat(x), y) + beta * CE(h_hat(x) with the true class masked out, K+1)
      second half: gamma * CE(h_hat(mixup of two different classes after layer2), K+1)
    K+1 is the placeholder class, whose logit is the strongest of the 5 dummy logits."""
    name = "proser"

    @classmethod
    def build_model(cls, cfg, device):
        """Vanilla checkpoint, fc widened from 10 to 10 + num_dummy outputs. Known rows are copied;
        dummy rows keep PyTorch's default random init. Known-class logits stay columns 0..9."""
        ckpt_path = checkpoint_dir("task4", cfg["method"]["init_from"]) / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CifarResNet18(num_classes=NUM_KNOWN)
        model.load_state_dict(ckpt["model"])
        old_fc = model.net.fc
        new_fc = nn.Linear(old_fc.in_features, NUM_KNOWN + int(cfg["method"]["num_dummy"]))
        with torch.no_grad():
            new_fc.weight[:NUM_KNOWN].copy_(old_fc.weight)
            new_fc.bias[:NUM_KNOWN].copy_(old_fc.bias)
        model.net.fc = new_fc
        return model

    def compute_loss(self, x, y):
        m = self.cfg["method"]
        net = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        half = x.shape[0] // 2
        x1, y1, x2, y2 = x[:half], y[:half], x[half:], y[half:]
        placeholder = torch.full_like(y1, NUM_KNOWN)

        # First half: normal classification, then the classifier-placeholder loss with the true class masked.
        out = with_placeholder(net(x1)).float()
        cls_loss = F.cross_entropy(out, y1)
        masked = out.scatter(1, y1[:, None], -1e9)
        placeholder_loss = F.cross_entropy(masked, placeholder)

        # Second half: mix layer2 states of two different classes; the mix should go to the placeholder class.
        h_mix = mix_different_classes(net.to_layer2(x2), y2, float(m["mixup_beta"]))
        _, mix_logits = net.from_layer3(h_mix)
        mix_out = with_placeholder(mix_logits).float()
        data_loss = F.cross_entropy(mix_out, torch.full((mix_out.shape[0],), NUM_KNOWN, device=x.device))

        loss = cls_loss + float(m["beta"]) * placeholder_loss + float(m["gamma"]) * data_loss
        return loss, {"cls_loss": cls_loss.detach(), "placeholder_loss": placeholder_loss.detach(),
                      "data_loss": data_loss.detach(),
                      "acc": (out[:, :NUM_KNOWN].argmax(1) == y1).float().mean().detach()}
