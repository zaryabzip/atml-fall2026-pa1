import torch.nn.functional as F

from task4.methods.base import CifarMethod


class Vanilla(CifarMethod):
    name = "vanilla"

    def compute_loss(self, x, y):
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"acc": (logits.argmax(1) == y).float().mean().detach()}
