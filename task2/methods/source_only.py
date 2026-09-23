import torch.nn.functional as F

from shared.pacs_train import PACSMethod


class SourceOnly(PACSMethod):
    """Plain cross-entropy on domain-balanced source batches. Also the Task 3 ERM baseline."""
    name = "source_only"

    def compute_loss(self, source, target, progress):
        logits = self.model(source["x"])
        loss = F.cross_entropy(logits, source["y"])
        return loss, {"cls_loss": loss.detach()}
