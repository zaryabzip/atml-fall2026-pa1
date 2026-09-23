"""ResNet-18 + 7-class head used by every Task 2 and Task 3 method (so Task 3 can load the Task 2
Source-only checkpoint unchanged)."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from shared.pacs import NUM_CLASSES


class PACSResNet18(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()  # outputs the 512-d pooled feature
        self.backbone = backbone
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        features = self.backbone(x)
        logits = self.head(features)
        return (features, logits) if return_features else logits


def load_pacs_checkpoint(path, device) -> tuple:
    """Returns (model, checkpoint_dict). Extra-module state (e.g. a discriminator) stays in the dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PACSResNet18(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
