"""CIFAR ResNet-18: 3x3 stride-1 first conv, no max-pool, 32x32 inputs, random init.
Split into two halves so PROSER can apply manifold mixup after layer2."""
import torch
from torch import nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes: int = 10):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def to_layer2(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        return n.layer2(n.layer1(x))

    def from_layer3(self, h):
        n = self.net
        h = n.avgpool(n.layer4(n.layer3(h)))
        features = torch.flatten(h, 1)
        return features, n.fc(features)

    def forward(self, x, return_features: bool = False):
        features, logits = self.from_layer3(self.to_layer2(x))
        return (features, logits) if return_features else logits
