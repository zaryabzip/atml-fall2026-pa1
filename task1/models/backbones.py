"""Frozen Task 1 backbones. Input: common 224x224 RGB tensors in [0, 1] (no normalization).
Each wrapper applies its own normalization and returns the required representation:
  resnet50 -> 2048-d global-average-pooled feature
  vit_b16  -> 768-d final class token
  clip_b32 -> 512-d L2-normalized image embedding
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

BACKBONES = ("resnet50", "vit_b16", "clip_b32")


class FrozenBackbone(nn.Module):
    def __init__(self, name, encoder, feature_dim, mean, std):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # always frozen

    @torch.no_grad()
    def forward(self, x01):
        return self.encoder((x01 - self.mean) / self.std)


class _ClipImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x):
        return F.normalize(self.clip.encode_image(x), dim=-1)


def load_backbone(name: str) -> FrozenBackbone:
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=weights)
        net.fc = nn.Identity()
        t = weights.transforms()
        return FrozenBackbone(name, net, 2048, t.mean, t.std)
    if name == "vit_b16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=weights)
        net.heads = nn.Identity()
        t = weights.transforms()
        return FrozenBackbone(name, net, 768, t.mean, t.std)
    if name == "clip_b32":
        import open_clip

        # "ViT-B-32-quickgelu", not "ViT-B-32": the OpenAI-released weights use the QuickGELU
        # activation. Loading them into the plain "ViT-B-32" config (regular GELU) silently mismatches
        # the activation function against a frozen pretrained model and gives a QuickGELU warning.
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        return FrozenBackbone(name, _ClipImageEncoder(model), 512,
                              open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD)
    raise ValueError(f"Unknown backbone {name}; choose from {BACKBONES}")


@torch.no_grad()
def extract_features(backbone: FrozenBackbone, loader, device) -> tuple:
    """Returns (features, labels) as numpy arrays. Loader yields (x01, label, ...)."""
    feats, labels = [], []
    for batch in loader:
        feats.append(backbone(batch[0].to(device, non_blocking=True)).float().cpu())
        labels.append(torch.as_tensor(batch[1]))
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def clip_zero_shot_text(backbone: FrozenBackbone, class_names, template: str, device) -> tuple:
    """Returns (normalized text embeddings [C, 512], logit_scale). Zero-shot logits = scale * img @ text.T."""
    import open_clip

    clip = backbone.encoder.clip
    tokens = open_clip.get_tokenizer("ViT-B-32")([template.format(c) for c in class_names]).to(device)
    text = F.normalize(clip.encode_text(tokens), dim=-1)
    return text, clip.logit_scale.exp()
