"""Shape/texture cue-conflict images via AdaIN.

    python -m task1.data.make_cue_conflicts

Uses the pretrained encoder/decoder from Huang & Belongie (2017), "Arbitrary Style Transfer in
Real-time with Adaptive Instance Normalization", via the unofficial PyTorch port by naoto0804:
https://github.com/naoto0804/pytorch-AdaIN (weights: same repo's GitHub Release v0.0.0). The
network architecture and the AdaIN math below are copied from that repository; attributed again
in task1/README.md. Weights are auto-downloaded on first run into task1/cache/adain/ (gitignored).

Rejection rule (fixed here, before any model ever sees these images; never uses a model's
predictions — see "What to Watch For" in the assignment). AdaIN only swaps per-channel feature
statistics; it does NOT move the deep feature map's spatial layout, so it preserves the source
photo's coarse shape/layout by construction. A rejected image is therefore one where the DECODER
itself broke, not one whose predicted class we happen to dislike:
  - the raw (pre-clamp) output contains a NaN/Inf value,
  - too much of the raw output falls outside [0, 1] (decoder numerically blew up),
  - the finished image is almost flat/blank (decoder collapsed to a near-constant image), or
  - the finished image's coarse (16x16, average-pooled) layout barely correlates with the
    original content photo's coarse layout (the reconstruction lost the photo's spatial layout).
Thresholds below were picked by measuring these exact quantities on real STL-10 content/style
pairs across alpha 0.0-1.0 (all scored 0.76-0.90) against unrelated-image negative controls
(scored -0.24 to -0.08); 0.3 sits well below every real AdaIN output we measured.
"""
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from common.config import load_config
from common.io import PROJECT_ROOT, save_json
from task1.data.stl10 import STL10_CLASSES, common_transform, load_stl10

ADAIN_DIR = PROJECT_ROOT / "task1" / "cache" / "adain"  # where the pretrained weights are cached
DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"

OUT_DIR = PROJECT_ROOT / "task1" / "data" / "cue_conflicts"  # accepted images + metadata go here
IMAGES_DIR = OUT_DIR / "images"
METADATA_PATH = OUT_DIR / "metadata.json"

CANDIDATES_PER_DIRECTION = 30  # content/style samples tried per (content_class, style_class) direction

MAX_CLIPPED_FRACTION = 0.5  # reject if over half the raw pixels fall outside [0, 1]: decoder blew up
MIN_OUTPUT_STD = 0.02  # reject if the finished image is almost perfectly flat/blank
MIN_COARSE_LAYOUT_SIMILARITY = 0.3  # reject if the coarse photo layout wasn't reconstructed at all


# ---------------------------------------------------------------------------
# The AdaIN network itself (architecture copied from naoto0804/pytorch-AdaIN's net.py)
# ---------------------------------------------------------------------------

def _build_decoder() -> nn.Module:
    """The decoder that turns a stylized relu4_1 feature map back into an RGB image."""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


def _build_full_vgg() -> nn.Module:
    """The full VGG-19-shaped encoder that vgg_normalised.pth was trained for. We only use its
    first 31 layers (up to relu4_1) once the weights are loaded; see `load_adain_models`.
    Layers are labeled by which VGG block they belong to; everything past relu4_1 is unused but
    has to stay here so the state_dict shapes match the pretrained weights file."""
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),  # a learned "normalization" layer baked into these weights
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),  # relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),  # relu1_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),  # relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),  # relu2_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),  # relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),  # relu3_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # relu4_1 <- we stop here
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4_2 (unused)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4_3 (unused)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu4_4 (unused)
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_1 (unused)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_2 (unused)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_3 (unused)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_4 (unused)
    )


def _ensure_adain_weights() -> None:
    """Download decoder.pth / vgg_normalised.pth once, if they aren't already cached."""
    ADAIN_DIR.mkdir(parents=True, exist_ok=True)

    decoder_path = ADAIN_DIR / "decoder.pth"
    if not decoder_path.exists():
        print(f"Downloading AdaIN decoder weights to {decoder_path}")
        urllib.request.urlretrieve(DECODER_URL, decoder_path)

    vgg_path = ADAIN_DIR / "vgg_normalised.pth"
    if not vgg_path.exists():
        print(f"Downloading AdaIN VGG encoder weights to {vgg_path}")
        urllib.request.urlretrieve(VGG_URL, vgg_path)


def load_adain_models(device):
    """Returns (encoder, decoder), both frozen and in eval mode, ready for stylize()."""
    _ensure_adain_weights()

    # Build both networks and load the pretrained weights into them.
    decoder = _build_decoder()
    decoder.load_state_dict(torch.load(ADAIN_DIR / "decoder.pth", map_location="cpu"))
    full_vgg = _build_full_vgg()
    full_vgg.load_state_dict(torch.load(ADAIN_DIR / "vgg_normalised.pth", map_location="cpu"))
    encoder = nn.Sequential(*list(full_vgg.children())[:31])  # trim down to relu4_1, per the paper

    # Move to device, freeze, and set to eval mode: we only ever run this forward.
    encoder, decoder = encoder.to(device), decoder.to(device)
    encoder.eval()
    decoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    return encoder, decoder


def _feature_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> tuple:
    """Per-image, per-channel mean and standard deviation of a (N, C, H, W) feature map.
    `eps` is added to the variance before the square root: some channels are exactly constant
    (dead ReLU channels are common), which gives a variance of exactly 0 and, without eps, a
    divide-by-zero (NaN) a few steps later in adaptive_instance_normalization."""
    n, c = feat.shape[0], feat.shape[1]
    flattened = feat.view(n, c, -1)
    mean = flattened.mean(dim=2).view(n, c, 1, 1)
    std = (flattened.var(dim=2) + eps).sqrt().view(n, c, 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Re-centers/re-scales the content feature map to have the style feature map's per-channel
    mean and std, while leaving the content feature map's spatial layout untouched."""
    content_mean, content_std = _feature_mean_std(content_feat)
    style_mean, style_std = _feature_mean_std(style_feat)
    normalized = (content_feat - content_mean) / content_std  # zero-mean, unit-std, same layout
    return normalized * style_std + style_mean  # re-painted with the style's statistics


def stylize(encoder, decoder, content: torch.Tensor, style: torch.Tensor, alpha: float) -> torch.Tensor:
    """One AdaIN forward pass. Returns the RAW decoder output (not yet clamped to [0, 1])."""
    with torch.no_grad():
        # Encode both images, swap the content's feature statistics for the style's, blend by
        # alpha (alpha=1 -> full style), and decode back into an image.
        content_feat = encoder(content)  # (1, 512, 28, 28) relu4_1 feature
        style_feat = encoder(style)
        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
        blended_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        return decoder(blended_feat)  # (1, 3, 224, 224), values may fall slightly outside [0, 1]


# ---------------------------------------------------------------------------
# The rejection rule
# ---------------------------------------------------------------------------

def coarse_layout_similarity(content_img: torch.Tensor, stylized_img: torch.Tensor, size: int = 16) -> float:
    """Average-pool both images down to a small (size x size) grayscale grid, which keeps the
    coarse brightness layout of the photo but washes out fine texture. Returns the correlation
    between the two grids: how much the original photo's layout survived the stylization."""
    # Reduce both images to a small grayscale grid, ignoring fine texture.
    content_small = F.adaptive_avg_pool2d(content_img.mean(dim=1, keepdim=True), size).flatten()
    stylized_small = F.adaptive_avg_pool2d(stylized_img.mean(dim=1, keepdim=True), size).flatten()

    # Correlate the two grids (mean-centered, then normalized dot product).
    content_centered = content_small - content_small.mean()
    stylized_centered = stylized_small - stylized_small.mean()
    denominator = (content_centered.norm() * stylized_centered.norm()).clamp(min=1e-8)
    return ((content_centered @ stylized_centered) / denominator).item()


def check_stylization(content_img: torch.Tensor, raw_output: torch.Tensor) -> dict:
    """Runs the fixed rejection rule on one stylization. Returns a dict with every measured
    quantity plus the final "accepted" flag; never looks at a classifier's prediction."""
    # Measure the 3 failure signals: non-finite values, values wildly outside [0, 1], and a
    # collapsed (flat) output, plus whether the photo's coarse layout survived.
    is_finite = torch.isfinite(raw_output).all().item()
    clipped_fraction = ((raw_output < 0) | (raw_output > 1)).float().mean().item()
    clamped_output = raw_output.clamp(0, 1)  # the actual image we'd save and evaluate
    output_std = clamped_output.std().item()
    layout_similarity = coarse_layout_similarity(content_img, clamped_output)

    accepted = (
        is_finite
        and clipped_fraction <= MAX_CLIPPED_FRACTION
        and output_std >= MIN_OUTPUT_STD
        and layout_similarity >= MIN_COARSE_LAYOUT_SIMILARITY
    )

    return {
        "is_finite": is_finite,
        "clipped_fraction": clipped_fraction,
        "output_std": output_std,
        "coarse_layout_similarity": layout_similarity,
        "accepted": accepted,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _class_index_lookup(class_names, all_labels) -> dict:
    """class name -> list of STL-10 test-split indices belonging to that class."""
    lookup = {}
    for class_name in class_names:
        class_idx = STL10_CLASSES.index(class_name)
        lookup[class_name] = [i for i, label in enumerate(all_labels) if label == class_idx]
    return lookup


def _directions_from_pairs(class_pairs) -> list:
    """Every unordered pair, expanded into both (content, style) directions."""
    directions = []
    for class_a, class_b in class_pairs:
        directions.append((class_a, class_b))  # shape/content = A, texture/style = B
        directions.append((class_b, class_a))  # shape/content = B, texture/style = A
    return directions


def main():
    # Load the design-choice settings from configs/default.yaml.
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    class_pairs = cfg["cue_conflict"]["class_pairs"]
    alpha = cfg["cue_conflict"]["style_strength"]
    min_valid = cfg["cue_conflict"]["min_valid"]
    seed = cfg["seed"]
    if len(class_pairs) < 5:
        raise ValueError(f"Need >= 5 class pairs in configs/default.yaml, got {len(class_pairs)}.")

    # Load the frozen AdaIN network and the STL-10 test split we'll draw content/style images from.
    device = torch.device("cpu")  # this network is small; CPU is fine and keeps this script simple
    encoder, decoder = load_adain_models(device)
    dataset = load_stl10("test", transform=common_transform())
    class_to_indices = _class_index_lookup(STL10_CLASSES, dataset.labels)
    rng = np.random.default_rng(seed)  # deterministic choice of which source images to try

    # For every (content_class, style_class) direction, try CANDIDATES_PER_DIRECTION random source
    # images, stylize each one, run the rejection rule, and save only the accepted images to disk.
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []  # one record per candidate, accepted or not
    accepted_count, rejected_count = 0, 0
    for content_class, style_class in _directions_from_pairs(class_pairs):
        content_pool = class_to_indices[content_class]
        style_pool = class_to_indices[style_class]
        content_ids = rng.choice(content_pool, size=CANDIDATES_PER_DIRECTION, replace=False)
        style_ids = rng.choice(style_pool, size=CANDIDATES_PER_DIRECTION, replace=True)

        for candidate_i in range(CANDIDATES_PER_DIRECTION):
            content_id, style_id = int(content_ids[candidate_i]), int(style_ids[candidate_i])
            content_img, _ = dataset[content_id]
            style_img, _ = dataset[style_id]
            content_img, style_img = content_img.unsqueeze(0), style_img.unsqueeze(0)  # add batch dim

            raw_output = stylize(encoder, decoder, content_img, style_img, alpha)
            check = check_stylization(content_img, raw_output)  # the rejection rule, run here

            file_name = f"{content_class}_shape_{style_class}_texture_{content_id}_{style_id}.jpg"
            metadata.append({
                "path": file_name,
                "content_class": content_class,
                "style_class": style_class,
                "source_image_id": content_id,  # the un-stylized content image's STL-10 test index
                "style_source_image_id": style_id,
                "accepted": check["accepted"],
                "is_finite": check["is_finite"],
                "clipped_fraction": check["clipped_fraction"],
                "output_std": check["output_std"],
                "coarse_layout_similarity": check["coarse_layout_similarity"],
            })

            if check["accepted"]:
                clamped_output = raw_output.clamp(0, 1).squeeze(0)  # (3, 224, 224), back to [0, 1]
                image_array = (clamped_output.permute(1, 2, 0).numpy() * 255).astype("uint8")
                Image.fromarray(image_array).save(IMAGES_DIR / file_name, quality=90)
                accepted_count += 1
            else:
                rejected_count += 1

    # Save the metadata for every candidate (accepted or not), and report the final counts.
    save_json(metadata, METADATA_PATH)
    print(f"Accepted: {accepted_count}   Rejected: {rejected_count}   Total tried: {len(metadata)}")
    print(f"Saved images to {IMAGES_DIR}")
    print(f"Saved metadata to {METADATA_PATH}")
    if accepted_count < min_valid:
        print(f"WARNING: only {accepted_count} accepted, need >= {min_valid}. "
              f"Raise CANDIDATES_PER_DIRECTION in this script and re-run.")


if __name__ == "__main__":
    main()
