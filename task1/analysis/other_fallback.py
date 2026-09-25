"""Extra analysis (not required by the handout): what are the "other" cue-conflict answers, and what does each head
answer when there is no object at all?

    python -m task1.analysis.other_fallback

1. For each model, the classes it predicts on cue conflicts that are neither the shape nor the texture class
   (from task1/results/<model>/predictions.json).
2. Content-free inputs, 100 of each (seed 6304): uniform noise, Gaussian noise (mean 0.5, std 0.2), noise at 14x14
   upsampled to 224 (blurred noise), and solid colors. Each model's linear head (retrained exactly as in run_task1)
   classifies them; the most common answer shows the head's default when it finds no object.
Output: task1/results/other_fallback.json.
"""
import collections
import json

import torch
import torch.nn.functional as F

from common.config import load_config
from common.device import get_device
from common.io import PROJECT_ROOT, save_json
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import STL10_CLASSES as C
from task1.models.backbones import load_backbone
from task1.scripts.run_task1 import RESULTS_DIR, cache_split_features, features_from_tensor, head_probabilities


def name(x):
    return C[x] if isinstance(x, int) else x


def main():
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    device = get_device(cfg.get("device", "auto"))
    fb = cfg["feature_batch"]
    g = torch.Generator().manual_seed(cfg["seed"])
    inputs = {"uniform noise": torch.rand(100, 3, 224, 224, generator=g),
              "gaussian noise": (0.5 + 0.2 * torch.randn(100, 3, 224, 224, generator=g)).clamp(0, 1),
              "blurred noise": F.interpolate(torch.rand(100, 3, 14, 14, generator=g), size=224, mode="bilinear"),
              "solid color": torch.rand(100, 3, 1, 1, generator=g).expand(100, 3, 224, 224).clone()}
    res = {}
    for m in cfg["backbones"]:
        p = json.load(open(RESULTS_DIR / m / "predictions.json"))["cue_conflict"]
        other = [name(c) for s, t, c in zip(p["shape"], p["texture"], p["pred"]) if name(c) not in (name(s), name(t))]
        res[m] = {"n_other": len(other), "other_labels": collections.Counter(other).most_common(),
                  "other_shape_to_label": collections.Counter(
                      f"{name(s)}->{name(c)}" for s, t, c in zip(p["shape"], p["texture"], p["pred"])
                      if name(c) not in (name(s), name(t))).most_common(6),
                  "content_free": {}}
        bb = load_backbone(m).to(device)
        ca = cache_split_features(bb, m, device, fb)
        head, _ = train_linear_head(*ca["train"], *ca["val"], cfg["head"], device)
        for k, x in inputs.items():
            pred = head_probabilities(head, features_from_tensor(bb, x, device, fb), device).argmax(1)
            res[m]["content_free"][k] = collections.Counter(C[i] for i in pred).most_common(3)
        print(m, res[m]["n_other"], res[m]["other_labels"][:3], res[m]["content_free"])
        del bb
    save_json(res, RESULTS_DIR / "other_fallback.json")


if __name__ == "__main__":
    main()
