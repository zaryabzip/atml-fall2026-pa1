"""Extra experiment (not required by the handout): grid-alignment test without any wrap-around.

    python -m task1.analysis.shuffle_offset_nowrap

Moving the cut lines by rolling the image would split the tiles that cross the image edge into two strips on
opposite sides. Here every tile is a full, equal square: only a central region is shuffled, and that
region is placed either on or between the model's patch borders. Both versions shuffle the same number of equal
tiles with the same permutations and leave the same border area (224^2 - region^2) unshuffled.
  16 px tiles: 13x13 tiles in a 208 px region, starting at 16 px (cuts on ViT-B/16 borders) or at 8 px (cuts mid-patch)
  32 px tiles: 6x6 tiles in a 192 px region, starting at 32 px (cuts on CLIP borders) or at 16 px (cuts mid-patch)
Output: task1/results/shuffle_offset_nowrap.json.
"""
import numpy as np
import torch

from common.config import load_config
from common.device import get_device
from common.io import PROJECT_ROOT, load_json, save_json
from task1.analysis.feature_similarity import cosine_stability
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import SPLIT_FILE
from task1.data.transforms import patch_shuffle
from task1.models.backbones import load_backbone
from task1.scripts.run_task1 import (RESULTS_DIR, cache_split_features, features_from_tensor, head_probabilities,
                                     load_split_tensors)

CASES = {"16px aligned": (16, 13, 16), "16px mid-patch": (16, 13, 8),
         "32px aligned": (32, 6, 32), "32px mid-patch": (32, 6, 16)}  # name: (tile px, tiles per side, start px)


def shuffle_region(x, tile, n, start, seed):
    out = x.clone()
    end = start + tile * n
    out[:, :, start:end, start:end] = patch_shuffle(x[:, :, start:end, start:end], n,
                                                    torch.Generator().manual_seed(seed))[0]
    return out


def main():
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    device = get_device(cfg.get("device", "auto"))
    fb = cfg["feature_batch"]
    x, y = load_split_tensors("test", load_json(SPLIT_FILE)["eval_subset"], fb)
    y = np.asarray(y)
    inputs = {k: shuffle_region(x, t, n, s, cfg["seed"]) for k, (t, n, s) in CASES.items()}
    res = {}
    for name in cfg["backbones"]:
        bb = load_backbone(name).to(device)
        ca = cache_split_features(bb, name, device, fb)
        head, _ = train_linear_head(*ca["train"], *ca["val"], cfg["head"], device)
        f0 = features_from_tensor(bb, x, device, fb)
        p0 = head_probabilities(head, f0, device).argmax(1)
        res[name] = {}
        for k, xt in inputs.items():
            f = features_from_tensor(bb, xt, device, fb)
            p = head_probabilities(head, f, device).argmax(1)
            res[name][k] = {"accuracy": float((p == y).mean()), "consistency": float((p == p0).mean()),
                            "cosine_stability": cosine_stability(f0, f)}
            print(name, k, {a: round(b, 3) for a, b in res[name][k].items()})
        del bb
    save_json(res, RESULTS_DIR / "shuffle_offset_nowrap.json")


if __name__ == "__main__":
    main()
