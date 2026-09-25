"""Extra analysis (not required by the handout): why do features move a lot (low I_T) while predictions barely change?

    python -m task1.analysis.head_readout

For grayscale, hue and the 4x4 shuffle, per backbone (frozen features, linear head retrained exactly as in run_task1):
- share of each image's feature change that lies in the 9 directions that decide the head's winner (the span of the
  class-weight differences), vs. 9/d for a random direction;
- the clean logit margin (winner minus runner-up) and how much of it survives the change;
- clean margins of the images whose prediction flips vs. those that don't.
Images the transform leaves unchanged (already-gray images under hue) are skipped for the share.
Output: task1/results/head_readout.json.
"""
import json
import numpy as np, torch
from common.config import load_config
from common.device import get_device
from common.io import PROJECT_ROOT, load_json
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import SPLIT_FILE
from task1.data.transforms import to_grayscale, extra_color, patch_shuffle
from task1.models.backbones import load_backbone
from task1.scripts.run_task1 import cache_split_features, features_from_tensor, load_split_tensors
cfg = load_config(PROJECT_ROOT / "task1/configs/default.yaml"); dev = get_device("auto"); fb = cfg["feature_batch"]
x, y = load_split_tensors("test", load_json(SPLIT_FILE)["eval_subset"], fb); y = np.asarray(y)
T = {"grayscale": to_grayscale(x), "hue": extra_color(x, hue_factor=0.3),
     "shuffle": patch_shuffle(x, 4, torch.Generator().manual_seed(cfg["seed"]))[0]}
def margin(l, cls):
    top = l[np.arange(len(l)), cls]; o = l.copy(); o[np.arange(len(l)), cls] = -np.inf
    return top - o.max(1)
out = {}
for m in ("resnet50", "vit_b16", "clip_b32"):
    bb = load_backbone(m).to(dev); ca = cache_split_features(bb, m, dev, fb)
    head, _ = train_linear_head(*ca["train"], *ca["val"], cfg["head"], dev)
    W = head.weight.detach().cpu().numpy(); b = head.bias.detach().cpu().numpy()
    D = (W[:, None, :] - W[None, :, :]).reshape(-1, W.shape[1])   # class-difference directions: what decides the winner
    Q = np.linalg.svd(D, full_matrices=False)[2][:9].T           # 9-d subspace
    f0 = features_from_tensor(bb, x, dev, fb); l0 = f0 @ W.T + b; c0 = l0.argmax(1)
    m0 = margin(l0, c0)
    for k, xt in T.items():
        f1 = features_from_tensor(bb, xt, dev, fb); d = f1 - f0
        nz = (d ** 2).sum(1) > 1e-12; share = (((d[nz] @ Q) ** 2).sum(1) / (d[nz] ** 2).sum(1)).mean()
        m1 = margin(f1 @ W.T + b, c0)
        flipped = m1 < 0
        out.setdefault(m, {})[k] = {"share_read": float(share), "random_share": 9 / W.shape[1], "clean_margin_median": float(np.median(m0)), "margin_kept_median": float(np.median(m1 / m0)), "flipped_clean_margin_median": float(np.median(m0[flipped])), "kept_clean_margin_median": float(np.median(m0[~flipped]))}
        print(f"{m:9s} {k:9s} head reads {100*share:4.1f}% of the feature change (random {100*9/W.shape[1]:.1f}%) | "
              f"clean margin median {np.median(m0):.2f}; kept {np.median(m1/m0)*100:.0f}% of it (median) | "
              f"flipped: clean margin {np.median(m0[flipped]):.2f} vs kept {np.median(m0[~flipped]):.2f}")
    del bb

from task1.scripts.run_task1 import RESULTS_DIR
json.dump(out, open(RESULTS_DIR / "head_readout.json", "w"), indent=1)
