"""CLIP zero-shot on the translation test (the main run compared zero-shot with the head only on grayscale, hue,
patch shuffle and cue conflicts).

    python -m task1.analysis.zero_shot_translation

Same images, shifts (8/16/32 px), four directions and reflection padding as run_task1.run_translation_curve.
Consistency is against zero-shot's own clean prediction; agreement is with the trained head on the same shifted image.
The head curve is recomputed as a check that it matches task1/results/summary.json.
Output: task1/results/clip_zero_shot_translation.json.
"""
import numpy as np
import torch
import torch.nn.functional as F

from common.config import load_config
from common.device import get_device
from common.io import PROJECT_ROOT, load_json, save_json
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import SPLIT_FILE, STL10_CLASSES
from task1.data.transforms import translate
from task1.models.backbones import clip_zero_shot_text, load_backbone
from task1.scripts.run_task1 import (RESULTS_DIR, cache_split_features, features_from_tensor, head_probabilities,
                                     load_split_tensors)


def main():
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    device = get_device(cfg.get("device", "auto"))
    fb = cfg["feature_batch"]
    eval_x, eval_y = load_split_tensors("test", load_json(SPLIT_FILE)["eval_subset"], fb)
    eval_y = np.asarray(eval_y)
    backbone = load_backbone("clip_b32").to(device)
    cached = cache_split_features(backbone, "clip_b32", device, fb)
    head, _ = train_linear_head(*cached["train"], *cached["val"], cfg["head"], device)
    text, scale = clip_zero_shot_text(backbone, STL10_CLASSES, cfg["clip_zero_shot"]["template"], device)

    def zero_shot(feats):
        logits = scale.item() * (torch.from_numpy(feats).to(device) @ text.T.to(device))
        return F.softmax(logits, dim=1).cpu().numpy().argmax(1)

    clean = features_from_tensor(backbone, eval_x, device, fb)
    zs_clean, head_clean = zero_shot(clean), head_probabilities(head, clean, device).argmax(1)
    curve = [{"shift_px": 0, "zs_accuracy": float((zs_clean == eval_y).mean()), "zs_consistency": 1.0,
              "head_accuracy": float((head_clean == eval_y).mean()), "head_consistency": 1.0,
              "agreement_with_head": float((zs_clean == head_clean).mean())}]
    for shift in cfg["interventions"]["translations"]:
        if shift == 0:
            continue
        rows = []
        for dx, dy in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
            feats = features_from_tensor(backbone, translate(eval_x, dx, dy), device, fb)
            zs, hd = zero_shot(feats), head_probabilities(head, feats, device).argmax(1)
            rows.append(((zs == eval_y).mean(), (zs == zs_clean).mean(), (hd == eval_y).mean(),
                         (hd == head_clean).mean(), (zs == hd).mean()))
        m = np.mean(rows, axis=0)
        curve.append({"shift_px": shift, "zs_accuracy": float(m[0]), "zs_consistency": float(m[1]),
                      "head_accuracy": float(m[2]), "head_consistency": float(m[3]), "agreement_with_head": float(m[4])})
    for c in curve:
        print({k: round(v, 3) for k, v in c.items()})
    save_json({"curve": curve}, RESULTS_DIR / "clip_zero_shot_translation.json")


if __name__ == "__main__":
    main()
