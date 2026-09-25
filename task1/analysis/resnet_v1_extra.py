"""Extra experiment (not required by the handout): same ResNet-50 architecture, older training recipe.

    python -m task1.analysis.resnet_v1_extra

The handout's ResNet-50 uses torchvision IMAGENET1K_V2 weights (trained with a heavy-augmentation recipe). This runs
IMAGENET1K_V1 (the original, plain recipe) through exactly the same pipeline -- same frozen-feature linear head
recipe, eval subset, interventions and 220 cue conflicts -- so any difference comes from the training recipe, not
the architecture. Output: task1/results/resnet50_v1_extra.json (V2 numbers from summary.json alongside).
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from common.config import load_config
from common.device import get_device
from common.io import PROJECT_ROOT, load_json, save_json
from task1.analysis.feature_similarity import cosine_stability
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import SPLIT_FILE, STL10_CLASSES
from task1.data.transforms import extra_color, patch_shuffle, to_grayscale
from task1.models.backbones import FrozenBackbone
from task1.scripts.run_task1 import (RESULTS_DIR, cache_split_features, evaluate_head, features_from_tensor,
                                     load_split_tensors, run_cue_conflict_eval)


def main():
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    device = get_device(cfg.get("device", "auto"))
    weights = ResNet50_Weights.IMAGENET1K_V1
    net = resnet50(weights=weights)
    net.fc = nn.Identity()
    t = weights.transforms()
    backbone = FrozenBackbone("resnet50_v1", net, 2048, t.mean, t.std).to(device)

    fb = cfg["feature_batch"]
    cached = cache_split_features(backbone, "resnet50_v1", device, fb)  # its own cache files
    head, history = train_linear_head(*cached["train"], *cached["val"], cfg["head"], device)
    eval_x, eval_y = load_split_tensors("test", load_json(SPLIT_FILE)["eval_subset"], fb)
    clean = features_from_tensor(backbone, eval_x, device, fb)
    out = {"weights": "IMAGENET1K_V1", "head_epochs": len(history)}
    out["clean"], clean_pred, _ = evaluate_head(head, clean, eval_y, device)

    color_cfg = dict(cfg["interventions"]["extra_color"]); kind = color_cfg.pop("kind")
    shuffled, _ = patch_shuffle(eval_x, cfg["interventions"]["patch_grid"], torch.Generator().manual_seed(cfg["seed"]))
    for name, x in (("grayscale", to_grayscale(eval_x)), ("extra_color", extra_color(eval_x, kind=kind, **color_cfg)),
                    ("patch_shuffle", shuffled)):
        feats = features_from_tensor(backbone, x, device, fb)
        out[name], _, _ = evaluate_head(head, feats, eval_y, device, pred_clean=clean_pred)
        out[name]["cosine_stability"] = cosine_stability(clean, feats)

    cue = run_cue_conflict_eval(backbone, head, device, fb, STL10_CLASSES)
    out["cue_conflict"] = cue[0]
    v2 = load_json(RESULTS_DIR / "summary.json")["resnet50"]
    save_json({"resnet50_v1": out, "resnet50_v2_from_summary": {k: v2[k] for k in
                                                               ("clean", "grayscale", "extra_color", "patch_shuffle",
                                                                "cue_conflict")}},
              RESULTS_DIR / "resnet50_v1_extra.json")
    for tag, r in (("V1", out), ("V2", v2)):
        c = r["cue_conflict"]
        print(f"{tag}: clean {r['clean']['accuracy']:.3f} gray {r['grayscale']['accuracy']:.3f} "
              f"hue {r['extra_color']['accuracy']:.3f} shuffle {r['patch_shuffle']['accuracy']:.3f} | shape bias "
              f"{c['shape_bias']:.1f} coverage {c['coverage']:.1f} (shape {c['n_shape']} texture {c['n_texture']} "
              f"other {c['n_other']})")


if __name__ == "__main__":
    main()
