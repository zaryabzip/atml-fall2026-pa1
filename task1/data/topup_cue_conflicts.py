"""Top-up for the cue-conflict set: extra candidates only for the directions that fell below 30 readable images
after the visual check (see task1/data/cue_conflicts/visual_rule.txt). Same AdaIN model, alpha and technical
rule as make_cue_conflicts.py; content images not used before; separate seed. Appends to metadata.json with
batch = "topup" and accepted = False until the visual check marks them.

    python -m task1.data.topup_cue_conflicts
"""
import numpy as np
import torch
from PIL import Image

from common.config import load_config
from common.io import PROJECT_ROOT, load_json, save_json
from task1.data.make_cue_conflicts import IMAGES_DIR, OUT_DIR, _class_index_lookup, check_stylization, load_adain_models, stylize
from task1.data.stl10 import STL10_CLASSES, common_transform, load_stl10

TOPUP = {("cat", "car"): 15, ("horse", "airplane"): 15, ("monkey", "ship"): 30}
SEED = 6305


def main():
    cfg = load_config(PROJECT_ROOT / "task1" / "configs" / "default.yaml")
    alpha = cfg["cue_conflict"]["style_strength"]
    meta = load_json(OUT_DIR / "metadata.json")
    used = {(m["content_class"], m["source_image_id"]) for m in meta}
    encoder, decoder = load_adain_models(torch.device("cpu"))
    dataset = load_stl10("test", transform=common_transform())
    pools = _class_index_lookup(STL10_CLASSES, dataset.labels)
    rng = np.random.default_rng(SEED)
    for (content_class, style_class), n in TOPUP.items():
        fresh = [i for i in pools[content_class] if (content_class, int(i)) not in used]
        content_ids = rng.choice(fresh, size=n, replace=False)
        style_ids = rng.choice(pools[style_class], size=n, replace=True)
        for content_id, style_id in zip(map(int, content_ids), map(int, style_ids)):
            content_img, style_img = dataset[content_id][0][None], dataset[style_id][0][None]
            raw = stylize(encoder, decoder, content_img, style_img, alpha)
            check = check_stylization(content_img, raw)
            name = f"{content_class}_shape_{style_class}_texture_{content_id}_{style_id}.jpg"
            if check["accepted"]:
                arr = (raw.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
                Image.fromarray(arr).save(IMAGES_DIR / name, quality=90)
            meta.append({"path": name, "content_class": content_class, "style_class": style_class,
                         "source_image_id": content_id, "style_source_image_id": style_id, "batch": "topup",
                         "technical_accepted": check["accepted"], "visual_rejected": False,
                         "dropped_for_balance": False, "accepted": False,
                         **{k: check[k] for k in ("is_finite", "clipped_fraction", "output_std", "coarse_layout_similarity")}})
        print(f"{content_class}->{style_class}: {n} top-up candidates")
    save_json(meta, OUT_DIR / "metadata.json")


if __name__ == "__main__":
    main()
