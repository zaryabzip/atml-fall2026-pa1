"""Extra analysis (not required by the handout): how far do our ViTs look, layer by layer, and what do their
position embeddings encode?

    python -m task1.analysis.attention_distance

For ViT-B/16 (14x14 grid of 16 px patches) and CLIP ViT-B/32 (7x7 grid of 32 px patches), both frozen as in Task 1:
1. Mean attention distance (Dosovitskiy et al. 2021, Sec. 4.5): for each query patch, the pixel distance to every other
   patch weighted by the attention weight; averaged over query patches and 100 evaluation images, per head.
   Patch-to-patch attention only (class token excluded, rows renormalized). "uniform_attention_px" is the value a head
   gets if it looks everywhere equally; small values mean a head looks at its neighbours.
2. Position embeddings: mean cosine between the embeddings of two patch slots, by their distance on the grid.
Output: task1/results/attention_distance.json.
"""
import torch

from common.io import load_json, save_json
from task1.data.stl10 import SPLIT_FILE
from task1.scripts.run_task1 import RESULTS_DIR, load_split_tensors


def load(name):
    """Returns (model, blocks whose .attn/.self_attention is an nn.MultiheadAttention, attr, grid, patch_px,
    patch position embeddings (grid*grid, dim), mean, std, forward function)."""
    if name == "vit_b16":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        w = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=w).eval()
        t = w.transforms()
        return (net.encoder.layers, "self_attention", 14, 16, net.encoder.pos_embedding[0, 1:].detach(),
                t.mean, t.std, net)
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    v = model.visual.eval()
    return (v.transformer.resblocks, "attn", 7, 32, v.positional_embedding[1:].detach(),
            open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD, v)


def measure(name, x):
    blocks, attr, grid, patch, pos, mean, std, net = load(name)
    maps = []
    for block in blocks:  # both libraries ask for no attention weights; wrap each layer to keep them
        mha = getattr(block, attr)
        original = mha.forward

        def keep_weights(q, k, v, *args, _original=original, **kw):
            kw.pop("need_weights", None)
            out, a = _original(q, k, v, *args, need_weights=True, average_attn_weights=False, **kw)
            maps.append(a.detach())
            return out, None
        mha.forward = keep_weights
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    with torch.no_grad():
        net(x)

    cells = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij"), -1).reshape(-1, 2).float()
    dist_px = torch.cdist(cells, cells) * patch
    layers = []
    for a in maps:
        if a.shape[0] != len(x):  # (L, N, ...) layouts: put the batch first
            a = a.transpose(0, 1)
        a = a[:, :, 1:, 1:]
        a = a / a.sum(-1, keepdim=True)
        per_head = (a * dist_px).sum(-1).mean(-1).mean(0)
        layers.append({"mean": float(per_head.mean()), "min_head": float(per_head.min()),
                       "max_head": float(per_head.max()), "per_head": per_head.tolist()})

    pe = pos / pos.norm(dim=1, keepdim=True)
    sim = pe @ pe.T
    dist_cells = torch.cdist(cells, cells)
    by_distance = {}
    for label, lo, hi in (("1 apart", 1, 1.01), ("diagonal", 1.4, 1.5), ("2 apart", 2, 2.01), ("4 apart", 4, 4.01),
                          (f"{grid // 2}+ apart", grid // 2, 99)):
        m = (dist_cells >= lo) & (dist_cells < hi)
        by_distance[label] = float(sim[m].mean())
    off = dist_cells > 0
    corr = float(torch.corrcoef(torch.stack([dist_cells[off], sim[off]]))[0, 1])
    return {"grid": grid, "patch_px": patch, "uniform_attention_px": float(dist_px.mean()), "layers": layers,
            "position_embedding_cosine_by_distance": by_distance, "position_embedding_distance_correlation": corr}


def main():
    x, _ = load_split_tensors("test", load_json(SPLIT_FILE)["eval_subset"][:100], 50)
    res = {"images": len(x)}
    for name in ("vit_b16", "clip_b32"):
        r = measure(name, x)
        res[name] = r
        print(f"\n{name}: uniform attention = {r['uniform_attention_px']:.1f} px")
        for i, l in enumerate(r["layers"], 1):
            n_local = sum(h < 0.45 * r["uniform_attention_px"] for h in l["per_head"])
            print(f"  layer {i:2d}: mean {l['mean']:6.1f} px, heads {l['min_head']:6.1f}-{l['max_head']:6.1f}, "
                  f"local heads (<45% of uniform) {n_local}")
        print("  position-embedding cosine:", {k: round(v, 3) for k, v in r["position_embedding_cosine_by_distance"].items()},
              "corr with distance", round(r["position_embedding_distance_correlation"], 3))
    save_json(res, RESULTS_DIR / "attention_distance.json")


if __name__ == "__main__":
    main()
