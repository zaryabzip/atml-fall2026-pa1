"""Numbers behind the t-SNE plots, so they can be described without eyeballing small markers.

    python -m task1.analysis.tsne_metrics

For each saved joint projection (task1/results/<model>/projection_<intervention>.json; rows 0..N-1 are clean,
rows N..2N-1 the transformed versions of the same images in the same order), using k = 10 nearest neighbours in the
2-D t-SNE plane:
- mixing: share of a transformed point's neighbours that are clean (0 = crosses sit apart from circles; about 0.5 =
  fully mixed);
- class kept: share of a transformed point's neighbours with the same class (chance about 0.1);
- nearest clean same class: whether the closest clean point has the transformed point's class;
- own original found: whether the transformed image's own clean version is among its 10 nearest clean points.
All values are averaged over the transformed points. They describe the plots, which t-SNE distorts: only local
neighbourhoods are meaningful, not distances between clusters or positions in the plane.
Output: task1/results/tsne_metrics.json.
"""
import json

import numpy as np

from task1.scripts.run_task1 import RESULTS_DIR

K = 10


def metrics(e, c, dom):
    n = int((dom == 0).sum())
    d = ((e[:, None, :] - e[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    t = np.arange(n, 2 * n)
    nn = np.argsort(d[t], axis=1)[:, :K]
    clean_d = d[t][:, :n]
    nn_clean = np.argsort(clean_d, axis=1)[:, :K]
    return {"mixing": float((dom[nn] == 0).mean()),
            "class_kept": float((c[nn] == c[t][:, None]).mean()),
            "nearest_clean_same_class": float((c[nn_clean[:, 0]] == c[t]).mean()),
            "own_original_in_10": float(np.mean([(i - n) in row for i, row in zip(t, nn_clean)])),
            # the same class-kept measure for the clean points, as a reference for how tight the clean clusters are
            "clean_class_kept": float((c[np.argsort(d[:n], axis=1)[:, :K]] == c[:n][:, None]).mean())}


def main():
    out = {}
    for m in ("resnet50", "vit_b16", "clip_b32"):
        for k in ("grayscale", "patch_shuffle", "translation", "cue_conflict"):
            p = json.load(open(RESULTS_DIR / m / f"projection_{k}.json"))
            r = metrics(np.array(p["embedding"]), np.array(p["class_labels"]), np.array(p["domain"]))
            out.setdefault(m, {})[k] = r
            print(f"{m:9s} {k:13s} " + "  ".join(f"{a} {b:.2f}" for a, b in r.items()))
    json.dump(out, open(RESULTS_DIR / "tsne_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
