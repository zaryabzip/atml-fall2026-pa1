"""Final-analysis extra (all Task 3 models and settings are fixed): why does DAN-DG lose "person" on Sketch?

    python -m task3.evaluation.person_analysis

Hypothesis tested: people look especially different across Photo, Art Painting and Cartoon, so aligning the source
domains erased much of what made a person recognizable.
1. Per-class domain dependence: within each class, how well a linear probe tells Photo / Art / Cartoon apart from
   the frozen features (balanced per domain, 70/30 split, logistic regression C = 1, seed 6304; chance 33.3%).
   If the hypothesis holds, "person" should be the most domain-dependent class under ERM, and lose the most under
   DAN-DG.
2. Per-class accuracy on the source validation images, ERM vs DAN-DG. If alignment erased person information in
   general, person accuracy should also drop on the sources, not only on Sketch.
3. Where Sketch persons land: mean cosine similarity of Sketch-person features to each source class centroid
   (class means of source-validation features), and the share whose nearest centroid is each class.
Output: task3/results/person_analysis.json.
"""
import numpy as np

from common.device import get_device, make_loader
from common.io import data_root, results_dir, save_json
from common.metrics import linear_probe_accuracy
from shared.pacs import CLASSES
from shared.pacs_eval import predict
from shared.pacs_protocol import eval_transform, load_splits, source_datasets, target_dataset
from task3.evaluate_sketch import _load_model

SEED = 6304


def per_class_domain_separability(feats_by_domain, ys_by_domain):
    out = {}
    rng = np.random.default_rng(SEED)
    for c, name in enumerate(CLASSES):
        parts = [f[y == c] for f, y in zip(feats_by_domain, ys_by_domain)]
        n = min(len(p) for p in parts)
        kept = [p[rng.choice(len(p), size=n, replace=False)] for p in parts]
        x = np.concatenate(kept)
        lab = np.concatenate([np.full(n, i) for i in range(len(kept))])
        out[name] = {"per_domain_n": n, "separability": linear_probe_accuracy(x, lab, SEED, 0.3, 1.0)}
    return out


def main():
    device = get_device()
    pacs = data_root() / "pacs"
    val = source_datasets(pacs, "val", eval_transform(), load_splits())
    sketch = target_dataset(pacs, eval_transform(), purpose="task3_final_eval")
    res = {}
    for run in ("erm", "dan_dg"):
        model = _load_model(run, device)
        src = {d: predict(model, make_loader(ds, 128, device), device, return_features=True) for d, ds in val.items()}
        sk = predict(model, make_loader(sketch, 128, device), device, return_features=True)
        fd = [src[d]["features"] for d in val]; yd = [src[d]["y"] for d in val]
        f_all, y_all = np.concatenate(fd), np.concatenate(yd)
        p_all = np.concatenate([src[d]["pred"] for d in val])

        # 3. centroids of source-validation features, compared by cosine
        norm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
        centroids = norm(np.stack([f_all[y_all == c].mean(0) for c in range(len(CLASSES))]))
        person = CLASSES.index("person")
        sp = norm(sk["features"][sk["y"] == person])
        cos = sp @ centroids.T
        res[run] = {
            "domain_separability_within_class": per_class_domain_separability(fd, yd),
            "source_val_accuracy_per_class": {n: float((p_all[y_all == c] == c).mean()) for c, n in enumerate(CLASSES)},
            "source_val_person_accuracy_per_domain": {d: float((src[d]["pred"][src[d]["y"] == person] == person).mean())
                                                      for d in val},
            "sketch_person": {
                "n": int(len(sp)),
                "accuracy": float((sk["pred"][sk["y"] == person] == person).mean()),
                "mean_cosine_to_centroid": {n: float(cos[:, c].mean()) for c, n in enumerate(CLASSES)},
                "nearest_centroid_share": {n: float((cos.argmax(1) == c).mean()) for c, n in enumerate(CLASSES)}},
        }
        r = res[run]
        print(f"\n== {run}")
        print(" within-class domain separability:",
              {k: round(v["separability"], 2) for k, v in r["domain_separability_within_class"].items()})
        print(" source-val accuracy per class:", {k: round(v, 3) for k, v in r["source_val_accuracy_per_class"].items()})
        print(" source-val person accuracy per domain:",
              {k: round(v, 3) for k, v in r["source_val_person_accuracy_per_domain"].items()})
        s = r["sketch_person"]
        print(f" sketch person acc {s['accuracy']:.3f}; mean cosine to centroids:",
              {k: round(v, 3) for k, v in s["mean_cosine_to_centroid"].items()})
        print("  nearest centroid share:", {k: round(v, 2) for k, v in s["nearest_centroid_share"].items()})
    save_json(res, results_dir("task3", "person_analysis") / "person_analysis.json")


if __name__ == "__main__":
    main()
