"""Extra analysis (after all models and scores were fixed): feature geometry behind the GCSC and PROSER results,
and paired bootstrap intervals for two small AUROC differences.

    python -m task4.evaluation.feature_geometry

Uses only the cached outputs (task4/cache/<run>/*.npz); nothing is retrained and no threshold changes.
- Class spread: mean distance of a training feature to its own class mean, divided by the mean distance between
  class means (lower = tighter classes).
- Distance to the nearest class mean (Euclidean, class means from training features): median for near and far
  unknowns relative to the median for CIFAR-10 test, and the known-vs-unknown AUROC of this distance.
- PROSER placeholder score (max dummy logit - max known logit): median on known test, near and far.
- Paired bootstrap (same resampled images for both sides): MSP minus MLS near AUROC on Vanilla, and GCSC minus
  Vanilla near AUROC under MLS; mean difference and 95% interval.
Output: task4/results/osr/feature_geometry.json.
"""
import numpy as np

from common.io import PROJECT_ROOT, results_dir, save_json
from common.metrics import auroc_known_vs_unknown
from task4.evaluate_osr import NUM_KNOWN, load_cache
from task4.scores.mls import mls
from task4.scores.msp import msp
from task4.scores.proser_score import proser_score

SEED = 6304
N_BOOT = 1000
CACHE = PROJECT_ROOT / "task4" / "cache"


def nearest_mean_distance(features, means):
    return np.min(np.linalg.norm(features[:, None, :] - means[None], axis=2), axis=1)


def geometry(cache):
    f, y = cache["train"]["features"], cache["train"]["labels"]
    means = np.stack([f[y == c].mean(0) for c in range(NUM_KNOWN)])
    within = np.mean([np.linalg.norm(f[y == c] - means[c], axis=1).mean() for c in range(NUM_KNOWN)])
    between = np.mean([np.linalg.norm(means[i] - means[j]) for i in range(NUM_KNOWN) for j in range(i + 1, NUM_KNOWN)])
    d = {p: nearest_mean_distance(cache[p]["features"], means) for p in ("test", "near", "far")}
    ref = np.median(d["test"])
    return {"class_spread": float(within / between),
            "nearest_mean_distance_vs_known": {p: float(np.median(d[p]) / ref) for p in ("near", "far")},
            "distance_auroc": {p: auroc_known_vs_unknown(d["test"], d[p]) for p in ("near", "far")}}


def paired_bootstrap(known_a, unknown_a, known_b, unknown_b, rng):
    """AUROC(a) - AUROC(b), resampling the same known and unknown images for both."""
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(known_a), len(known_a))
        j = rng.integers(0, len(unknown_a), len(unknown_a))
        diffs.append(auroc_known_vs_unknown(known_a[i], unknown_a[j]) - auroc_known_vs_unknown(known_b[i], unknown_b[j]))
    return {"mean": float(np.mean(diffs)), "ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])]}


def main():
    van, gcsc, proser = (load_cache(r, CACHE) for r in ("vanilla", "gcsc", "proser"))
    known = lambda d: d["logits"][:, :NUM_KNOWN]
    res = {"geometry": {"vanilla": geometry(van), "gcsc": geometry(gcsc)},
           "proser_placeholder_median": {p: float(np.median(proser_score(proser[p]["logits"], NUM_KNOWN)))
                                         for p in ("test", "near", "far")}}
    rng = np.random.default_rng(SEED)
    res["bootstrap"] = {
        "vanilla_near_msp_minus_mls": paired_bootstrap(msp(known(van["test"])), msp(known(van["near"])),
                                                       mls(known(van["test"])), mls(known(van["near"])), rng),
        "near_mls_gcsc_minus_vanilla": paired_bootstrap(mls(known(gcsc["test"])), mls(known(gcsc["near"])),
                                                        mls(known(van["test"])), mls(known(van["near"])), rng)}
    save_json(res, results_dir("task4", "osr") / "feature_geometry.json")
    print(res)


if __name__ == "__main__":
    main()
