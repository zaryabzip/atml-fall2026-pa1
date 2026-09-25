"""Open-set evaluation from cached outputs (task4/cache/<run>/*.npz).

    python -m task4.evaluate_osr
Needs task4.extract_outputs --include-unknowns for vanilla, gcsc and proser.
Writes task4/results/osr/{osr_tables.json, score_table.csv, model_table.csv, score_figure.png,
failures.json, failures.png, score_agreement.json}.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.data import Subset

from common.io import PROJECT_ROOT, results_dir, save_json
from common.metrics import auroc_known_vs_unknown, rejection_report, threshold_from_validation
from common.plotting import save_figure
from task4.data.cifar import cifar100_unknowns
from task4.evaluation.failure_analysis import CIFAR10_CLASSES, accepted_unknowns
from task4.scores.energy import energy
from task4.scores.mahalanobis import fit_class_gaussians, mahalanobis
from task4.scores.mls import mls
from task4.scores.msp import msp
from task4.scores.proser_score import proser_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM_KNOWN = 10
PARTS = ("train", "val", "test", "near", "far")


def load_cache(run: str, cache_root) -> dict:
    folder = Path(cache_root) / run
    missing = [p for p in PARTS if not (folder / f"{p}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}. Run task4.extract_outputs --run {run} --include-unknowns.")
    return {p: dict(np.load(folder / f"{p}.npz")) for p in PARTS}


def score_all(cache: dict, score_fn) -> dict:
    """score_fn(part_dict) -> u. Returns u for val, test, near, far."""
    return {p: score_fn(cache[p]) for p in ("val", "test", "near", "far")}


def osr_row(u: dict) -> dict:
    tau = threshold_from_validation(u["val"])
    all_unknown = np.concatenate([u["near"], u["far"]])
    near, far, both = (rejection_report(u["test"], x, tau) for x in (u["near"], u["far"], all_unknown))
    return {
        "auroc_near": auroc_known_vs_unknown(u["test"], u["near"]),
        "auroc_far": auroc_known_vs_unknown(u["test"], u["far"]),
        "auroc_all": auroc_known_vs_unknown(u["test"], all_unknown),
        "threshold": tau,
        "test_known_acceptance": near["known_acceptance"],
        "near_rejection": near["unknown_rejection"], "far_rejection": far["unknown_rejection"],
        "all_rejection": both["unknown_rejection"],
        "near_fpr_at_95tpr": near["fpr_at_95tpr"], "far_fpr_at_95tpr": far["fpr_at_95tpr"],
        "all_fpr_at_95tpr": both["fpr_at_95tpr"],
    }


def csa(cache: dict) -> float:
    return float((cache["test"]["logits"][:, :NUM_KNOWN].argmax(1) == cache["test"]["labels"]).mean())


def write_csv(rows: list, path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_figure(scores: dict, path) -> None:
    """Top row: score histograms (known test / near / far) with the val threshold. Bottom row: ROC curves."""
    # Drawn close to its printed size (full text width) so the text stays readable.
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(2, len(scores), figsize=(7.0, 4.9))
    colors = {"test": "#4C72B0", "near": "#DD8452", "far": "#55A868"}
    labels = {"test": "known (test)", "near": "near unknown", "far": "far unknown"}
    for col, (name, u) in enumerate(scores.items()):
        ax = axes[0, col]
        lo, hi = np.percentile(np.concatenate([u["test"], u["near"], u["far"]]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for part in ("test", "near", "far"):
            ax.hist(u[part], bins=bins, density=True, alpha=0.45, color=colors[part], label=labels[part])
        ax.axvline(threshold_from_validation(u["val"]), color="k", ls="--", lw=1, label="threshold")
        ax.set_title(f"{name} (vanilla)")
        ax.set_xlabel("unknownness $u(x)$")
        ax.set_yticks([])
        ax = axes[1, col]
        for part in ("near", "far"):
            y = np.r_[np.zeros(len(u["test"])), np.ones(len(u[part]))]
            fpr, tpr, _ = roc_curve(y, np.r_[u["test"], u[part]])
            ax.plot(fpr, tpr, color=colors[part], label=f"{part}: {auroc_known_vs_unknown(u['test'], u[part]):.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
        ax.set_xlabel("FPR (known rejected)")
        ax.set_ylabel("TPR (unknown rejected)")
        ax.legend(loc="lower right", title="AUROC", title_fontsize=7.5)
    axes[0, 0].legend()
    fig.tight_layout()
    save_figure(fig, path)


def failure_figure(failures: dict, datasets: dict, path) -> None:
    items = [(g, f) for g in ("near", "far") for f in failures[g]]
    if not items:  # possible only for a barely trained model (e.g. the preflight check)
        print("no accepted unknowns: failures.png not written")
        return
    ncol = 4  # near unknowns first, then far; the shared threshold goes in the figure caption
    fig, axes = plt.subplots(-(-len(items) // ncol), ncol, figsize=(6.4, 5.7))
    mean = np.array([0.4914, 0.4822, 0.4465])[:, None, None]
    std = np.array([0.2470, 0.2435, 0.2616])[:, None, None]
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (group, f) in zip(np.atleast_1d(axes).ravel(), items):
        img = datasets[group][f["index"]][0].numpy() * std + mean
        ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
        ax.set_title(f"{group}: {f['unknown_class'].replace('_', ' ')}\n→ {f['predicted_class']} (u = {f['score']:.2f})",
                     fontsize=9, color="#B5541C" if group == "near" else "#2F7D3A")
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, path)


def score_agreement(scores: dict, fine_names, labels: dict) -> dict:
    """Where the Vanilla scores agree and disagree (research question 2).
    rank_correlation: Spearman correlation between every pair of scores, per part (test, near, far).
    decisions: at each score's own 95%-validation threshold, how often a pair makes the same accept/reject call on
    unknowns, and how often only one of the two rejects. per_class_rejection: rejection rate of every unknown class
    under every score, so it is visible which classes one score catches and another misses."""
    from itertools import combinations

    from scipy.stats import spearmanr

    names = list(scores)
    taus = {n: threshold_from_validation(scores[n]["val"]) for n in names}
    out = {"rank_correlation": {}, "decisions": {}, "per_class_rejection": {}}
    for part in ("test", "near", "far"):
        out["rank_correlation"][part] = {f"{a} vs {b}": float(spearmanr(scores[a][part], scores[b][part])[0])
                                         for a, b in combinations(names, 2)}
    for part in ("near", "far"):
        rejected = {n: scores[n][part] > taus[n] for n in names}
        out["decisions"][part] = {f"{a} vs {b}": {"same_call": float((rejected[a] == rejected[b]).mean()),
                                                  f"only_{a}_rejects": float((rejected[a] & ~rejected[b]).mean()),
                                                  f"only_{b}_rejects": float((rejected[b] & ~rejected[a]).mean())}
                                  for a, b in combinations(names, 2)}
        out["per_class_rejection"][part] = {
            fine_names[c]: {n: float(rejected[n][labels[part] == c].mean()) for n in names}
            for c in np.unique(labels[part])}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=3, default=["vanilla", "gcsc", "proser"], metavar=("VANILLA", "GCSC", "PROSER"))
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "task4" / "cache"))
    parser.add_argument("--out-dir", default=None, help="Default: task4/results/osr")
    parser.add_argument("--proser-final", default="proser_final",
                        help="Cache of PROSER's final-epoch model, reported as extra rows if present ('' to skip).")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else results_dir("task4", "osr")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_names = dict(zip(("vanilla", "gcsc", "proser"), args.runs))
    caches = {model: load_cache(run, args.cache_root) for model, run in run_names.items()}
    van = caches["vanilla"]

    # 1. Four post-hoc scores on the SAME vanilla logits/features.
    means, variances = fit_class_gaussians(van["train"]["features"], van["train"]["labels"], NUM_KNOWN)
    vanilla_scores = {
        "MSP": score_all(van, lambda d: msp(d["logits"])),
        "MLS": score_all(van, lambda d: mls(d["logits"])),
        "Energy": score_all(van, lambda d: energy(d["logits"])),
        "Mahalanobis": score_all(van, lambda d: mahalanobis(d["features"], means, variances)),
    }
    score_table = [{"score": name, **osr_row(u)} for name, u in vanilla_scores.items()]

    # 2. Vanilla vs GCSC vs PROSER with MLS on the 10 known logits, plus PROSER with its placeholder score.
    model_table = []
    for run in ("vanilla", "gcsc", "proser"):
        u = score_all(caches[run], lambda d: mls(d["logits"][:, :NUM_KNOWN]))
        model_table.append({"model": run, "score": "MLS", "csa": csa(caches[run]), **osr_row(u)})
    u = score_all(caches["proser"], lambda d: proser_score(d["logits"], NUM_KNOWN))
    model_table.append({"model": "proser", "score": "placeholder", "csa": csa(caches["proser"]), **osr_row(u)})

    # Extra, clearly labelled rows (NOT the handout's model): PROSER after its final epoch instead of the epoch picked
    # by validation accuracy (epoch 1). Picking the last epoch uses no unknown data. Skipped if not extracted.
    if args.proser_final and (Path(args.cache_root) / args.proser_final / "far.npz").exists():
        final = load_cache(args.proser_final, args.cache_root)
        for score_name, fn in (("MLS", lambda d: mls(d["logits"][:, :NUM_KNOWN])),
                               ("placeholder", lambda d: proser_score(d["logits"], NUM_KNOWN))):
            model_table.append({"model": "proser_final_epoch50 (extra)", "score": score_name, "csa": csa(final),
                                **osr_row(score_all(final, fn))})

    # 3. Figure for MSP, MLS, Mahalanobis.
    score_figure({k: vanilla_scores[k] for k in ("MSP", "MLS", "Mahalanobis")}, out_dir / "score_figure.png")

    # 4. Failure analysis at the vanilla MLS threshold.
    tau = threshold_from_validation(vanilla_scores["MLS"]["val"])
    datasets = {g: Subset(cifar100_unknowns(g), range(len(van[g]["labels"]))) for g in ("near", "far")}
    fine_names = datasets["near"].dataset.dataset.classes
    failures, per_class = {}, {}
    for group in ("near", "far"):
        preds = van[group]["logits"].argmax(1)
        failures[group], per_class[group] = accepted_unknowns(
            vanilla_scores["MLS"][group], tau, preds, van[group]["labels"], fine_names, k=6)
    failure_figure(failures, datasets, out_dir / "failures.png")

    # 5. Agreement and disagreement between the four Vanilla scores.
    agreement = score_agreement(vanilla_scores, fine_names, {g: van[g]["labels"] for g in ("near", "far")})
    save_json(agreement, out_dir / "score_agreement.json")

    save_json({"vanilla_scores": score_table, "models": model_table, "known_classes": list(CIFAR10_CLASSES)},
              out_dir / "osr_tables.json")
    save_json({"threshold_mls_vanilla": tau, "failures": failures, "per_class": per_class}, out_dir / "failures.json")
    write_csv(score_table, out_dir / "score_table.csv")
    write_csv(model_table, out_dir / "model_table.csv")

    for title, rows, key in (("Vanilla scores", score_table, "score"), ("Models", model_table, "model")):
        print(f"\n{title}")
        for r in rows:
            extra = f" csa={r['csa']:.4f}" if "csa" in r else ""
            print(f"  {r[key]:<30} {r.get('score', '') if key == 'model' else '':<12}{extra} "
                  f"AUROC near={r['auroc_near']:.4f} far={r['auroc_far']:.4f} all={r['auroc_all']:.4f} "
                  f"accept={r['test_known_acceptance']:.3f} rej near={r['near_rejection']:.3f} far={r['far_rejection']:.3f}")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
