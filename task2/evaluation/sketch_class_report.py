"""Final-analysis-only Sketch report for Task 2 or Task 3 (run after every checkpoint and setting is fixed).

    python -m task2.evaluation.sketch_class_report --task task2 --runs source_only dan dann cdan
    python -m task2.evaluation.sketch_class_report --task task3 --runs erm dan_dg sam --compare task2:dan task3:dan_dg

For every run (baseline included): per-class Sketch accuracy, full confusion matrix, and top confusions per class.
Also a confusion-matrix figure and a failure figure (for each run, Sketch images of its weakest class that it gets
wrong, with the predicted class). With --compare, a per-class table of two runs' changes vs their shared baseline
(e.g. target-aware DAN vs target-free DAN-DG, both vs the same Source-only/ERM checkpoint).
Outputs: <task>/results/sketch_class_report/{report.json, per_class.csv, confusions.png, failures.png}.
"""
import argparse
import csv

import numpy as np

from common.device import get_device, make_loader
from common.io import data_root, results_dir, save_json
from shared.checkpoints import checkpoint_file
from shared.pacs import CLASSES, NUM_CLASSES
from shared.pacs_eval import predict
from shared.pacs_models import load_pacs_checkpoint
from shared.pacs_protocol import IMAGENET_MEAN, IMAGENET_STD, eval_transform, target_dataset

N_FAILURES = 6


def _checkpoint(task: str, run: str):
    if task == "task3" and run == "erm":
        return checkpoint_file("task2", "source_only")  # Task 3's ERM is the Task 2 Source-only checkpoint
    return checkpoint_file(task, run)


def _confusion(y, pred) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def _run_report(cm: np.ndarray) -> dict:
    per_class = {}
    for c, name in enumerate(CLASSES):
        row = cm[c]
        wrong = [(CLASSES[j], int(row[j])) for j in np.argsort(-row) if j != c and row[j] > 0][:3]
        per_class[name] = {"n": int(row.sum()), "accuracy": float(row[c] / max(row.sum(), 1)),
                           "top_confusions": [{"predicted_as": p, "count": n} for p, n in wrong]}
    return {"accuracy": float(np.trace(cm) / cm.sum()), "per_class": per_class, "confusion_matrix": cm.tolist()}


def _plot_confusions(reports: dict, path) -> None:
    """Row-normalized confusion matrices in %, drawn close to their printed size (full text width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(reports)
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5})
    if len(runs) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6))
    else:
        fig, axes = plt.subplots(1, len(runs), figsize=(7.0, 2.9))
    axes = np.atleast_1d(axes).ravel()
    for k, (ax, run) in enumerate(zip(axes, runs)):
        cm = np.array(reports[run]["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, rotation=90)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES if k % (2 if len(runs) == 4 else len(runs)) == 0 else [])
        ax.set_title(f"{run} (acc {100 * reports[run]['accuracy']:.1f}%)")
        ax.set_xlabel("predicted")
        if k % (2 if len(runs) == 4 else len(runs)) == 0:
            ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_failures(dataset, rows, path) -> None:
    """rows: [(run, weakest class name, [(index, predicted class name), ...])]; one row per run, labelled on the left."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(rows), N_FAILURES, figsize=(7.0, 1.35 * len(rows) + 0.2))
    axes = np.atleast_2d(axes)
    for r, (run, worst, items) in enumerate(rows):
        for k in range(N_FAILURES):
            ax = axes[r, k]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k < len(items):
                img = dataset[items[k][0]][0].numpy() * std + mean
                ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
                ax.set_title(f"→ {items[k][1]}", fontsize=9)
        axes[r, 0].set_ylabel(f"{run}\ntrue: {worst}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_failures(dataset, y, preds: dict, path) -> list:
    """For each run: its weakest class, and the first N_FAILURES Sketch images of that class it gets wrong."""
    rows, listed = [], []
    for run, pred in preds.items():
        acc = [(pred[y == c] == c).mean() for c in range(NUM_CLASSES)]
        worst = int(np.argmin(acc))
        idx = np.flatnonzero((y == worst) & (pred != worst))[:N_FAILURES]
        rows.append((run, CLASSES[worst], [(int(i), CLASSES[pred[i]]) for i in idx]))
        listed += [{"run": run, "index": int(i), "true": CLASSES[worst], "predicted": CLASSES[pred[i]]} for i in idx]
    _draw_failures(dataset, rows, path)
    return listed


def replot(task: str) -> None:
    """Redraw confusions.png and failures.png from the saved report.json (no model is run)."""
    import json
    out_dir = results_dir(task, "sketch_class_report")
    rep = json.load(open(out_dir / "report.json"))
    _plot_confusions(rep["runs"], out_dir / "confusions.png")
    dataset = target_dataset(data_root() / "pacs", eval_transform(), purpose=f"{task}_final_eval")
    rows = {}
    for f in rep["failures"]:
        rows.setdefault(f["run"], (f["true"], []))[1].append((f["index"], f["predicted"]))
    _draw_failures(dataset, [(run, t, items) for run, (t, items) in rows.items()], out_dir / "failures.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("task2", "task3"), required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="First run is the baseline.")
    parser.add_argument("--compare", nargs=2, default=None, metavar="TASK:RUN",
                        help="Two runs whose per-class changes vs the shared Source-only/ERM baseline are compared.")
    args = parser.parse_args()

    for run in args.runs:  # every checkpoint must exist before Sketch is touched
        if not _checkpoint(args.task, run).exists():
            raise FileNotFoundError(_checkpoint(args.task, run))

    device = get_device()
    dataset = target_dataset(data_root() / "pacs", eval_transform(), purpose=f"{args.task}_final_eval")
    loader = make_loader(dataset, 128, device)
    preds, reports, y = {}, {}, None
    for run in args.runs:
        out = predict(load_pacs_checkpoint(_checkpoint(args.task, run), device)[0], loader, device)
        y = out["y"]
        preds[run] = out["pred"]
        reports[run] = _run_report(_confusion(y, out["pred"]))

    out_dir = results_dir(args.task, "sketch_class_report")
    result = {"baseline": args.runs[0], "runs": reports}
    with open(out_dir / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n"] + [f"{r}_acc" for r in args.runs] + [f"{r}_change" for r in args.runs[1:]])
        for name in CLASSES:
            base = reports[args.runs[0]]["per_class"][name]["accuracy"]
            accs = [reports[r]["per_class"][name]["accuracy"] for r in args.runs]
            w.writerow([name, reports[args.runs[0]]["per_class"][name]["n"]] + accs + [a - base for a in accs[1:]])

    if args.compare:  # e.g. task2:dan vs task3:dan_dg, both measured against the same Source-only/ERM checkpoint
        base_cm = _confusion(y, predict(load_pacs_checkpoint(checkpoint_file("task2", "source_only"), device)[0],
                                        loader, device)["pred"])
        base = _run_report(base_cm)["per_class"]
        cmp = {}
        for spec in args.compare:
            task, run = spec.split(":")
            p = predict(load_pacs_checkpoint(_checkpoint(task, run), device)[0], loader, device)["pred"]
            cmp[spec] = {c: _run_report(_confusion(y, p))["per_class"][c]["accuracy"] - base[c]["accuracy"] for c in CLASSES}
        result["per_class_change_vs_source_only"] = cmp
        with open(out_dir / "compare_per_class.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "source_only_acc"] + [f"{s}_change" for s in args.compare])
            for c in CLASSES:
                w.writerow([c, base[c]["accuracy"]] + [cmp[s][c] for s in args.compare])

    _plot_confusions(reports, out_dir / "confusions.png")
    result["failures"] = _plot_failures(dataset, y, preds, out_dir / "failures.png")
    save_json(result, out_dir / "report.json")
    print(open(out_dir / "per_class.csv").read())
    if args.compare:
        print(open(out_dir / "compare_per_class.csv").read())


if __name__ == "__main__":
    main()
