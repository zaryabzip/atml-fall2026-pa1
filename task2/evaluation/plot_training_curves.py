"""Training curves: classification loss + MMD/domain loss, comparing several runs on one figure.
Required evidence per the README ("Training curves: classification loss + MMD or domain loss").

    python -m task2.evaluation.plot_training_curves --runs source_only dan dann cdan

Works for Task 3 runs too. A run name is looked up under --task (default task2), or give it as
"<task>:<run>" to mix tasks on one figure, e.g. ERM (the Task 2 Source-only run) next to Task 3 runs:

    python -m task2.evaluation.plot_training_curves --runs task2:source_only task3:dan_dg task3:sam \\
        --out task3/results/training_curves.png
"""
import argparse
import json

from common.config import load_config
from common.io import PROJECT_ROOT
from common.plotting import save_figure

PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C")

# Which train_log.jsonl key holds this method's "alignment" loss (None = has no such term).
METHOD_SPECIFIC_KEY = {
    "source_only": None,
    "dan": "train/mmd",
    "dann": "train/domain_loss",
    "cdan": "train/domain_loss",
    "dan_dg": "train/mmd",  # Task 3: mean MMD over the 3 source-domain pairs
    "sam": None,  # Task 3: no alignment term
}


def _load_run_log(run_name: str, default_task: str = "task2") -> tuple:
    """Returns (config, records) for one run, reading its saved config.yaml + train_log.jsonl.
    `run_name` is either "<run>" (looked up under default_task) or "<task>:<run>"."""
    task, _, name = run_name.rpartition(":")
    out_dir = PROJECT_ROOT / (task or default_task) / "results" / name  # e.g. task2/results/dann/ (read-only: no mkdir)
    config = load_config(out_dir / "config.yaml")

    records = []  # one dict per logged epoch
    with open(out_dir / "train_log.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return config, records


def plot_training_curves(run_names, path, default_task: str = "task2") -> None:
    """One figure, four panels, one line per run: classification loss, MMD/domain loss, discriminator accuracy
    (adversarial runs only; 0.5 = chance) and mean source-validation macro-F1 (the checkpoint-selection metric)."""
    import matplotlib.pyplot as plt

    # Drawn close to its printed size (about 5.2 in wide in the report) so the text stays readable.
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "legend.fontsize": 8.5})
    has_disc = any(r and "train/domain_acc" in r[0] for r in (_load_run_log(n, default_task)[1] for n in run_names))
    if has_disc:
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2))
        (cls_ax, method_ax), (disc_ax, f1_ax) = axes
    else:  # no adversarial run: one row of three panels instead of an empty discriminator panel
        fig, (cls_ax, method_ax, f1_ax) = plt.subplots(1, 3, figsize=(7.0, 2.6))
        disc_ax = fig.add_subplot(111); disc_ax.set_visible(False)
    fig.patch.set_facecolor("white")

    # Plot each run's classification loss, and its MMD/domain loss if it has one.
    any_method_specific = False
    for i, run_name in enumerate(run_names):
        config, records = _load_run_log(run_name, default_task)
        if not records:  # an empty log (e.g. training crashed before logging) -- skip quietly
            continue
        method_name = config["method"]["name"]
        color = PALETTE[i % len(PALETTE)]

        epochs = [r["epoch"] for r in records]
        cls_losses = [r["train/cls_loss"] for r in records]
        cls_ax.plot(epochs, cls_losses, marker="o", markersize=3, linewidth=1.5, color=color, label=run_name)

        specific_key = METHOD_SPECIFIC_KEY.get(method_name)
        if specific_key is not None and specific_key in records[0]:
            any_method_specific = True
            values = [r[specific_key] for r in records]
            term_name = "MMD" if method_name in ("dan", "dan_dg") else "domain loss"
            method_ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.5, color=color,
                          label=f"{run_name} ({term_name})")
        if "train/domain_acc" in records[0]:
            disc_ax.plot(epochs, [r["train/domain_acc"] for r in records], marker="o", markersize=3, linewidth=1.5,
                         color=color, label=run_name)
        f1_ax.plot(epochs, [r["val/mean_macro_f1"] for r in records], marker="o", markersize=3, linewidth=1.5,
                   color=color, label=run_name)

    # Style the left panel (classification loss).
    cls_ax.set_xlabel("epoch")
    cls_ax.set_ylabel("classification loss")
    cls_ax.set_title("Classification loss", fontweight="bold")
    cls_ax.grid(True, alpha=0.25, linewidth=0.7)
    cls_ax.legend()
    for spine in ("top", "right"):
        cls_ax.spines[spine].set_visible(False)

    # Style the right panel (MMD / domain loss), or say plainly that it's empty.
    method_ax.set_xlabel("epoch")
    method_ax.set_ylabel("alignment loss")
    method_ax.set_title("MMD / domain loss", fontweight="bold")
    method_ax.grid(True, alpha=0.25, linewidth=0.7)
    if any_method_specific:
        method_ax.legend()
    else:
        method_ax.text(0.5, 0.5, "no adaptation runs selected", ha="center", va="center",
                       transform=method_ax.transAxes, color="#888888")
    for spine in ("top", "right"):
        method_ax.spines[spine].set_visible(False)

    disc_ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    for ax, ylabel, title in ((disc_ax, "discriminator accuracy", "Discriminator accuracy (0.5 = chance)"),
                              (f1_ax, "macro-F1", "Mean source-validation macro-F1")):
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    if not has_disc:
        fig.delaxes(disc_ax)
    fig.tight_layout()
    save_figure(fig, path, dpi=220)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--task", default="task2", help="Task to look runs up under unless given as task:run")
    parser.add_argument("--out", default=None, help="Defaults to <task>/results/training_curves.png")
    args = parser.parse_args()

    out_path = args.out or (PROJECT_ROOT / args.task / "results" / "training_curves.png")
    plot_training_curves(args.runs, out_path, args.task)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
