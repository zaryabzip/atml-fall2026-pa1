"""Compact plot for one controlled design study (the required "alignment-strength" figure).

Works for any single-parameter study, because the x-value of each run is read from that run's own
saved config.yaml. Evaluate the study first into its own folder, then plot it:

    python -m task2.evaluate_final --runs source_only dan_lmmd0.1 dan dan_lmmd10 \\
        --baseline source_only --out-name design_study_dan
    python -m task2.evaluation.plot_design_study --eval-name design_study_dan \\
        --runs dan_lmmd0.1 dan dan_lmmd10 --param method.lambda_mmd --reference source_only --log-x

Writes task2/results/<eval-name>/study_plot.png.

Task 3 studies work the same way with --task task3 (the panels then show source F1, Sketch accuracy,
source-domain separability and the sharpness proxy):

    python -m task3.evaluate_sketch --runs erm dan_dg_lam0.1 dan_dg dan_dg_lam10 --out-name design_study_dan_dg
    python -m task2.evaluation.plot_design_study --task task3 --eval-name design_study_dan_dg \\
        --runs dan_dg_lam0.1 dan_dg dan_dg_lam10 --param method.lambda_dg --reference erm --log-x
"""
import argparse

from common.config import load_config
from common.io import PROJECT_ROOT, load_json
from common.plotting import save_figure

PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")

# (title, y-axis label, how to read the value out of one run's entry in final_eval.json)
PANELS_BY_TASK = {
    "task2": (
        ("Mean source macro-F1", "macro-F1", lambda r: r["source_val"]["mean_macro_f1"]),
        ("Sketch accuracy", "accuracy", lambda r: r["sketch"]["accuracy"]),
        ("Domain separability", "held-out accuracy (50% = chance)", lambda r: r["domain_separability"]),
    ),
    "task3": (
        ("Mean source macro-F1", "macro-F1", lambda r: r["source_val"]["mean_macro_f1"]),
        ("Sketch accuracy", "accuracy", lambda r: r["sketch"]["accuracy"]),
        ("Source-domain separability", "held-out accuracy (33% = chance)", lambda r: r["source_domain_separability"]),
        ("Sharpness proxy", "loss increase", lambda r: r["sharpness"]),
    ),
}


def _param_value(run_name: str, dotted_key: str, task: str = "task2"):
    """Reads e.g. 'method.lambda_mmd' out of the config that run was actually trained with."""
    node = load_config(PROJECT_ROOT / task / "results" / run_name / "config.yaml")  # read-only: no mkdir
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Run {run_name!r} has no config value {dotted_key!r}.")
        node = node[part]
    return node


def plot_design_study(eval_name, runs, param, reference=None, log_x=False, path=None, task="task2") -> None:
    import matplotlib.pyplot as plt

    results = load_json(PROJECT_ROOT / task / "results" / eval_name / "final_eval.json")
    panels = PANELS_BY_TASK[task]

    # Pair every run with the parameter value it was trained with, and order the points by it.
    points = sorted((_param_value(run, param, task), run) for run in runs)
    xs = [x for x, _ in points]

    # Drawn close to its printed size (full text width) so the text stays readable.
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "legend.fontsize": 8.5})
    if len(panels) <= 3:
        fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.6))
    else:
        fig, axes = plt.subplots(2, (len(panels) + 1) // 2, figsize=(6.6, 5.0))
        axes = axes.ravel()
    fig.patch.set_facecolor("white")
    for ax, (title, ylabel, getter) in zip(axes, panels):
        # One line through the study runs, with each point's value written next to it.
        ys = [getter(results[run]) for _, run in points]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=PALETTE[0])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

        # Optional dashed reference line, e.g. Source-only, which has no value for this parameter.
        if reference is not None:
            ax.axhline(getter(results[reference]), linestyle="--", linewidth=1.4, color="#888888",
                       label=f"{reference}")

        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel(param.split(".")[-1])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.margins(x=0.15, y=0.18)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    if reference is not None:
        axes[0].legend(frameon=True)

    fig.tight_layout()
    save_figure(fig, path or (PROJECT_ROOT / task / "results" / eval_name / "study_plot.png"), dpi=220)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task2", choices=list(PANELS_BY_TASK), help="Which task's results/configs to read.")
    parser.add_argument("--eval-name", required=True, help="The --out-name the study was evaluated into.")
    parser.add_argument("--runs", nargs="+", required=True, help="The study runs (not the reference).")
    parser.add_argument("--param", required=True, help="Dotted config key varied in the study, e.g. method.lambda_mmd")
    parser.add_argument("--reference", default=None, help="Optional run drawn as a dashed line, e.g. source_only")
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plot_design_study(args.eval_name, args.runs, args.param, args.reference, args.log_x, args.out, args.task)
    print(f"Saved study plot for {args.eval_name}")


if __name__ == "__main__":
    main()
