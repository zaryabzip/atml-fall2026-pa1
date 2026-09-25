"""Training-stability check for DANN and CDAN: the handout setup vs. the normalized discriminator input.

    python -m task2.evaluation.stability_check

Each run trains for 120 steps with the whole gradient-reversal schedule (alpha from 0 to 1) compressed into those
steps, and records every step's mean backbone-feature norm, classification loss, domain loss and discriminator
accuracy, plus the mean source-validation macro-F1 at the end. Only unlabeled Sketch images are used (as in Task 2
training). Outputs: task2/results/stability_check/{steps.json, summary.csv, stability.png}.
"""
import csv
import os
import tempfile

import torch

from common.config import load_config
from common.io import PROJECT_ROOT, load_json, save_json

OUT = PROJECT_ROOT / "task2" / "results" / "stability_check"
STEPS = 120
RUNS = [("dann", False), ("dann", True), ("cdan", False), ("cdan", True)]


def run_one(method_name: str, norm: bool) -> dict:
    from shared.pacs_train import fit
    from task2.methods import METHODS

    base = METHODS[method_name]
    records = []

    class Logged(base):
        def __init__(self, cfg, model, device):
            super().__init__(cfg, model, device)
            self._norms = []
            backbone = getattr(model, "module", model).backbone
            backbone.register_forward_hook(lambda _m, _i, out: self._norms.append(out.detach().float().norm(dim=1).mean().item()))

        def compute_loss(self, source, target, progress):
            loss, logs = super().compute_loss(source, target, progress)
            records.append({"step": len(records) + 1, "alpha": logs["alpha"].item(), "feature_norm": self._norms[-1],
                            "cls_loss": logs["cls_loss"].item(), "domain_loss": logs["domain_loss"].item(),
                            "domain_acc": logs["domain_acc"].item()})
            return loss, logs

    tag = f"{method_name}_{'layernorm' if norm else 'handout'}"
    cfg = load_config(PROJECT_ROOT / "task2" / "configs" / f"{method_name}.yaml", [
        "train.max_epochs=1", f"train.steps_per_epoch={STEPS}", f"run_name=stability_check/{tag}",
        f"method.disc_input_norm={str(norm).lower()}", "method.disc_lr_mult=1.0",
    ])
    best = fit(cfg, Logged, task="task2")
    return {"run": tag, "method": method_name, "disc_input_norm": norm, "steps": records,
            "source_val_macro_f1": best["mean_macro_f1"]}


def plot(results: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn close to its printed size (full text width) so the text stays readable.
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for r in results:
        steps = [s["step"] for s in r["steps"]]
        style = "-" if r["disc_input_norm"] else "--"
        label = f"{r['method'].upper()} ({'LayerNorm' if r['disc_input_norm'] else 'handout'})"
        axes[0].plot(steps, [s["feature_norm"] for s in r["steps"]], style, label=label)
        axes[1].plot(steps, [s["cls_loss"] for s in r["steps"]], style, label=label)
        axes[2].plot(steps, [s["domain_loss"] for s in r["steps"]], style, label=label)
    for ax, title in zip(axes, ("mean feature norm", "classification loss", "domain loss")):
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "stability.png", dpi=200, bbox_inches="tight")


def replot() -> None:
    """Redraw stability.png from the saved steps.json without rerunning training."""
    import json
    plot(json.load(open(OUT / "steps.json")))


def main():
    os.environ["PA1_OUT_ROOT"] = tempfile.mkdtemp()  # the short runs' checkpoints are not kept
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_one(m, n) for m, n in RUNS]
    save_json(results, OUT / "steps.json")
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "max_feature_norm", "final_feature_norm", "max_cls_loss", "max_domain_loss",
                    "final_domain_acc", "source_val_macro_f1"])
        for r in results:
            s = r["steps"]
            w.writerow([r["run"], max(x["feature_norm"] for x in s), s[-1]["feature_norm"], max(x["cls_loss"] for x in s),
                        max(x["domain_loss"] for x in s), s[-1]["domain_acc"], r["source_val_macro_f1"]])
    plot(results)
    print(open(OUT / "summary.csv").read())


if __name__ == "__main__":
    main()
