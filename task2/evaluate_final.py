"""Task 2 final evaluation. Run ONLY after every checkpoint and setting is locked.

    python -m task2.evaluate_final --runs source_only dan dann cdan

Any subset of runs can be evaluated on its own, e.g. one design study, without touching the main
results: give it its own output folder with --out-name (results go to task2/results/<out-name>/).

    python -m task2.evaluate_final --runs source_only dan_lmmd0.1 dan dan_lmmd10 \\
        --baseline source_only --out-name design_study_dan
"""
import argparse
import csv

from common.device import get_device, make_loader
from common.io import data_root, results_dir, save_json
from common.metrics import classification_metrics
from shared.pacs import CLASSES, NUM_CLASSES
from shared.checkpoints import checkpoint_file
from shared.pacs_eval import evaluate_sources, predict, val_loaders
from shared.pacs_models import load_pacs_checkpoint
from shared.pacs_protocol import eval_transform, load_splits, source_datasets, target_dataset
from task2.evaluation.class_analysis import per_class_changes
from task2.evaluation.domain_separability import domain_separability

RUNS = ("source_only", "dan", "dann", "cdan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--baseline", default="source_only",
                        help="Which --runs entry is the Source-only baseline everything else is compared "
                             "against. Must match a run_name exactly, e.g. --baseline dann_grl0.25 for a "
                             "renamed design-study run. Defaults to 'source_only'.")
    parser.add_argument("--out-name", default="final_eval",
                        help="Results go to task2/results/<out-name>/. Use a different name per study so "
                             "the main final_eval results are never overwritten.")
    parser.add_argument("--eval-batch", type=int, default=128)
    args = parser.parse_args()

    if args.baseline not in args.runs:  # fail loudly here, not with a confusing KeyError deep in the loop below
        raise ValueError(f"--baseline {args.baseline!r} is not in --runs {args.runs!r}.")

    # Step 1: every checkpoint must exist BEFORE we ever touch Sketch.
    ckpt_paths = {}
    for run in args.runs:
        path = checkpoint_file("task2", run)  # path only: a mistyped run name must not create a folder
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for run {run!r}: {path}. Train it first with task2.train.")
        ckpt_paths[run] = path

    # Shared setup: device, source-validation loaders, and the Sketch loader (loaded here, for the
    # first time in Task 2, only now that every checkpoint is fixed).
    device = get_device()
    pacs_root = data_root() / "pacs"
    splits = load_splits()
    val_datasets = source_datasets(pacs_root, "val", eval_transform(), splits)
    val = val_loaders(val_datasets, args.eval_batch, device)
    sketch_ds = target_dataset(pacs_root, eval_transform(), purpose="task2_final_eval")
    sketch_loader = make_loader(sketch_ds, args.eval_batch, device)

    # Steps 2-5: for each run, load its checkpoint and compute every metric on it.
    results = {}  # run -> {source_val, sketch, domain_separability}
    sketch_predictions = {}  # run -> predicted labels on Sketch, for class_analysis
    sketch_y_true = None  # ground truth is identical across runs (same loader, same order)
    for run in args.runs:
        model, _ = load_pacs_checkpoint(ckpt_paths[run], device)

        source_metrics = evaluate_sources(model, val, device)

        sketch_out = predict(model, sketch_loader, device, return_features=True)
        sketch_metrics = classification_metrics(sketch_out["y"], sketch_out["pred"], NUM_CLASSES)
        sketch_predictions[run] = sketch_out["pred"]
        if sketch_y_true is None:
            sketch_y_true = sketch_out["y"]

        separability = domain_separability(model, val, sketch_loader, device)

        results[run] = {
            "source_val": source_metrics,
            "sketch": sketch_metrics,
            "domain_separability": separability,
        }
        print(f"{run}: sketch_accuracy={sketch_metrics['accuracy']:.4f}  "
              f"mean_source_macro_f1={source_metrics['mean_macro_f1']:.4f}  separability={separability:.4f}")

    # Compare every run's Sketch accuracy against the baseline's.
    baseline_acc = results[args.baseline]["sketch"]["accuracy"]
    for run in args.runs:
        results[run]["sketch"]["accuracy_change_vs_baseline"] = results[run]["sketch"]["accuracy"] - baseline_acc

    # Step 6: per-class changes + confusions vs. the baseline, skipped for the baseline itself.
    for run in args.runs:
        if run == args.baseline:
            continue
        results[run]["class_analysis"] = per_class_changes(
            sketch_y_true, sketch_predictions[run], sketch_predictions[args.baseline], CLASSES,
        )

    # Step 7: save everything.
    out_dir = results_dir("task2", args.out_name)
    save_json(results, out_dir / "final_eval.json")
    _save_summary_csv(results, args.runs, out_dir / "final_eval_summary.csv")
    print(f"Saved {out_dir / 'final_eval.json'} and {out_dir / 'final_eval_summary.csv'}")


def _save_summary_csv(results: dict, runs, path) -> None:
    """One row per run: the numbers the report's main comparison table needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run", "mean_source_accuracy", "mean_source_macro_f1", "sketch_accuracy", "sketch_macro_f1",
                   "sketch_accuracy_change_vs_baseline", "domain_separability"]
        writer.writerow(header)
        for run in runs:
            r = results[run]
            writer.writerow([
                run,
                r["source_val"]["mean_accuracy"],
                r["source_val"]["mean_macro_f1"],
                r["sketch"]["accuracy"],
                r["sketch"]["macro_f1"],
                r["sketch"]["accuracy_change_vs_baseline"],
                r["domain_separability"],
            ])


if __name__ == "__main__":
    main()
