"""Task 3 final evaluation: the ONLY Task 3 script allowed to load Sketch. Run after all decisions are fixed.

    python -m task3.evaluate_sketch

Runs are evaluated by name. "erm" is the Task 2 Source-only checkpoint; every other name is a Task 3 run
trained with task3.train (checkpoints/task3/<run>/best.pt). Any subset can be evaluated on its own, e.g. one
design study, into its own folder so the main results are never overwritten:

    python -m task3.evaluate_sketch --runs erm dan_dg_lam0.1 dan_dg dan_dg_lam10 --out-name design_study_dan_dg
"""
import argparse
import csv

from common.device import get_device, make_loader
from common.io import PROJECT_ROOT, data_root, load_json, results_dir, save_json
from common.metrics import classification_metrics
from shared.pacs import CLASSES, NUM_CLASSES
from shared.pacs_eval import evaluate_sources, predict, val_loaders
from shared.pacs_models import load_pacs_checkpoint
from shared.checkpoints import checkpoint_file
from shared.pacs_protocol import eval_transform, load_splits, source_datasets, target_dataset
from task2.evaluation.class_analysis import per_class_changes
from task3.evaluation.sharpness import fixed_validation_batch, sharpness_proxy
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.methods.erm import load_erm

DEFAULT_RUNS = ("erm", "dan_dg", "sam")


def _checkpoint_path(run: str):
    """ERM is the Task 2 Source-only checkpoint; every other run lives under checkpoints/task3.
    Builds the path only: checking a mistyped run name must not create a folder for it."""
    if run == "erm":
        return checkpoint_file("task2", "source_only")
    return checkpoint_file("task3", run)


def _load_model(run: str, device):
    if run == "erm":
        return load_erm(device)[0]
    return load_pacs_checkpoint(_checkpoint_path(run), device)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--baseline", default="erm", help="Run everything is compared against (default: erm).")
    parser.add_argument("--out-name", default="final_eval",
                        help="Results go to task3/results/<out-name>/. Use a different name per study.")
    parser.add_argument("--eval-batch", type=int, default=128)
    args = parser.parse_args()

    if args.baseline not in args.runs:
        raise ValueError(f"--baseline {args.baseline!r} is not in --runs {args.runs!r}.")

    # Step 1: every checkpoint must exist BEFORE anything else happens, Sketch included.
    for run in args.runs:
        if not _checkpoint_path(run).exists():
            raise FileNotFoundError(f"Missing checkpoint for run {run!r}: {_checkpoint_path(run)}.")

    device = get_device()
    pacs_root = data_root() / "pacs"
    out_dir = results_dir("task3", args.out_name)
    splits = load_splits()
    val_datasets = source_datasets(pacs_root, "val", eval_transform(), splits)
    val = val_loaders(val_datasets, args.eval_batch, device)

    # Step 2: SOURCE SIDE ONLY (no Sketch is loaded yet): per-domain, mean and worst source metrics,
    # source-domain separability, and the sharpness proxy on the fixed 32-per-domain validation batch.
    batch_x, batch_y, batch_indices = fixed_validation_batch(val_datasets)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    save_json(batch_indices, out_dir / "sharpness_batch_indices.json")

    results = {}
    for run in args.runs:
        model = _load_model(run, device)
        results[run] = {
            "source_val": evaluate_sources(model, val, device),
            "source_domain_separability": source_domain_separability(model, val, device),
            "sharpness": sharpness_proxy(model, batch_x, batch_y),
        }
        r = results[run]
        print(f"{run}: mean_src_F1={r['source_val']['mean_macro_f1']:.4f} worst_src_F1={r['source_val']['worst_macro_f1']:.4f} "
              f"src_domain_sep={r['source_domain_separability']:.4f} sharpness={r['sharpness']:.4f}")

    # Written to disk BEFORE Sketch is touched, so the source-side story can't be revised after seeing Sketch.
    save_json(results, out_dir / "source_side.json")

    # Step 3: only now load Sketch, and score every run on it.
    sketch_ds = target_dataset(pacs_root, eval_transform(), purpose="task3_final_eval")
    sketch_loader = make_loader(sketch_ds, args.eval_batch, device)
    sketch_predictions, sketch_y_true = {}, None
    for run in args.runs:
        model = _load_model(run, device)
        out = predict(model, sketch_loader, device)
        results[run]["sketch"] = classification_metrics(out["y"], out["pred"], NUM_CLASSES)
        sketch_predictions[run] = out["pred"]
        if sketch_y_true is None:
            sketch_y_true = out["y"]
        print(f"{run}: sketch_accuracy={results[run]['sketch']['accuracy']:.4f} sketch_macro_f1={results[run]['sketch']['macro_f1']:.4f}")

    # Sketch accuracy change versus the baseline, and per-class changes + confusions for every other run.
    baseline_acc = results[args.baseline]["sketch"]["accuracy"]
    for run in args.runs:
        results[run]["sketch"]["accuracy_change_vs_baseline"] = results[run]["sketch"]["accuracy"] - baseline_acc
        if run != args.baseline:
            results[run]["class_analysis"] = per_class_changes(
                sketch_y_true, sketch_predictions[run], sketch_predictions[args.baseline], CLASSES,
            )

    # Step 4: put the Task 2 numbers next to these ones (Source-only should equal ERM exactly, since ERM is that checkpoint).
    task2_path = PROJECT_ROOT / "task2" / "results" / "final_eval" / "final_eval.json"
    if task2_path.exists():
        task2 = load_json(task2_path)
        results["task2_reference"] = {name: task2[name] for name in ("source_only", "dan", "dann", "cdan") if name in task2}
        if "erm" in results and "source_only" in task2:
            same = abs(results["erm"]["sketch"]["accuracy"] - task2["source_only"]["sketch"]["accuracy"]) < 1e-9
            print(f"ERM Sketch accuracy equals Task 2 Source-only: {same}")

    # Step 5: save everything.
    save_json(results, out_dir / "final_eval.json")
    _save_summary_csv(results, args.runs, out_dir / "final_eval_summary.csv")
    print(f"Saved {out_dir / 'final_eval.json'} and {out_dir / 'final_eval_summary.csv'}")


def _save_summary_csv(results: dict, runs, path) -> None:
    """One row per run: every number the Task 3 comparison table needs."""
    domains = list(results[runs[0]]["source_val"]["per_domain"].keys())
    header = ["run"]
    for d in domains:
        header += [f"{d}_acc", f"{d}_macro_f1"]
    header += ["mean_source_acc", "mean_source_macro_f1", "worst_source_acc", "worst_source_macro_f1",
               "sketch_acc", "sketch_macro_f1", "sketch_acc_change_vs_baseline",
               "source_domain_separability", "sharpness"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for run in runs:
            r = results[run]
            row = [run]
            for d in domains:
                row += [r["source_val"]["per_domain"][d]["accuracy"], r["source_val"]["per_domain"][d]["macro_f1"]]
            row += [r["source_val"]["mean_accuracy"], r["source_val"]["mean_macro_f1"],
                    r["source_val"]["worst_accuracy"], r["source_val"]["worst_macro_f1"],
                    r["sketch"]["accuracy"], r["sketch"]["macro_f1"], r["sketch"]["accuracy_change_vs_baseline"],
                    r["source_domain_separability"], r["sharpness"]]
            writer.writerow(row)


if __name__ == "__main__":
    main()
