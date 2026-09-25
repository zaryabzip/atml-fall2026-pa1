"""Task 1 end-to-end driver.

    python -m task1.scripts.run_task1 --config task1/configs/default.yaml

Expects `task1.data.make_subset` and `task1.data.make_cue_conflicts` to have already been run
once, as described in the README.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common.config import config_arg_parser, load_config, save_config
from common.device import get_device, make_loader
from common.io import PROJECT_ROOT, load_json, save_json
from common.plotting import save_figure
from common.seed import seed_everything
from task1.analysis.evaluate_bias import prediction_report, shape_bias
from task1.analysis.feature_similarity import cosine_stability
from task1.analysis.representation import joint_projection
from task1.analysis.train_heads import train_linear_head
from task1.data.stl10 import STL10_CLASSES, SPLIT_FILE, common_transform, load_stl10
from task1.data.transforms import extra_color, patch_shuffle, to_grayscale, translate
from task1.models.backbones import BACKBONES, clip_zero_shot_text, extract_features, load_backbone

RESULTS_DIR = PROJECT_ROOT / "task1" / "results"  # everything this script produces goes under here
CACHE_DIR = PROJECT_ROOT / "task1" / "cache"  # cached frozen features, so re-runs are fast
CUE_CONFLICT_DIR = PROJECT_ROOT / "task1" / "data" / "cue_conflicts"  # written by make_cue_conflicts.py
CUE_CONFLICT_META = CUE_CONFLICT_DIR / "metadata.json"  # one record per cue-conflict image


# ---------------------------------------------------------------------------
# Small helpers shared by every intervention below
# ---------------------------------------------------------------------------

def load_split_tensors(split: str, indices, batch_size: int) -> tuple:
    """Load the common 224x224 [0, 1] tensors + labels for `indices` of an STL-10 split into memory."""
    # Build a loader over just the requested indices, keeping their original order, and pull
    # every batch out of it into memory.
    dataset = load_stl10(split, transform=common_transform())
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    image_batches, label_batches = [], []
    for batch_images, batch_labels in loader:
        image_batches.append(batch_images)
        label_batches.append(torch.as_tensor(batch_labels))

    images = torch.cat(image_batches)  # (N, 3, 224, 224)
    labels = torch.cat(label_batches).numpy()  # (N,)
    return images, labels


@torch.no_grad()
def features_from_tensor(backbone, x: torch.Tensor, device, batch_size: int) -> np.ndarray:
    """Run a frozen backbone over an in-memory tensor of images, batched to bound GPU/MPS memory."""
    feature_batches = []
    for start in range(0, len(x), batch_size):
        chunk = x[start:start + batch_size].to(device, non_blocking=True)
        feature_batches.append(backbone(chunk).float().cpu())
    return torch.cat(feature_batches).numpy()  # (N, feature_dim)


def head_probabilities(head, feats: np.ndarray, device) -> np.ndarray:
    """Softmax probabilities from a trained linear head (nn.Linear on frozen features)."""
    head = head.to(device)
    head.eval()
    feats_tensor = torch.from_numpy(feats).float().to(device)
    with torch.no_grad():
        probs = F.softmax(head(feats_tensor), dim=1)
    return probs.cpu().numpy()


def evaluate_head(head, feats: np.ndarray, y_true: np.ndarray, device, pred_clean=None) -> tuple:
    """Predict with the linear head and score it. Returns (report, predictions, probabilities)."""
    probs = head_probabilities(head, feats, device)
    pred = probs.argmax(axis=1)
    report = prediction_report(y_true, pred, probs, pred_clean=pred_clean)
    return report, pred, probs


def cache_split_features(backbone, backbone_name: str, device, feature_batch: int) -> dict:
    """Cache train/val frozen features for one backbone, so re-runs skip the forward pass."""
    split_ids = load_json(SPLIT_FILE)  # {"train": [...], "val": [...], "eval_subset": [...], ...}

    # For each of train/val: reuse the cached .npz if one already exists, otherwise run the
    # backbone over that split's images once and save the result for next time.
    cached = {}
    splits_to_cache = [("train", "train", split_ids["train"]), ("val", "train", split_ids["val"])]
    for cache_name, source_split, indices in splits_to_cache:
        cache_path = CACHE_DIR / f"{backbone_name}_{cache_name}.npz"

        if cache_path.exists():
            cached_data = np.load(cache_path)
            cached[cache_name] = (cached_data["features"], cached_data["labels"])
            continue

        dataset = load_stl10(source_split, transform=common_transform())
        subset = Subset(dataset, indices)
        loader = make_loader(subset, feature_batch, device)
        feats, labels = extract_features(backbone, loader, device)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=feats, labels=labels)
        cached[cache_name] = (feats, labels)

    return cached  # {"train": (feats, labels), "val": (feats, labels)}


# ---------------------------------------------------------------------------
# Per-intervention evaluation blocks
# ---------------------------------------------------------------------------

def run_translation_curve(backbone, head, eval_x, eval_y, clean_pred, clean_feats, shifts, device, feature_batch):
    """Accuracy at each shift magnitude, averaged over the 4 cardinal directions. Also returns the
    cosine stability at the largest configured shift, averaged the same way, and one example
    feature set at the largest shift (for the representation plot)."""
    max_shift = max(shifts) if shifts else 0
    curve = []  # one entry per shift magnitude
    max_shift_feats = []  # features at the largest shift, collected across its 4 directions

    for shift in shifts:
        if shift == 0:  # a 0px "shift" is just the clean image again, so reuse work already done
            clean_accuracy = float((clean_pred == eval_y).mean())
            curve.append({"shift_px": 0, "accuracy": clean_accuracy, "consistency": 1.0, "cosine_stability": 1.0,
                          "n_directions": 1})
            continue

        # Test this shift magnitude in all 4 cardinal directions, averaging the accuracy.
        directions = [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]  # right, left, down, up
        direction_accuracies, direction_consistencies, direction_cosines = [], [], []
        for dx, dy in directions:
            shifted_x = translate(eval_x, dx, dy)
            feats = features_from_tensor(backbone, shifted_x, device, feature_batch)
            pred = head_probabilities(head, feats, device).argmax(axis=1)
            direction_accuracies.append(float((pred == eval_y).mean()))
            direction_consistencies.append(float((pred == clean_pred).mean()))  # prediction unchanged vs clean
            direction_cosines.append(cosine_stability(clean_feats, feats))
            if shift == max_shift:  # keep features at the largest shift for cosine stability + the plot
                max_shift_feats.append(feats)

        curve.append({
            "shift_px": shift,
            "accuracy": float(np.mean(direction_accuracies)),
            "per_direction_accuracy": direction_accuracies,
            "consistency": float(np.mean(direction_consistencies)),
            "per_direction_consistency": direction_consistencies,
            "cosine_stability": float(np.mean(direction_cosines)),
        })

    # Cosine stability at the largest shift, averaged over its 4 directions.
    cosine_at_max_shift, example_feats = None, None
    if max_shift_feats:
        per_direction_cosine = [cosine_stability(clean_feats, feats) for feats in max_shift_feats]
        cosine_at_max_shift = float(np.mean(per_direction_cosine))
        example_feats = max_shift_feats[0]  # just need one direction's features for the representation plot

    return curve, cosine_at_max_shift, example_feats


def run_cue_conflict_eval(backbone, head, device, feature_batch, classes):
    """Evaluate accepted cue-conflict images, if `make_cue_conflicts.py` has already produced them."""
    if not CUE_CONFLICT_META.exists():
        print(f"Skipping cue conflicts: {CUE_CONFLICT_META} not found (run task1.data.make_cue_conflicts first).")
        return None

    from PIL import Image

    all_meta = load_json(CUE_CONFLICT_META)
    meta = [m for m in all_meta if m.get("accepted")]  # keep only the ones that passed the rejection rule
    if not meta:
        print("Skipping cue conflicts: no accepted images in metadata.")
        return None

    # Load every accepted cue-conflict image, and its shape (content) / texture (style) labels.
    name_to_idx = {name: i for i, name in enumerate(classes)}
    transform = common_transform()
    image_tensors, content_labels, style_labels = [], [], []
    for record in meta:
        image_path = CUE_CONFLICT_DIR / "images" / record["path"]
        image_tensors.append(transform(Image.open(image_path).convert("RGB")))
        content_labels.append(name_to_idx[record["content_class"]])
        style_labels.append(name_to_idx[record["style_class"]])
    images = torch.stack(image_tensors)
    content_labels = np.array(content_labels)
    style_labels = np.array(style_labels)

    # Run the backbone + head on them, and score with the shape/texture/other breakdown.
    feats = features_from_tensor(backbone, images, device, feature_batch)
    probs = head_probabilities(head, feats, device)
    pred = probs.argmax(axis=1)
    bias = shape_bias(pred, content_labels, style_labels)
    bias["mean_max_confidence"] = float(probs.max(axis=1).mean())

    # If the source (un-stylized) image ids were recorded, also measure how far the representation
    # moved from that original content photo.
    has_source_ids = all("source_image_id" in m for m in meta)
    content_feats = None
    if has_source_ids:
        source_ids = [m["source_image_id"] for m in meta]
        content_x, _ = load_split_tensors("test", source_ids, feature_batch)
        content_feats = features_from_tensor(backbone, content_x, device, feature_batch)
        bias["cosine_stability_vs_content_source"] = cosine_stability(content_feats, feats)

    paths = [m["path"] for m in meta]
    # scores, features, shape/texture labels, head predictions, source-photo features, image file names
    return bias, feats, content_labels, style_labels, pred, content_feats, paths


# ---------------------------------------------------------------------------
# Representation plot
# ---------------------------------------------------------------------------

PLOT_PALETTE = (  # a fixed 10-color qualitative palette, used consistently across every figure
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)


def save_projection_plot(proj: dict, path, title: str, class_names) -> None:
    """Scatter the joint projection: color = true (shape, for cue conflicts) class, marker = clean
    (circle) vs transformed (x). Every class gets its own color AND a labeled legend entry --
    without that legend, a color-by-class plot is unreadable (which was the bug here before)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    embedding = proj["embedding"]  # (2N, 2) x/y coordinates to scatter
    classes = proj["class_labels"]  # (2N,) which class each point belongs to (controls color)
    domain = proj["domain"]  # (2N,) 0 = clean point, 1 = transformed point (controls marker shape)
    point_colors = [PLOT_PALETTE[c % len(PLOT_PALETTE)] for c in classes]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor("white")

    # Clean points as filled circles, transformed points as bold x's, both colored by class.
    clean_mask = domain == 0
    ax.scatter(embedding[clean_mask, 0], embedding[clean_mask, 1],
               c=[point_colors[i] for i in range(len(classes)) if clean_mask[i]],
               marker="o", s=32, alpha=0.85, linewidths=0.4, edgecolors="white", zorder=2)
    transformed_mask = domain == 1
    ax.scatter(embedding[transformed_mask, 0], embedding[transformed_mask, 1],
               c=[point_colors[i] for i in range(len(classes)) if transformed_mask[i]],
               marker="x", s=36, alpha=0.85, linewidths=1.4, zorder=3)

    # Two separate legends: which color is which class, and which marker is clean vs transformed.
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PLOT_PALETTE[i % len(PLOT_PALETTE)],
               markeredgecolor="none", markersize=7, label=name)
        for i, name in enumerate(class_names)
    ]
    class_legend = ax.legend(handles=class_handles, title="class", bbox_to_anchor=(1.02, 1),
                             loc="upper left", fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(class_legend)  # a second legend would otherwise replace this one
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#555555", markeredgecolor="white",
               markersize=8, label="clean"),
        Line2D([0], [0], marker="x", linestyle="", color="#555555", markersize=8, markeredgewidth=1.4,
               label="transformed"),
    ]
    ax.legend(handles=marker_handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.9)

    # Absolute t-SNE/UMAP coordinates aren't meaningful (the PDF: "don't compare coordinates across
    # projections"), so the numeric ticks would only invite over-interpretation -- hide them.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # NOTE: no fig.tight_layout() here -- it would compute the layout before knowing the
    # class_legend needs room outside the axes, and then bbox_inches="tight" at save time would
    # crop that legend clean out of the PNG. Passing extra_artists=[class_legend] below is what
    # actually reserves it space (verified: without this, the class-color legend was silently
    # missing from every saved plot, even though matplotlib had it in memory).
    save_figure(fig, path, dpi=220, extra_artists=[class_legend])


def save_translation_curve_plot(all_reports: dict, path) -> None:
    """One line per backbone: accuracy (y) against pixel shift (x). This is a required plot,
    not just numbers in the JSON, so every backbone's translation robustness is visible at a glance."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    markers = ("o", "s", "^")  # a different marker shape per backbone, in addition to color

    for ax, key, ylabel in zip(axes, ("accuracy", "consistency"), ("accuracy (%)", "prediction consistency (%)")):
        for i, (backbone_name, report) in enumerate(all_reports.items()):
            curve = report["translation_curve"]
            shifts = [point["shift_px"] for point in curve]
            values = [100.0 * point[key] for point in curve]
            ax.plot(shifts, values, marker=markers[i % len(markers)], markersize=7, linewidth=2.2,
                    color=PLOT_PALETTE[i % len(PLOT_PALETTE)], label=backbone_name)
        ax.set_xlabel("translation (pixels)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=True, framealpha=0.9, fontsize=10)
    fig.suptitle("Translation, averaged over 4 directions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, dpi=220)


def save_projection(backbone_name, transform_name, clean_feats, transformed_feats, labels, method, seed, settings,
                    class_names):
    """Fit, plot and save one joint t-SNE/UMAP projection (clean vs one transform). `labels` must
    already have the same length as `transformed_feats` -- for cue conflicts that means passing the
    per-image shape (content) class, not the 500-image eval_y (a silent length-mismatch fallback used
    to paint every cue-conflict point the same color; that bug is why this assertion exists now)."""
    assert len(transformed_feats) == len(labels), (
        f"save_projection({transform_name}): {len(transformed_feats)} feature rows but "
        f"{len(labels)} labels -- pass the labels that actually match these images."
    )
    assert len(clean_feats) == len(transformed_feats), f"save_projection({transform_name}): clean/transformed rows differ"
    matched_clean_feats = clean_feats

    # Fit the projection, then save both the plot and the raw coordinates/settings.
    proj = joint_projection(matched_clean_feats, transformed_feats, labels, method, seed, **settings)
    plot_path = RESULTS_DIR / backbone_name / f"projection_{transform_name}.png"
    plot_title = f"{backbone_name}: clean vs {transform_name} ({method})"
    save_projection_plot(proj, plot_path, plot_title, class_names)

    proj_to_save = dict(proj)
    proj_to_save["embedding"] = proj["embedding"].tolist()  # numpy arrays aren't JSON-serializable directly
    json_path = RESULTS_DIR / backbone_name / f"projection_{transform_name}.json"
    save_json(proj_to_save, json_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_clip_zero_shot(backbone, cfg, device, feature_batch, eval_x, eval_y, clean_feats) -> dict:
    """CLIP-only baseline: classify by comparing image features to text prompts, no linear head."""
    # Embed every class name as a text prompt, then classify each image by its closest prompt.
    template = cfg["clip_zero_shot"]["template"]  # e.g. "a photo of a {}."
    text_feats, logit_scale = clip_zero_shot_text(backbone, STL10_CLASSES, template, device)
    image_feats = torch.from_numpy(clean_feats).to(device)
    similarities = image_feats @ text_feats.T.to(device)
    logits = logit_scale.item() * similarities  # CLIP's learned temperature scaling

    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    return prediction_report(eval_y, pred, probs)


def compare_zero_shot_and_head(backbone, cfg, device, eval_y, conditions: dict, cue) -> dict:
    """For each condition: zero-shot accuracy / consistency (vs zero-shot on clean) and how often zero-shot and the
    trained head predict the same class. On cue conflicts: zero-shot shape bias and coverage."""
    text_feats, logit_scale = clip_zero_shot_text(backbone, STL10_CLASSES, cfg["clip_zero_shot"]["template"], device)

    def zero_shot_probs(feats):
        logits = logit_scale.item() * (torch.from_numpy(feats).to(device) @ text_feats.T.to(device))
        return F.softmax(logits, dim=1).cpu().numpy()

    clean_zs_pred = zero_shot_probs(conditions["clean"][0]).argmax(axis=1)
    out = {}
    for name, (feats, head_pred) in conditions.items():
        probs = zero_shot_probs(feats)
        pred = probs.argmax(axis=1)
        out[name] = prediction_report(eval_y, pred, probs, pred_clean=None if name == "clean" else clean_zs_pred)
        out[name]["agreement_with_head"] = float((pred == np.asarray(head_pred)).mean())
    if cue is not None:
        cue_feats, content_labels, style_labels, head_pred = cue
        pred = zero_shot_probs(cue_feats).argmax(axis=1)
        out["cue_conflict"] = shape_bias(pred, content_labels, style_labels)
        out["cue_conflict"]["agreement_with_head"] = float((pred == np.asarray(head_pred)).mean())
    return out


def run_backbone(backbone_name: str, cfg: dict, device) -> dict:
    """Run every Task 1 experiment for one backbone (e.g. resnet50) and save its report."""
    print(f"=== {backbone_name} ===")
    backbone = load_backbone(backbone_name).to(device)
    split_ids = load_json(SPLIT_FILE)  # train/val/eval_subset indices, all from seed 6304
    feature_batch = cfg["feature_batch"]

    # Cache (or reuse cached) frozen features for the linear head's training/validation data.
    cached = cache_split_features(backbone, backbone_name, device, feature_batch)
    train_feats, train_y = cached["train"]
    val_feats, val_y = cached["val"]
    report = {"backbone": backbone_name}

    # Extract clean features for the 500-image eval subset -- reused by every intervention below.
    eval_x, eval_y = load_split_tensors("test", split_ids["eval_subset"], feature_batch)
    clean_feats = features_from_tensor(backbone, eval_x, device, feature_batch)
    if backbone_name == "clip_b32":  # CLIP also gets a zero-shot baseline, in addition to its trained head
        report["clip_zero_shot"] = run_clip_zero_shot(backbone, cfg, device, feature_batch, eval_x, eval_y, clean_feats)

    # Train the linear head, then evaluate the clean baseline every later intervention compares against.
    head, history = train_linear_head(train_feats, train_y, val_feats, val_y, cfg["head"], device)
    report["head_training_history"] = history
    clean_report, clean_pred, _ = evaluate_head(head, clean_feats, eval_y, device)
    report["clean"] = clean_report

    # Grayscale: removes all color, keeps shape and brightness.
    gray_x = to_grayscale(eval_x)
    gray_feats = features_from_tensor(backbone, gray_x, device, feature_batch)
    gray_report, gray_pred, _ = evaluate_head(head, gray_feats, eval_y, device, pred_clean=clean_pred)
    report["grayscale"] = gray_report
    cosine_gray = cosine_stability(clean_feats, gray_feats)

    # Extra color: the chosen design-choice color change (default hue rotation).
    color_cfg = dict(cfg["interventions"]["extra_color"])  # e.g. {"kind": "hue_rotation", "hue_factor": 0.3}
    color_kind = color_cfg.pop("kind")
    color_x = extra_color(eval_x, kind=color_kind, **color_cfg)
    color_feats = features_from_tensor(backbone, color_x, device, feature_batch)
    color_report, color_pred, _ = evaluate_head(head, color_feats, eval_y, device, pred_clean=clean_pred)
    report["extra_color"] = color_report
    cosine_color = cosine_stability(clean_feats, color_feats)

    # Patch shuffle: destroys global layout, keeps local pixel evidence. The permutations get
    # saved to disk so this exact shuffle is reproducible.
    grid = cfg["interventions"]["patch_grid"]
    generator = torch.Generator().manual_seed(cfg["seed"])
    shuffled_x, perms = patch_shuffle(eval_x, grid, generator)
    shuffled_feats = features_from_tensor(backbone, shuffled_x, device, feature_batch)
    patch_report, patch_pred, _ = evaluate_head(head, shuffled_feats, eval_y, device, pred_clean=clean_pred)
    report["patch_shuffle"] = patch_report
    cosine_patch = cosine_stability(clean_feats, shuffled_feats)
    save_json(perms.tolist(), RESULTS_DIR / backbone_name / "patch_shuffle_permutations.json")

    # Translation: shifts the object around the frame, at several pixel amounts.
    shifts = cfg["interventions"]["translations"]
    curve, cosine_translation, translation_example_feats = run_translation_curve(
        backbone, head, eval_x, eval_y, clean_pred, clean_feats, shifts, device, feature_batch,
    )
    report["translation_curve"] = curve

    # Cue conflicts: shape says one class, texture says another (skipped if not generated yet).
    cue_result = run_cue_conflict_eval(backbone, head, device, feature_batch, STL10_CLASSES)
    cue_feats, cue_content_labels = None, None
    if cue_result is not None:
        cue_report, cue_feats, cue_content_labels, cue_style_labels, cue_pred, cue_source_feats, cue_paths = cue_result
        report["cue_conflict"] = cue_report

    # CLIP only: zero-shot decisions vs the trained head, on the same images under every intervention.
    if backbone_name == "clip_b32":
        conditions = {"clean": (clean_feats, clean_pred), "grayscale": (gray_feats, gray_pred),
                      "extra_color": (color_feats, color_pred), "patch_shuffle": (shuffled_feats, patch_pred)}
        cue = (cue_feats, cue_content_labels, cue_style_labels, cue_pred) if cue_result is not None else None
        report["clip_zero_shot_vs_head"] = compare_zero_shot_and_head(backbone, cfg, device, eval_y, conditions, cue)

    # Collect every intervention's cosine stability into one place in the report.
    cue_cosine = report.get("cue_conflict", {}).get("cosine_stability_vs_content_source")
    report["cosine_stability"] = {
        "grayscale": cosine_gray,
        "extra_color": cosine_color,
        "patch_shuffle": cosine_patch,
        "translation": cosine_translation,
        "cue_conflict": cue_cosine,
    }

    # t-SNE / UMAP projections: clean vs each required transform, one plot per transform.
    rep_cfg = cfg["representation"]
    method = rep_cfg["method"]  # "tsne" or "umap"
    if method == "tsne":
        settings = {"perplexity": rep_cfg["tsne_perplexity"]}
    else:
        settings = {"n_neighbors": rep_cfg["umap_neighbors"], "min_dist": rep_cfg["umap_min_dist"]}

    save_projection(backbone_name, "grayscale", clean_feats, gray_feats, eval_y, method, cfg["seed"], settings, STL10_CLASSES)
    save_projection(backbone_name, "patch_shuffle", clean_feats, shuffled_feats, eval_y, method, cfg["seed"], settings, STL10_CLASSES)
    if translation_example_feats is not None:  # only plot translation if we actually tested a nonzero shift
        save_projection(backbone_name, "translation", clean_feats, translation_example_feats, eval_y, method, cfg["seed"], settings, STL10_CLASSES)
    if cue_feats is not None:  # only plot cue conflicts if they've been generated
        # colored by SHAPE class (content_class), not the eval-subset's true class -- cue-conflict
        # images are a different, smaller set of images with their own (shape, texture) labels.
        # the clean half is each cue conflict's own source (content) photo, so both halves are the same images
        save_projection(backbone_name, "cue_conflict", cue_source_feats, cue_feats, cue_content_labels, method, cfg["seed"], settings, STL10_CLASSES)

    # Per-image predictions (class indices), so agreements, mismatches and failures can be inspected per image.
    predictions = {"eval_subset_ids": split_ids["eval_subset"], "true": eval_y.tolist(), "clean": clean_pred.tolist(),
                   "grayscale": gray_pred.tolist(), "extra_color": color_pred.tolist(), "patch_shuffle": patch_pred.tolist()}
    if cue_result is not None:
        predictions["cue_conflict"] = {"path": cue_paths, "shape": cue_content_labels.tolist(),
                                       "texture": cue_style_labels.tolist(), "pred": cue_pred.tolist()}
    save_json(predictions, RESULTS_DIR / backbone_name / "predictions.json")

    # Save everything computed above to disk.
    report_path = RESULTS_DIR / backbone_name / "report.json"
    save_json(report, report_path)
    print(f"Saved {report_path}")
    return report


def save_cue_conflict_examples(backbones, path, n_per_group: int = 4) -> None:
    """Informative cue-conflict cases with every model's prediction: (1) models disagree, (2) every model follows the
    texture, (3) every model predicts neither class. Picked after evaluation, for analysis only; written to JSON too."""
    import matplotlib.pyplot as plt
    from PIL import Image

    preds = {b: load_json(RESULTS_DIR / b / "predictions.json").get("cue_conflict") for b in backbones}
    if any(p is None for p in preds.values()):
        return
    first = preds[backbones[0]]
    n = len(first["path"])

    def outcome(b, i):
        p = preds[b]["pred"][i]
        return "shape" if p == first["shape"][i] else "texture" if p == first["texture"][i] else "other"

    groups = {"models disagree": [], "all follow texture": [], "all predict neither": []}
    for i in range(n):
        outs = {outcome(b, i) for b in backbones}
        if len(outs) > 1:
            groups["models disagree"].append(i)
        elif outs == {"texture"}:
            groups["all follow texture"].append(i)
        elif outs == {"other"}:
            groups["all predict neither"].append(i)
    # Spread examples over directions (shape/texture pairs); among disagreements prefer ones where a model follows texture.
    def spread(idx, k):
        idx = sorted(idx, key=lambda i: not any(outcome(b, i) == "texture" for b in backbones))
        picked, seen = [], set()
        for i in idx:
            d = (first["shape"][i], first["texture"][i])
            if d not in seen:
                seen.add(d); picked.append(i)
            if len(picked) == k:
                break
        return picked

    chosen = [(g, i) for g, idx in groups.items() for i in spread(idx, n_per_group)]
    save_json({"counts": {g: len(idx) for g, idx in groups.items()},
               "examples": [{"group": g, "path": first["path"][i], "shape": STL10_CLASSES[first["shape"][i]],
                             "texture": STL10_CLASSES[first["texture"][i]],
                             **{b: STL10_CLASSES[preds[b]["pred"][i]] for b in backbones}} for g, i in chosen]},
              path.with_suffix(".json"))
    if not chosen:
        return
    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 4.3 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, (g, i) in zip(np.atleast_1d(axes).flat, chosen):
        ax.imshow(Image.open(CUE_CONFLICT_DIR / "images" / first["path"][i]))
        lines = [f"{g}", f"shape {STL10_CLASSES[first['shape'][i]]} / texture {STL10_CLASSES[first['texture'][i]]}"]
        lines += [f"{b}: {STL10_CLASSES[preds[b]['pred'][i]]}" for b in backbones]
        ax.set_title("\n".join(lines), fontsize=8)
    fig.tight_layout()
    save_figure(fig, path, dpi=160)


def main():
    # Load config, seed everything, and pick a device.
    args = config_arg_parser("Task 1").parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} not found. Run: python -m task1.data.make_subset")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_config(cfg, RESULTS_DIR / "config.yaml")  # record exactly which settings produced these results

    # Run every backbone in turn, then save a combined summary and the required translation-curve plot.
    all_reports = {}
    for backbone_name in cfg["backbones"]:
        if backbone_name not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone_name!r}; choose from {BACKBONES}")
        all_reports[backbone_name] = run_backbone(backbone_name, cfg, device)

    summary_path = RESULTS_DIR / "summary.json"
    save_json(all_reports, summary_path)

    save_cue_conflict_examples(list(all_reports), RESULTS_DIR / "cue_conflict_examples.png")

    curve_path = RESULTS_DIR / "translation_curve.png"
    save_translation_curve_plot(all_reports, curve_path)
    print(f"Saved {curve_path}")
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
