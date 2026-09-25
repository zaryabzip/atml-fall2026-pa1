"""Report-sized Task 1 figures, drawn from the saved results; nothing is recomputed.

    python -m task1.analysis.report_figures

Every figure is drawn at about its printed size (the report's text width is 5.5 in), with 8.5-10 pt fonts, so text
stays readable in the PDF. Writes to report/figures/:
  task1_stability.png      prediction consistency, the handout's cosine stability I_T, and I_T rescaled by each
                           model's different-class baseline
  task1_translation.png    accuracy, consistency and I_T per shift (+ CLIP zero-shot)
  task1_tsne_cue.png       cue-conflict t-SNE row;  task1_tsne_grid.png  4 interventions x 3 models
  task1_resnet_texture.png cue conflicts where ResNet picked the texture class
  task1_head_training.png  linear-head training curves
  task1_cue_examples.png   cue conflicts where the models disagree / all predict neither class
  cue_final_<k>.png        all 220 cue conflicts, one figure per class pair (one row per direction)
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from common.io import PROJECT_ROOT  # noqa: E402
from task1.data.stl10 import STL10_CLASSES  # noqa: E402

RES = PROJECT_ROOT / "task1" / "results"
FIG = PROJECT_ROOT / "report" / "figures"
CUE_DIR = PROJECT_ROOT / "task1" / "data" / "cue_conflicts"
MODELS = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "clip_b32": "CLIP ViT-B/32"}
SHORT = {"resnet50": "R", "vit_b16": "V", "clip_b32": "C"}
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "legend.fontsize": 8.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def _clean_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _name(x):
    return STL10_CLASSES[x] if isinstance(x, int) else x


# ---------------------------------------------------------------- t-SNE
def _tsne(ax, model, name, title):
    d = json.load(open(RES / model / f"projection_{name}.json"))
    e, c, dom = np.array(d["embedding"]), np.array(d["class_labels"]), np.array(d["domain"])
    for k in range(10):
        for dm, mk in ((0, "o"), (1, "x")):
            m = (c == k) & (dom == dm)
            ax.scatter(e[m, 0], e[m, 1], s=5 if dm == 0 else 7, marker=mk, color=PALETTE[k], alpha=0.75,
                       linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def _legend(fig, ncol=6, y=-0.01):
    handles = [Line2D([], [], marker="s", ls="", color=PALETTE[k], markersize=7, label=STL10_CLASSES[k]) for k in range(10)]
    handles += [Line2D([], [], marker="o", ls="", color="k", markersize=5, label="clean"),
                Line2D([], [], marker="x", ls="", color="k", markersize=6, label="transformed")]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               handletextpad=0.2, columnspacing=1.0)


def tsne_figures():
    rows = (("grayscale", "grayscale"), ("patch_shuffle", "patch shuffle"), ("translation", "translation 32 px"),
            ("cue_conflict", "cue conflict"))
    fig, axes = plt.subplots(4, 3, figsize=(6.6, 8.8))
    for r, (name, label) in enumerate(rows):
        for j, (m, mlabel) in enumerate(MODELS.items()):
            _tsne(axes[r, j], m, name, f"{mlabel}: {label}")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "task1_tsne_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for j, (m, mlabel) in enumerate(MODELS.items()):
        _tsne(axes[j], m, "cue_conflict", mlabel)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(FIG / "task1_tsne_cue.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- stability and translation
def cosine_baseline(model):
    """Mean cosine between val features of different classes (all pairs): what 'unrelated' looks like for this model."""
    d = np.load(PROJECT_ROOT / "task1" / "cache" / f"{model}_val.npz")
    f = d["features"] / np.linalg.norm(d["features"], axis=1, keepdims=True)
    y = d["labels"]
    sim = f @ f.T
    return float(sim[y[:, None] != y[None, :]].mean())


def stability_figure():
    """Consistency, the handout's cosine stability I_T (raw), and I_T rescaled by each model's different-class
    baseline (an extra: raw values can't be compared across models because their baselines differ)."""
    s = json.load(open(RES / "summary.json"))
    base = {m: cosine_baseline(m) for m in MODELS}
    print("cosine baselines (different-class pairs):", {m: round(b, 3) for m, b in base.items()})
    labels = ("grayscale", "hue", "shuffle", "shift 32 px", "cue conflict")
    keys = ("grayscale", "extra_color", "patch_shuffle", "translation", "cue_conflict")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    x = np.arange(len(labels)); w = 0.26
    for j, (m, mlabel) in enumerate(MODELS.items()):
        r = s[m]
        t32 = [t for t in r["translation_curve"] if t["shift_px"] == 32][0]
        cons = [r["grayscale"]["prediction_consistency"], r["extra_color"]["prediction_consistency"],
                r["patch_shuffle"]["prediction_consistency"], t32["consistency"], np.nan]
        raw = [r["cosine_stability"][k] for k in keys]
        # rescaled: 0 = as far apart as two images of different classes, 1 = unchanged
        resc = [(c - base[m]) / (1 - base[m]) for c in raw]
        for ax, vals in zip(axes, (cons, raw, resc)):
            ax.bar(x + (j - 1) * w, vals, w, color=PALETTE[j], label=mlabel)
    axes[0].set_title("Prediction consistency")
    axes[1].set_title("Cosine stability $I_T$")
    axes[2].set_title("$I_T$ rescaled (extra)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3); _clean_axes(ax)
    axes[0].text(x[-1], 0.03, "n/a", ha="center", fontsize=8.5, color="#666")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(FIG / "task1_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def translation_figure():
    s = json.load(open(RES / "summary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for key, ax, title in (("accuracy", axes[0], "Accuracy"), ("consistency", axes[1], "Prediction consistency"),
                           ("cosine_stability", axes[2], "Cosine stability $I_T$")):
        for j, (m, mlabel) in enumerate(MODELS.items()):
            c = s[m]["translation_curve"]
            ax.plot([t["shift_px"] for t in c], [t[key] for t in c], marker="o", lw=1.6, ms=4, color=PALETTE[j],
                    label=mlabel)
        ax.set_title(title); ax.set_xlabel("shift (px)"); ax.set_xticks([0, 8, 16, 32])
        ax.grid(alpha=0.3); _clean_axes(ax)
    zs = json.load(open(RES / "clip_zero_shot_translation.json"))["curve"]  # from zero_shot_translation.py
    for ax, key in ((axes[0], "zs_accuracy"), (axes[1], "zs_consistency")):
        ax.plot([t["shift_px"] for t in zs], [t[key] for t in zs], marker="s", lw=1.6, ms=3.5, ls="--",
                color=PALETTE[2], label="CLIP zero-shot")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    fig.savefig(FIG / "task1_translation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- cue-conflict images
def resnet_texture_figure():
    """Every cue conflict where ResNet-50 predicted the texture class, with ViT's and CLIP's (head) predictions."""
    P = {m: json.load(open(RES / m / "predictions.json"))["cue_conflict"] for m in MODELS}
    r = P["resnet50"]
    rows = [i for i in range(len(r["pred"])) if _name(r["pred"][i]) == _name(r["texture"][i])]
    ncol = 5; nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 1.72 * nrow))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, rows):
        ax.imshow(Image.open(CUE_DIR / "images" / r["path"][i]))
        ax.set_title(f"{_name(r['shape'][i])} / {_name(r['texture'][i])}\n"
                     f"V: {_name(P['vit_b16']['pred'][i])}, C: {_name(P['clip_b32']['pred'][i])}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "task1_resnet_texture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("ResNet texture picks:", len(rows))


def cue_examples_figure():
    """Cue conflicts on which the models disagree (top row) or all predict neither class (bottom row)."""
    ex = json.load(open(RES / "cue_conflict_examples.json"))["examples"]
    groups = [g for g in ("models disagree", "all follow texture", "all predict neither")
              if any(e["group"] == g for e in ex)]
    ncol = max(sum(e["group"] == g for e in ex) for g in groups)
    fig, axes = plt.subplots(len(groups), ncol, figsize=(6.6, 2.3 * len(groups)))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for r, g in enumerate(groups):
        for k, e in enumerate([e for e in ex if e["group"] == g]):
            ax = axes[r, k]
            ax.imshow(Image.open(CUE_DIR / "images" / e["path"]))
            ax.set_title(f"{e['shape']} / {e['texture']}\n"
                         f"R: {e['resnet50']}, V: {e['vit_b16']}\nC: {e['clip_b32']}", fontsize=8.5)
        axes[r, 0].set_ylabel(g, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "task1_cue_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def cue_grid_figures():
    """All kept cue conflicts, one figure per class pair; each row is one direction (shape class / texture class)."""
    meta = [m for m in json.load(open(CUE_DIR / "metadata.json")) if m.get("accepted")]
    pairs = []
    for m in meta:
        key = frozenset((m["content_class"], m["style_class"]))
        if key not in pairs:
            pairs.append(key)
    for k, pair in enumerate(pairs, 1):
        dirs = []
        for m in meta:
            d = (m["content_class"], m["style_class"])
            if frozenset(d) == pair and d not in dirs:
                dirs.append(d)
        per_dir = {d: [m for m in meta if (m["content_class"], m["style_class"]) == d] for d in dirs}
        ncol = 11
        nrow = sum(-(-len(v) // ncol) for v in per_dir.values())
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 0.66 * nrow + 0.1))
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        r = 0
        for d, items in per_dir.items():
            for start in range(0, len(items), ncol):
                for c, m in enumerate(items[start:start + ncol]):
                    axes[r, c].imshow(Image.open(CUE_DIR / "images" / m["path"]))
                if start == 0:
                    axes[r, 0].set_ylabel(f"{d[0]} shape\n{d[1]} texture", fontsize=8.5, rotation=0, ha="right",
                                          va="center")
                r += 1
        fig.subplots_adjust(left=0.13, right=1, top=1, bottom=0, wspace=0.04, hspace=0.06)
        fig.savefig(FIG / f"cue_final_{k}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("cue grids:", len(pairs))


# ---------------------------------------------------------------- linear heads
def head_training_figure():
    s = json.load(open(RES / "summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for j, (m, mlabel) in enumerate(MODELS.items()):
        h = s[m]["head_training_history"]
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["train_loss"] for e in h], marker="o", ms=3, lw=1.5, color=PALETTE[j], label=mlabel)
        axes[1].plot(ep, [e["val_accuracy"] for e in h], marker="o", ms=3, lw=1.5, color=PALETTE[j], label=mlabel)
    axes[0].set_title("Training loss"); axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); _clean_axes(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    fig.savefig(FIG / "task1_head_training.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    resnet_texture_figure()
    cue_examples_figure()
    cue_grid_figures()
    head_training_figure()
    tsne_figures()
    stability_figure()
    translation_figure()
    print("saved")
