# Task 1: Inductive Biases and Feature Representations

Which visual cues (shape, texture, color, position) do ResNet-50, ViT-B/16 and CLIP rely on, and do their features change when their predictions don't?

## Setup
- Dataset: STL-10 (Hugging Face mirror `tanganke/stl10`; saved image ids are row indices in its parquet files). Stratified 80/20 train/val split of the official train set and a 500-image class-balanced test subset, all seed 6304.
- Backbones (frozen): ResNet-50 `IMAGENET1K_V2` (2048-d pooled), ViT-B/16 `IMAGENET1K_V1` (768-d class token), OpenCLIP ViT-B-32 `openai` (512-d normalized).
- One linear head per backbone: AdamW, lr 1e-3, wd 1e-4, max 50 epochs, early stop after 5. Plus CLIP zero-shot with "a photo of a {class}."
- Every intervention is applied to the same 224x224 RGB image in [0, 1]. Each model applies its own normalization afterwards.

## Run
```bash
python -m task1.data.make_subset          # once: splits + eval subset ids
python -m task1.data.make_cue_conflicts   # once: cue-conflict images
python -m task1.scripts.run_task1 --config task1/configs/default.yaml
```
Runs fine on a 16 GB M4 (MPS). If ViT runs out of memory, lower `feature_batch`.

## Files
| File | Role |
|---|---|
| `configs/default.yaml` | all settings, including the design choices below |
| `data/stl10.py` | STL-10 loading + common 224x224 transform |
| `data/make_subset.py` | splits and 500-image eval subset (saved to `data/splits/`) |
| `data/transforms.py` | grayscale, extra color, translation, patch shuffle |
| `data/make_cue_conflicts.py` | AdaIN shape/texture conflicts |
| `models/backbones.py` | frozen backbones, feature extraction, CLIP text features |
| `analysis/train_heads.py` | linear heads |
| `analysis/evaluate_bias.py` | accuracy, macro-F1, confidence, consistency, shape bias, coverage |
| `analysis/feature_similarity.py` | cosine representation stability |
| `analysis/representation.py` | t-SNE / UMAP |
| `scripts/run_task1.py` | runs everything in order |

## Design choices
- Extra color change: hue rotation (`hue_factor` 0.3).
- Cue conflicts: 5 class pairs (cat/truck, dog/airplane, bird/car, horse/ship, monkey/deer), both directions, AdaIN style strength 0.75, 30 candidates per direction. Bad stylizations are rejected by a fixed rule on the image itself (never on model predictions), documented at the top of `data/make_cue_conflicts.py`.
- Representation plots: t-SNE, perplexity 30, seed 6304, one joint projection per backbone.

## Outputs (`results/`)
- `summary.json` and `<backbone>/report.json`: clean / grayscale / extra color / patch shuffle (accuracy, macro-F1, confidence, change, consistency), translation curve, shape / texture / other counts with shape bias and coverage, cosine stability.
- `translation_curve.png`: accuracy at 0, 8, 16, 32 px shifts, averaged over 4 directions.
- `<backbone>/projection_{grayscale,patch_shuffle,translation,cue_conflict}.png`: joint clean-vs-transformed t-SNE plots.
- `<backbone>/patch_shuffle_permutations.json`: the exact patch permutation used for every image.
- `data/cue_conflicts/`: accepted cue-conflict images and their metadata.
