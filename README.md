# PA1: Beyond IID (ATML Fall 2026)

Programming Assignment 1 for ATML (Fall 2026): how vision models behave beyond the IID setting. This repository contains Task 1.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Run every script from this folder as a module, e.g. `python -m task1.scripts.run_task1 --config task1/configs/default.yaml`.
Task 1 runs fine on a 16 GB Apple Silicon Mac (MPS); Kaggle is only faster.

## Paths
| Variable | Default | Meaning |
|---|---|---|
| `PA1_DATA_ROOT` | `./data` | datasets (`stl10/`) |
| `PA1_OUT_ROOT` | `.` | checkpoints root |

Small results (configs, JSON tables, plots) go to `task1/results/`.

## Data
- **STL-10**: downloaded automatically from the Hugging Face mirror `tanganke/stl10` (labeled train + test, 230 MB).

Splits (seed 6304), made once and saved:
```bash
python -m task1.data.make_subset      # STL-10 80/20 + 500-image eval subset
```

## Layout
```
common/    seed, device (CUDA/MPS/AMP), config, io, logger, metrics, plotting
task1/     inductive biases (STL-10; ResNet-50, ViT-B/16, CLIP)
report/    NeurIPS-style report source
```
`task1/README.md` has Task 1's setup, commands, files, rules and required outputs.

## How to run
```bash
python -m task1.data.make_subset                                     # once: splits + eval subset
python -m task1.data.make_cue_conflicts                              # once: cue-conflict images (AdaIN)
python -m task1.scripts.run_task1 --config task1/configs/default.yaml
```

## Status
Task 1: done. Results are in `task1/results/`.

## External code
| What | Source | Where used |
|---|---|---|
| AdaIN encoder/decoder architecture, AdaIN math, and pretrained weights (`decoder.pth`, `vgg_normalised.pth`) | Huang & Belongie (2017), via the unofficial PyTorch port https://github.com/naoto0804/pytorch-AdaIN (weights: GitHub Release v0.0.0) | `task1/data/make_cue_conflicts.py` |
| ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`) | torchvision | `task1/models/backbones.py` |
| CLIP ViT-B-32 (`openai` weights) | OpenCLIP | `task1/models/backbones.py` |
