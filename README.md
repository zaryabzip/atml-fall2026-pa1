# PA1: (ATML Fall 2026)

Programming Assignment 1 for ATML (Fall 2026): how vision models behave beyond the IID setting. This repository contains Tasks 1 to 4.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Run every script from this folder as a module, e.g. `python -m task1.scripts.run_task1 --config task1/configs/default.yaml`.
Tasks 1 and 2 run on a 16 GB Apple Silicon Mac (MPS); Tasks 3 and 4 were run on Kaggle (`kaggle/`).

## Paths
| Variable | Default | Meaning |
|---|---|---|
| `PA1_DATA_ROOT` | `./data` | datasets (`stl10/`, `pacs/`, `cifar/`) |
| `PA1_OUT_ROOT` | `.` | checkpoints root |

Small results (configs, JSON tables, plots) go to `taskN/results/`.

## Data
- **STL-10** (Task 1): downloaded automatically from the Hugging Face mirror `tanganke/stl10` (labeled train + test, 230 MB).
- **PACS** (Tasks 2 and 3): `python -m shared.download_pacs` (Hugging Face `flwrlabs/pacs`, 9,991 images).
- **CIFAR-10 / CIFAR-100** (Task 4): downloaded automatically from the Hugging Face copies `uoft-cs/cifar10` and `uoft-cs/cifar100`.

Splits (seed 6304), made once and saved:
```bash
python -m task1.data.make_subset      # STL-10 80/20 + 500-image eval subset
python -m shared.make_pacs_splits     # PACS: stratified 80/20 per source domain (Sketch not split)
python -m task4.data.make_splits      # CIFAR-10 90/10 train/val
```

## Layout
```
common/    seed, device (CUDA/MPS/AMP), config, io, logger, metrics, plotting
task1/     inductive biases (STL-10; ResNet-50, ViT-B/16, CLIP)
task2/     unsupervised domain adaptation (PACS; Source-only, DAN, DANN, CDAN)
task3/     domain generalization (PACS; ERM, DAN-DG, SAM)
task4/     open-set recognition (CIFAR-10 vs CIFAR-100; Vanilla, GCSC, PROSER)
shared/    PACS data, splits, frozen-BN training loop, evaluation, multi-kernel MMD
kaggle/    notebooks and a helper script used to run Tasks 2 to 4 on Kaggle GPUs
```
Each `taskN/README.md` has that task's setup, commands, files, rules and required outputs.

## How to run
```bash
python -m task1.data.make_subset                                     # once: splits + eval subset
python -m task1.data.make_cue_conflicts                              # once: cue-conflict images (AdaIN)
python -m task1.scripts.run_task1 --config task1/configs/default.yaml
```
Task 2: see `task2/README.md` (train each method with `python -m task2.train --config task2/configs/<method>.yaml`, then `python -m task2.evaluate_final`).
Task 3: see `task3/README.md` (`python -m task3.train --config task3/configs/<method>.yaml`, then `python -m task3.evaluate_sketch`).
Task 4: see `task4/README.md` (`python -m task4.train --config task4/configs/<method>.yaml`, `python -m task4.extract_outputs`, then `python -m task4.evaluate_osr`).

## Status
Task 1: done. Results are in `task1/results/`.

Task 2: done. DAN uses the unbiased MMD estimator. DANN and CDAN normalize the discriminator input
(`method.disc_input_norm`), which stops the backbone features from growing without bound under gradient reversal.
Both deviations are documented in the report. Results are in `task2/results/`.

Task 3: done. DAN-DG uses the same unbiased MMD estimator. Results are in `task3/results/`.

Task 4: done (optional RPL not implemented). Results are in `task4/results/`.

## External code
| What | Source | Where used |
|---|---|---|
| AdaIN encoder/decoder architecture, AdaIN math, and pretrained weights (`decoder.pth`, `vgg_normalised.pth`) | Huang & Belongie (2017), via the unofficial PyTorch port https://github.com/naoto0804/pytorch-AdaIN (weights: GitHub Release v0.0.0) | `task1/data/make_cue_conflicts.py` |
| ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`) | torchvision | `task1/models/backbones.py` |
| CLIP ViT-B-32 (`openai` weights) | OpenCLIP | `task1/models/backbones.py` |

## Use of AI tools
Claude Code was used to help generate code in this repository.
