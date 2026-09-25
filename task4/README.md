# Task 4: Open-Set Recognition

A CIFAR-10 classifier should reject images from classes it never saw (CIFAR-100) instead of confidently mislabeling them.

## Setup
- Known: all 10 CIFAR-10 classes. Stratified 90/10 train/val split (seed 6304). Final known test = full CIFAR-10 test set.
- Unknown (CIFAR-100 **test** images only, 800 each):
  - Near: bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel
  - Far: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe
- CIFAR ResNet-18: 3x3 stride-1 first conv, no max-pool, 32x32 input, random init.
- Recipe: random crop (pad 4) + flip, SGD lr 0.1, momentum 0.9, wd 5e-4, cosine decay, batch 128, 100 epochs. Keep the checkpoint with the best CIFAR-10 val accuracy.

## Run
```bash
python -m task4.data.make_splits
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml    # starts from vanilla
python -m task4.extract_outputs --run vanilla               # known data only
# after ALL models and scores are fixed:
python -m task4.extract_outputs --run vanilla --include-unknowns
python -m task4.evaluate_osr
```
Kaggle is recommended: `kaggle/task4_kaggle.ipynb` runs all of the above (Vanilla and GCSC at the same time, one per GPU) plus PROSER's final-epoch model (`proser_final`, extra rows only). On a Mac, use `train.epochs=1` for smoke tests only.

## Models and scores
| Model | Config | Idea |
|---|---|---|
| Vanilla | `vanilla.yaml` | plain cross-entropy |
| GCSC | `gcsc.yaml` | Vanilla + RandAugment(2, 9). The only change. |
| PROSER | `proser.yaml` | from Vanilla, + 5 dummy classes; classifier placeholders (beta 1) + manifold-mixup data placeholders after layer2 (gamma 0.1); 50 epochs, lr 1e-3 |
| RPL (optional) | `rpl.yaml` | reciprocal points |

Scores (larger = more novel): MSP `1 - max softmax`, MLS `-max logit`, Energy `-logsumexp`, Mahalanobis (class means + shared diagonal covariance from unaugmented train features, +1e-6), PROSER placeholder score.

## Files
| File | Role |
|---|---|
| `data/cifar.py` | CIFAR-10 splits/transforms + fixed CIFAR-100 unknowns |
| `models/resnet_cifar.py` | CIFAR ResNet-18, split after layer2 for mixup |
| `methods/` | vanilla, gcsc, proser, manifold_mixup, rpl |
| `scores/` | msp, mls, energy, mahalanobis, proser_score |
| `train.py` | training entry point |
| `extract_outputs.py` | saves features + logits to `cache/` so all scores use the same outputs |
| `evaluation/failure_analysis.py` | accepted unknowns at the vanilla MLS threshold |
| `evaluate_osr.py` | AUROC, thresholds, tables, figures, score agreement (`results/osr/score_agreement.json`) |

Threshold: tau = 95th percentile of the score on **CIFAR-10 val**; accept when u(x) <= tau.

## Rules
- CIFAR-100 never touches training, checkpoint choice, score design or thresholds.
- Unknown outputs are extracted only after every model and score is fixed.
- PROSER accuracy (CSA) uses only the 10 known-class logits.

## Status
Done: data, splits, model, training loop, Vanilla, GCSC, PROSER + mixup, scores, output extraction, OSR evaluation, failure analysis.
Not done: optional RPL.

## Required outputs
- Table: MSP / MLS / Energy / Mahalanobis on Vanilla (near, far, all AUROC + rejection at tau)
- Table: Vanilla / GCSC / PROSER with MLS (+ PROSER placeholder-score row): CSA + near/far OSR metrics
- One figure: MSP, MLS, Mahalanobis score distributions or ROC
- At least 3 near and 3 far accepted unknowns (class, predicted class, score, threshold)
- Short analysis of augmentation vs PROSER
