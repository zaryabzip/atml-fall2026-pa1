# Task 2: Unsupervised Domain Adaptation

Train on labeled Photo, Art Painting and Cartoon. Use unlabeled Sketch images to adapt. Test on Sketch.

## Setup
- PACS. Sources: photo, art_painting, cartoon (stratified 80/20 split each, seed 6304). Target: sketch (all images, no labels during training).
- ResNet-18 `IMAGENET1K_V1` + 7-class head, full fine-tuning.
- Resize 256x256, random 224 crop + flip for training; center 224 crop for eval.
- BatchNorm running stats frozen at ImageNet values (gamma/beta still train).
- AdamW lr 1e-4, wd 1e-4, max 30 epochs, early stop after 5 without better **mean source-val macro-F1**.
- Each step: 8 images per source domain + 24 Sketch images.
- One "source epoch" = pooled source train images / 24 steps (`steps_per_epoch: auto`). State this in the report.

The training loop is shared with Task 3 (`shared/pacs_train.py`), so all methods use the same data, optimizer and budget.

## Run
```bash
python -m shared.download_pacs        # once
python -m shared.make_pacs_splits     # once
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
# design study (pick one), e.g.
python -m task2.train --config task2/configs/dann.yaml method.max_grl=0.25 run_name=dann_grl0.25
python -m task2.evaluate_final        # ONLY after everything above is locked
```
Kaggle is recommended. Train two runs at once with `kaggle/run_parallel.sh`.

## Running one design study
Each study is independent: train the extra runs under their own names, evaluate only the runs in
that study into their own folder (so the main `final_eval` is never overwritten), then plot it.
Example, the DAN study over lambda_mmd (lambda = 1 is the main `dan` run, already trained):
```bash
python -m task2.train --config task2/configs/dan.yaml method.lambda_mmd=0.1 run_name=dan_lmmd0.1
python -m task2.train --config task2/configs/dan.yaml method.lambda_mmd=10 run_name=dan_lmmd10
python -m task2.evaluate_final --runs source_only dan_lmmd0.1 dan dan_lmmd10 --baseline source_only --out-name design_study_dan
python -m task2.evaluation.plot_design_study --eval-name design_study_dan --runs dan_lmmd0.1 dan dan_lmmd10 --param method.lambda_mmd --reference source_only --log-x
```
The DANN study works the same way: `method.max_grl=0.25 run_name=dann_grl0.25`, `--param method.max_grl`.
Outputs: `task2/results/design_study_dan/{final_eval.json, final_eval_summary.csv, study_plot.png}`.

## Methods
| Method | Config | Idea |
|---|---|---|
| Source-only | `source_only.yaml` | Cross-entropy on sources. Also Task 3's ERM baseline. |
| DAN | `dan.yaml` | + MMD between source and Sketch features (lambda = 1, 3 RBF kernels) |
| DANN | `dann.yaml` | + domain discriminator behind a gradient-reversal layer |
| CDAN | `cdan.yaml` | Like DANN, but the discriminator sees features x class probabilities |

## Files
| File | Role |
|---|---|
| `configs/` | base protocol + one config per method |
| `methods/` | one loss per method |
| `models/grl.py` | gradient reversal + alpha(p) schedule |
| `models/domain_discriminator.py` | 256 hidden, ReLU, dropout 0.5, 2 outputs |
| `evaluation/domain_separability.py` | source-vs-Sketch logistic regression (50% = chance) |
| `evaluation/class_analysis.py` | per-class changes and confusions vs Source-only |
| `train.py` | training entry point |
| `evaluate_final.py` | final Sketch evaluation |

Shared pieces: `shared/pacs*.py`, `shared/mmd.py`.

## Rules
- Sketch **labels** are used only in `evaluate_final.py`, after all checkpoints and settings are fixed.
- Checkpoints are chosen by mean source-val macro-F1 only.

## Status
Done: Source-only, DAN, DANN, CDAN, domain separability, class analysis, final evaluation, the lambda_MMD design study.

Deviations from the handout (documented in the report):
- DAN uses the unbiased MMD estimator (`shared/mmd.py`; the biased one collapsed DAN-DG at small batches).
- DANN and CDAN apply a parameter-free LayerNorm to the discriminator input (`method.disc_input_norm`); with the
  handout setup both diverged. Evidence: `python -m task2.evaluation.stability_check` -> `results/stability_check/`.
- An extra attempt with a 10x discriminator learning rate is kept separate (`results/extra_disc_lr10/`,
  `results/disc_lr_decision.txt`); it is not the main result.

Final-analysis extras: `python -m task2.evaluation.sketch_class_report --task task2 --runs source_only dan dann cdan`
(per-class accuracy, confusion matrices, failure images -> `results/sketch_class_report/`).

## Required outputs
- Table: each source-val domain, mean source accuracy + macro-F1, Sketch accuracy + macro-F1, Sketch accuracy change, domain separability
- Training curves: classification loss + MMD or domain loss
- Per-class Sketch changes + confusions
- Design-study table or plot
