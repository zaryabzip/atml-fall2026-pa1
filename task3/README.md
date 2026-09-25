# Task 3: Domain Generalization

Same PACS setup as Task 2, but Sketch is **never seen** until the final test.

## Setup
Exactly the Task 2 protocol: same splits, model, preprocessing, frozen BatchNorm, optimizer, budget, early stopping, seed 6304, and 8 images per source domain per step.

## Run
```bash
# ERM is not retrained: it loads checkpoints/task2/source_only/best.pt
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
python -m task3.evaluate_sketch       # ONLY after everything is trained and locked
python -m task2.evaluation.plot_training_curves --runs task2:source_only task3:dan_dg task3:sam --out task3/results/training_curves.png
```
Kaggle is recommended. SAM does two forward/backward passes per step.

## Running one design study
Pick ONE study: vary lambda_DG in {0.1, 1, 10} for DAN-DG, or rho in {0.01, 0.05, 0.1} for SAM. The main
runs (`dan_dg`, `sam`) always keep lambda_DG = 1 and rho = 0.05. Train the extra runs under their own names,
evaluate only that study into its own folder, then plot it. Example, the DAN-DG study:
```bash
python -m task3.train --config task3/configs/dan_dg.yaml method.lambda_dg=0.1 run_name=dan_dg_lam0.1
python -m task3.train --config task3/configs/dan_dg.yaml method.lambda_dg=10 run_name=dan_dg_lam10
python -m task3.evaluate_sketch --runs erm dan_dg_lam0.1 dan_dg dan_dg_lam10 --out-name design_study_dan_dg
python -m task2.evaluation.plot_design_study --task task3 --eval-name design_study_dan_dg --runs dan_dg_lam0.1 dan_dg dan_dg_lam10 --param method.lambda_dg --reference erm --log-x
```
The SAM study works the same way: `method.rho=0.01 run_name=sam_rho0.01`, `--param method.rho`.
Outputs: `task3/results/<out-name>/{final_eval.json, final_eval_summary.csv, source_side.json, study_plot.png}`.

Final-analysis extras (after evaluate_sketch): per-class accuracy, confusion matrices, failure images and the
per-class DAN (Task 2) vs DAN-DG comparison:
```bash
python -m task2.evaluation.sketch_class_report --task task3 --runs erm dan_dg sam --compare task2:dan task3:dan_dg
```

## Methods
| Method | Config | Idea |
|---|---|---|
| ERM | `erm.yaml` | Task 2 Source-only checkpoint, unchanged |
| DAN-DG | `dan_dg.yaml` | + MMD between each pair of source domains (lambda = 1). Same MMD as Task 2. |
| SAM | `sam.yaml` | Minimize the worst loss within radius rho = 0.05 of the weights |

## Files
| File | Role |
|---|---|
| `configs/` | one config per method (inherit Task 2's base) |
| `methods/erm.py` | loads the Task 2 Source-only checkpoint |
| `methods/dan_dg.py` | pairwise source MMD loss |
| `methods/sam.py` | two-pass SAM step |
| `evaluation/source_domain_separability.py` | Photo/Art/Cartoon logistic regression (33.3% = chance) |
| `evaluation/sharpness.py` | loss increase after one step of size 0.05 on a fixed 96-image val batch |
| `train.py` | training entry point (refuses to load Sketch) |
| `evaluate_sketch.py` | the only Task 3 script allowed to load Sketch |

## Rules
- No Sketch image is loaded by training, diagnostics, checkpoint selection or setting choices. `shared/pacs_protocol.target_dataset` raises an error if you try.
- Task 2 Sketch results must not change any Task 3 setting.
- Main comparison keeps lambda_DG = 1 and rho = 0.05, whatever the design study shows.

## Status
Done: DAN-DG, SAM, source-domain separability, sharpness proxy, Sketch evaluation, study plots.

## Required outputs
- Table: each source-val domain, mean and worst source, Sketch accuracy + macro-F1, Sketch change vs ERM
- Source-domain separability + sharpness proxy for all 3 models
- Training curves (classification loss + MMD where used)
- Design-study table or plot
- Per-class Sketch changes + failures, compared with Task 2
