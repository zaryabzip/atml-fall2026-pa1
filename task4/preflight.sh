#!/usr/bin/env bash
# Task 4 preflight: runs EVERY Task 4 step for a few batches (train Vanilla/GCSC/PROSER, extract, evaluate, plots)
# so that package / CUDA / fp16 errors show up in ~3 minutes instead of an hour into the real run.
# Writes only to a temporary folder; the real checkpoints, caches and results are not touched.
#
#   bash task4/preflight.sh            (from the pa1 folder)
set -euo pipefail
PF=${PF_ROOT:-/tmp/pa1_task4_preflight}
rm -rf "$PF" task4/results/pf_vanilla task4/results/pf_gcsc task4/results/pf_proser; mkdir -p "$PF"
export PA1_OUT_ROOT="$PF"   # checkpoints go here, not into the real checkpoints folder
FAST="train.epochs=1 train.max_steps=5 train.resume=false train.num_workers=2"

echo "== versions"
python - <<'PY'
import sys, torch, torchvision, numpy, scipy, sklearn, matplotlib, yaml
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
from task4.data.cifar import SPLIT_FILE
assert SPLIT_FILE.exists(), f"missing {SPLIT_FILE}: copy the repo's task4/data/splits folder (do NOT regenerate it)"
PY

echo "== data (downloads CIFAR-10/100 once)"
python -c "from task4.data.cifar import cifar10_test, cifar100_unknowns; cifar10_test(); cifar100_unknowns('near'); cifar100_unknowns('far')"

echo "== train (5 steps each, fp16 on GPU)"
python -m task4.train --config task4/configs/vanilla.yaml train.epochs=2 train.max_steps=5 train.num_workers=2 run_name=pf_vanilla
python -m task4.train --config task4/configs/gcsc.yaml $FAST run_name=pf_gcsc
python -m task4.train --config task4/configs/proser.yaml $FAST run_name=pf_proser method.init_from=pf_vanilla

echo "== resume (rerunning a finished run must load last.pt and not retrain)"
python -m task4.train --config task4/configs/vanilla.yaml train.epochs=2 train.max_steps=5 train.num_workers=2 run_name=pf_vanilla | grep "Resuming pf_vanilla after epoch 2"

echo "== extract (first 512 examples of each part)"
for r in pf_vanilla pf_gcsc pf_proser; do
  python -m task4.extract_outputs --run $r --include-unknowns --limit 512 --cache-root "$PF/cache"
done

echo "== evaluate"
python -m task4.evaluate_osr --runs pf_vanilla pf_gcsc pf_proser --cache-root "$PF/cache" --out-dir "$PF/osr" > "$PF/evaluate.log"
ls "$PF/osr"

rm -rf task4/results/pf_vanilla task4/results/pf_gcsc task4/results/pf_proser
echo "PREFLIGHT OK: safe to start the real run"
