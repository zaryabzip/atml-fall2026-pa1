#!/usr/bin/env bash
# Run two configs at once, one per T4.
# Usage: bash kaggle/run_parallel.sh <python module> <config for GPU 0> <config for GPU 1> [overrides for both...]
# Example: bash kaggle/run_parallel.sh task4.train task4/configs/vanilla.yaml task4/configs/gcsc.yaml train.num_workers=2
# Logs: logs/<config name>.log  (watch with: tail -n 5 logs/*.log)
set -uo pipefail
module=$1; cfg0=$2; cfg1=$3; shift 3
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -m "$module" --config "$cfg0" "$@" > "logs/$(basename "$cfg0" .yaml).log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 python -m "$module" --config "$cfg1" "$@" > "logs/$(basename "$cfg1" .yaml).log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit $status
