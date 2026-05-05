#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/INR
export OUT_ROOT=/mnt/data/INR/outputs/droid_lowdim_full_shards707of2048_p345_balanced_min300_remove_hk300_suite_20260503
export FAST_BATCH=512
export FAST_EVAL_BATCH=512
export MAML_BATCH=32
export MAML_EVAL_BATCH=64
export WORKERS=0
bash scripts/run_droid_balanced_min300_remove_suite.sh 2>&1 | tee "$OUT_ROOT/run.log"
