#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/INR
export OUT_ROOT=/mnt/data/INR/outputs/droid_lowdim_full_10x_min200_suite_20260502_mat512
export WORKERS=0
export BATCH=512
export FAST_VISIBLE=0,1,2,3
export FAST_GPUS=4
export MAML_VISIBLE=0,1,2,3
export MAML_GPUS=4
bash scripts/run_droid_10x_min200_suite_full_util.sh 2>&1 | tee "$OUT_ROOT/run.log"
