#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_ROOT="outputs/_5p_master_logs"
mkdir -p "$LOG_ROOT"
stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

stage "wait for droid 5col warmup"
while pgrep -fa 'data=droid_lowdim_full_balanced_min300_remove_5col model=cvae experiment=no_shift_5p' >/dev/null 2>&1; do
  sleep 30
done
stage "warmup done"

stage "droid balanced 5col hk300 — 5 models x 7 exp x 2 seeds"
bash scripts/run_droid_balanced_5col_suite.sh > "$LOG_ROOT/droid_5col.log" 2>&1
stage "droid 5col done"

stage "all_policy_metrics on each new suite"
for root in outputs/lichess/2x_hk240_top5 \
            outputs/synthetic/baseline_10x_5p \
            outputs/droid/balanced_min300_remove_hk300_5col; do
  python scripts/append_all_policy_metrics.py "$root" --n-gpus 4 \
    > "$LOG_ROOT/$(basename $root)_all_policy_metrics.log" 2>&1 || \
    echo "[warn] all_policy_metrics on $root returned non-zero"
done
stage "all_policy_metrics done"

stage "regenerate aggregate_all.md"
python scripts/regenerate_aggregate_all.py
stage "ALL DONE"
