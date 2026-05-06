#!/usr/bin/env bash
# Master driver for the 5-policy all-policies extension.
# Sequentially trains lichess top5, synthetic 5p, droid 5col on 4 GPUs;
# then runs all_policy_metrics on each and regenerates aggregate_all.md.
#
# Usage: nohup bash scripts/run_5p_master.sh > outputs/_5p_master.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_ROOT="outputs/_5p_master_logs"
mkdir -p "$LOG_ROOT"

stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

stage "wait for warmup materializes"
for pidf in outputs/_materialize_logs/*.log; do
  : # nothing — they were started outside this script; we just wait via pgrep below
done
while pgrep -fa 'data=lichess_top5_full_2Xepisode|data=droid_lowdim_full_balanced_min300_remove_5col' >/dev/null 2>&1; do
  sleep 30
done
stage "warmups done"

stage "lichess top5 (2x_hk240) — 4 models x 7 exp x 1 seed"
bash scripts/run_lichess_top5_2Xepisode.sh \
  > "$LOG_ROOT/lichess_top5.log" 2>&1
stage "lichess top5 done"

stage "synthetic_grf 10x 5p — 4 models x 7 exp x 2 seeds"
bash scripts/run_synthetic_grf_10x_5p_suite.sh \
  > "$LOG_ROOT/synthetic_5p.log" 2>&1
stage "synthetic 5p done"

stage "droid balanced 5col hk300 — 5 models x 7 exp x 2 seeds"
bash scripts/run_droid_balanced_5col_suite.sh \
  > "$LOG_ROOT/droid_5col.log" 2>&1
stage "droid 5col done"

stage "all_policy_metrics on each new suite"
for root in outputs/lichess/2x_hk240_top5 \
            outputs/synthetic/baseline_10x_5p \
            outputs/droid/balanced_min300_remove_hk300_5col; do
  python scripts/append_all_policy_metrics.py "$root" --n-gpus 4 \
    > "$LOG_ROOT/$(basename $root)_all_policy_metrics.log" 2>&1 || \
    echo "[warn] all_policy_metrics on $root returned non-zero (check log)"
done
stage "all_policy_metrics done"

stage "regenerate aggregate_all.md"
python scripts/regenerate_aggregate_all.py
stage "aggregate_all.md regenerated"

stage "ALL DONE"
