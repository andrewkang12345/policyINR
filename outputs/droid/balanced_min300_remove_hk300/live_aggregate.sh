#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/INR
OUT=/mnt/data/INR/outputs/droid_lowdim_full_shards707of2048_p345_balanced_min300_remove_hk300_suite_20260503
while true; do
  TS=$(date -Is)
  N=$(find "$OUT" -name summary.json | wc -l)
  if [ "$N" -gt 0 ]; then
    python -m eval.summary "$OUT" --out "$OUT/aggregate.csv.tmp" --md "$OUT/aggregate.md.tmp" >/tmp/droid_bal_min300_aggregate.log 2>&1 || true
    if [ -s "$OUT/aggregate.md.tmp" ]; then
      mv "$OUT/aggregate.csv.tmp" "$OUT/aggregate.csv"
      mv "$OUT/aggregate.md.tmp" "$OUT/aggregate.md"
      printf '[%s] aggregate updated with %s summaries\n' "$TS" "$N" >> "$OUT/live_aggregate.log"
    else
      printf '[%s] aggregate update failed with %s summaries\n' "$TS" "$N" >> "$OUT/live_aggregate.log"
    fi
  else
    printf '[%s] waiting for summaries\n' "$TS" >> "$OUT/live_aggregate.log"
  fi
  sleep 300
done
