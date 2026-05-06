#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/INR
export PYTHONPATH=/mnt/data/INR:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/mnt/data/INR/outputs/droid_lowdim_full_10x_min200_suite_20260502_mat512
python scripts/multi_gpu_launch.py \
  --n-gpus 4 \
  --seeds 0,1 \
  --out-root "$OUT" \
  --datasets droid_lowdim_full_10x_min200 \
  --models inr_transformer_infer_latent_maml \
  --experiments no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization \
  --skip-completed \
  --overrides \
    shift.kind=predefined_split \
    train.epochs=30 \
    train.batch_size=64 \
    train.eval_batch_size=128 \
    train.num_workers=0 \
    train.materialize_dataset=true \
    train.history_k=200
python -m eval.summary "$OUT" --out "$OUT/aggregate.csv" --md "$OUT/aggregate.md"
