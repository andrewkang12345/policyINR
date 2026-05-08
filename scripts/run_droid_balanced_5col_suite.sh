#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_ROOT="${OUT_ROOT:-outputs/droid/balanced_min300_remove_hk300_5col}"
SEEDS="${SEEDS:-0,1}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-0}"
EXPERIMENTS="${EXPERIMENTS:-no_shift_5p,new_policy_5p,single_shift_5p,conflation_5p,generalization_5p,specialization_5p,novel_generalization_5p}"

mkdir -p "${OUT_ROOT}"

CUDA_VISIBLE_DEVICES="${FAST_VISIBLE:-0,1,2,3}" python scripts/multi_gpu_launch.py \
  --n-gpus "${FAST_GPUS:-4}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "droid_lowdim_full_balanced_min300_remove_5col" \
  --models "${FAST_MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}" \
  --experiments "${EXPERIMENTS}" \
  --skip-completed \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${FAST_BATCH:-512}" \
      "train.eval_batch_size=${FAST_EVAL_BATCH:-512}" \
      "train.num_workers=${WORKERS}" \
      "train.materialize_dataset=true" \
      "train.history_k=300"

CUDA_VISIBLE_DEVICES="${MAML_VISIBLE:-0,1,2,3}" python scripts/multi_gpu_launch.py \
  --n-gpus "${MAML_GPUS:-4}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "droid_lowdim_full_balanced_min300_remove_5col" \
  --models "${MAML_MODELS:-inr_transformer_infer_latent_maml}" \
  --experiments "${EXPERIMENTS}" \
  --skip-completed \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${MAML_BATCH:-32}" \
      "train.eval_batch_size=${MAML_EVAL_BATCH:-64}" \
      "train.num_workers=${WORKERS}" \
      "train.materialize_dataset=true" \
      "train.history_k=300"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
