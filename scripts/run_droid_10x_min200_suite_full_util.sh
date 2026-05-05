#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/droid_lowdim_full_10x_min200_suite_20260502}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-0}"
HISTORY_K="${HISTORY_K:-200}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

FAST_VISIBLE="${FAST_VISIBLE:-0,1,2,3,0,1,2,3}"
FAST_GPUS="${FAST_GPUS:-8}"
FAST_MODELS="${FAST_MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"

MAML_VISIBLE="${MAML_VISIBLE:-0,1,2,3}"
MAML_GPUS="${MAML_GPUS:-4}"
MAML_MODELS="${MAML_MODELS:-inr_transformer_infer_latent_maml}"

mkdir -p "${OUT_ROOT}"

CUDA_VISIBLE_DEVICES="${FAST_VISIBLE}" python scripts/multi_gpu_launch.py \
  --n-gpus "${FAST_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "droid_lowdim_full_10x_min200" \
  --models "${FAST_MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --skip-completed \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.materialize_dataset=true" \
      "train.history_k=${HISTORY_K}"

CUDA_VISIBLE_DEVICES="${MAML_VISIBLE}" python scripts/multi_gpu_launch.py \
  --n-gpus "${MAML_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "droid_lowdim_full_10x_min200" \
  --models "${MAML_MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --skip-completed \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.materialize_dataset=true" \
      "train.history_k=${HISTORY_K}"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
