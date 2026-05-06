#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/_combined/full_suite_new_datasets}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "lichess_top3_full" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=120"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "dmlab_seekavoid_full" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=301"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
