#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
OUT_ROOT="${OUT_ROOT:-outputs/synthetic/baseline_10x_5p}"
SEEDS="${SEEDS:-0,1}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-2}"
HISTORY_K="${HISTORY_K:-16}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift_5p,new_policy_5p,single_shift_5p,conflation_5p,generalization_5p,specialization_5p,novel_generalization_5p}"

mkdir -p "${OUT_ROOT}"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "synthetic_grf_10x_5p" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --skip-completed \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.history_k=${HISTORY_K}"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
