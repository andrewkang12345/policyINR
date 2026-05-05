#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
OUT_ROOT="${OUT_ROOT:-outputs/lichess_full_2Xepisode}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-4}"
HISTORY_K="${HISTORY_K:-240}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift}"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "lichess_full_2Xepisode" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.history_k=${HISTORY_K}"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
