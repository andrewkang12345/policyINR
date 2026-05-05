#!/usr/bin/env bash
# Full suite across the 2 supported new datasets (discrete-action / featurized-state).
# Lichess supports the full 7 experiments; DMLab omits the pid=2 experiments.
# Total default sweep: (7 + 5) x 4 models x 2 seeds = 96 runs.
# Runtime ~30 min on 4x A10G with the defaults below.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/full_suite_new_datasets}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
HISTORY_K="${HISTORY_K:-16}"

MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
LICHESS_EXPERIMENTS="${LICHESS_EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"
DMLAB_EXPERIMENTS="${DMLAB_EXPERIMENTS:-no_shift,single_shift,conflation,generalization,specialization}"

mkdir -p "${OUT_ROOT}"
find "${OUT_ROOT}" -mindepth 1 -delete 2>/dev/null || true

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "lichess_top3" \
  --models "${MODELS}" \
  --experiments "${LICHESS_EXPERIMENTS}" \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "dmlab_seekavoid" \
  --models "${MODELS}" \
  --experiments "${DMLAB_EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}"

echo
echo "== new-datasets full-suite aggregate =="
python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
echo "Full suite complete. Results in ${OUT_ROOT}/"
