#!/usr/bin/env bash
# Full experiment suite over custom checkpoint-generated MuJoCo datasets.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/custom_mujoco_suite}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
HISTORY_K="${HISTORY_K:-16}"
MAX_EPS="${MAX_EPS:-384}"

DATASETS="${DATASETS:-custom_mujoco_hopper,custom_mujoco_halfcheetah,custom_mujoco_walker2d,custom_mujoco_ant,custom_mujoco_humanoid}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

mkdir -p "${OUT_ROOT}"
find "${OUT_ROOT}" -mindepth 1 -delete 2>/dev/null || true

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "${DATASETS}" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}" \
      "++data.max_episodes_per_policy=${MAX_EPS}"

echo
echo "== custom-suite aggregate =="
python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
echo "Custom MuJoCo suite complete. Results in ${OUT_ROOT}/"
