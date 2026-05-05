#!/usr/bin/env bash
# Synthetic + step-resampled MuJoCo suite with clean subfolder layout.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-outputs/suites/state_resampled_v1}"
BUILD_ROOT="${BUILD_ROOT:-${SUITE_ROOT}/build}"
RUNS_ROOT="${RUNS_ROOT:-${SUITE_ROOT}/runs}"
SYNTH_ROOT="${SYNTH_ROOT:-${RUNS_ROOT}/synthetic}"
CUSTOM_ROOT="${CUSTOM_ROOT:-${RUNS_ROOT}/custom_mujoco_step_resampled}"
STORE_CACHE_ROOT="${STORE_CACHE_ROOT:-${SUITE_ROOT}/cache/custom_mujoco_store}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
HISTORY_K="${HISTORY_K:-16}"
MAX_EPS="${MAX_EPS:-384}"

CUSTOM_DATASETS="${CUSTOM_DATASETS:-custom_mujoco_step_resampled_hopper,custom_mujoco_step_resampled_halfcheetah,custom_mujoco_step_resampled_walker2d,custom_mujoco_step_resampled_ant,custom_mujoco_step_resampled_humanoid}"
SYNTHETIC_DATASETS="${SYNTHETIC_DATASETS:-synthetic_grf}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

mkdir -p "${SUITE_ROOT}" "${BUILD_ROOT}" "${RUNS_ROOT}" "${SYNTH_ROOT}" "${CUSTOM_ROOT}" "${STORE_CACHE_ROOT}"
export INR_CUSTOM_MUJOCO_CACHE="${STORE_CACHE_ROOT}"
find "${BUILD_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${SYNTH_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${CUSTOM_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${STORE_CACHE_ROOT}" -mindepth 1 -delete 2>/dev/null || true
rm -f "${SUITE_ROOT}/aggregate.csv" "${SUITE_ROOT}/aggregate.md"

python scripts/build_custom_mujoco.py \
  --generation-mode resampled_steps \
  --n-gpus "${N_GPUS}" \
  --out-root "${BUILD_ROOT}" \
  --force-rebuild

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${SYNTH_ROOT}" \
  --datasets "${SYNTHETIC_DATASETS}" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${CUSTOM_ROOT}" \
  --datasets "${CUSTOM_DATASETS}" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}" \
      "++data.max_episodes_per_policy=${MAX_EPS}"

echo
echo "== step-resampled suite aggregate =="
python -m eval.summary "${RUNS_ROOT}" --out "${SUITE_ROOT}/aggregate.csv" --md "${SUITE_ROOT}/aggregate.md"
echo "Step-resampled suite complete. Results in ${SUITE_ROOT}/"
