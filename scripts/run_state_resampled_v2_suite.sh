#!/usr/bin/env bash
# Shared-state-bag v2 state-shift MuJoCo suite.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-outputs/suites/state_resampled_v2}"
BUILD_ROOT="${BUILD_ROOT:-${SUITE_ROOT}/build}"
RUNS_ROOT="${RUNS_ROOT:-${SUITE_ROOT}/runs}"
CUSTOM_ROOT="${CUSTOM_ROOT:-${RUNS_ROOT}/custom_mujoco_state_resampled_v2}"
STORE_CACHE_ROOT="${STORE_CACHE_ROOT:-${SUITE_ROOT}/cache/custom_mujoco_store}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
HISTORY_K="${HISTORY_K:-16}"
MAX_EPS="${MAX_EPS:-384}"
KNN_K="${KNN_K:-32}"
TAIL_FRACTION="${TAIL_FRACTION:-0.4}"

CUSTOM_DATASETS="${CUSTOM_DATASETS:-custom_mujoco_state_resampled_v2_hopper,custom_mujoco_state_resampled_v2_halfcheetah,custom_mujoco_state_resampled_v2_walker2d,custom_mujoco_state_resampled_v2_ant,custom_mujoco_state_resampled_v2_humanoid}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

mkdir -p "${SUITE_ROOT}" "${BUILD_ROOT}" "${RUNS_ROOT}" "${CUSTOM_ROOT}" "${STORE_CACHE_ROOT}"
export INR_CUSTOM_MUJOCO_CACHE="${STORE_CACHE_ROOT}"
find "${BUILD_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${CUSTOM_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${STORE_CACHE_ROOT}" -mindepth 1 -delete 2>/dev/null || true
rm -f "${SUITE_ROOT}/aggregate.csv" "${SUITE_ROOT}/aggregate.md"

python scripts/build_custom_mujoco.py \
  --generation-mode state_resampled_v2 \
  --n-gpus "${N_GPUS}" \
  --knn-k "${KNN_K}" \
  --tail-fraction "${TAIL_FRACTION}" \
  --out-root "${BUILD_ROOT}" \
  --force-rebuild

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
echo "== state-resampled v2 suite aggregate =="
python -m eval.summary "${RUNS_ROOT}" --out "${SUITE_ROOT}/aggregate.csv" --md "${SUITE_ROOT}/aggregate.md"
echo "State-resampled v2 suite complete. Results in ${SUITE_ROOT}/"
