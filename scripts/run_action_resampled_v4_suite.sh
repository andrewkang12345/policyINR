#!/usr/bin/env bash
# Minari-ID + synthetic-OOD action-resampled v4 MuJoCo suite.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
SUITE_ROOT="${SUITE_ROOT:-outputs/mujoco/suites/action_resampled_v4}"
BUILD_ROOT="${BUILD_ROOT:-${SUITE_ROOT}/build}"
RUNS_ROOT="${RUNS_ROOT:-${SUITE_ROOT}/runs}"
CUSTOM_ROOT="${CUSTOM_ROOT:-${RUNS_ROOT}/custom_mujoco_action_resampled_v4}"
STORE_CACHE_ROOT="${STORE_CACHE_ROOT:-${SUITE_ROOT}/cache/custom_mujoco_store}"
MINARI_CACHE_ROOT="${MINARI_CACHE_ROOT:-${SUITE_ROOT}/cache/minari}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
HISTORY_K="${HISTORY_K:-16}"
MAX_EPS="${MAX_EPS:-384}"
TAIL_FRACTION="${TAIL_FRACTION:-0.4}"
PARTITION_TRAIN_FRAC="${PARTITION_TRAIN_FRAC:-0.7}"
PARTITION_VAL_FRAC="${PARTITION_VAL_FRAC:-0.15}"
ACCEPTANCE_THRESHOLD="${ACCEPTANCE_THRESHOLD:-2.5}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-2048}"
MAX_CANDIDATE_BATCHES="${MAX_CANDIDATE_BATCHES:-128}"
PROBE_LIKE_SAMPLES="${PROBE_LIKE_SAMPLES:-8}"
PROBE_LIKE_REPEATS="${PROBE_LIKE_REPEATS:-4}"

CUSTOM_DATASETS="${CUSTOM_DATASETS:-custom_mujoco_action_resampled_v4_hopper,custom_mujoco_action_resampled_v4_halfcheetah,custom_mujoco_action_resampled_v4_walker2d,custom_mujoco_action_resampled_v4_ant,custom_mujoco_action_resampled_v4_humanoid}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

mkdir -p "${SUITE_ROOT}" "${BUILD_ROOT}" "${RUNS_ROOT}" "${CUSTOM_ROOT}" "${STORE_CACHE_ROOT}" "${MINARI_CACHE_ROOT}"
export INR_CUSTOM_MUJOCO_CACHE="${STORE_CACHE_ROOT}"
export INR_MINARI_CACHE="${MINARI_CACHE_ROOT}"
find "${BUILD_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${CUSTOM_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${STORE_CACHE_ROOT}" -mindepth 1 -delete 2>/dev/null || true
find "${MINARI_CACHE_ROOT}" -mindepth 1 -delete 2>/dev/null || true
rm -f "${SUITE_ROOT}/aggregate.csv" "${SUITE_ROOT}/aggregate.md"

python scripts/build_custom_mujoco.py \
  --generation-mode action_resampled_v4 \
  --n-gpus "${N_GPUS}" \
  --tail-fraction "${TAIL_FRACTION}" \
  --partition-train-frac "${PARTITION_TRAIN_FRAC}" \
  --partition-val-frac "${PARTITION_VAL_FRAC}" \
  --acceptance-threshold "${ACCEPTANCE_THRESHOLD}" \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --max-candidate-batches "${MAX_CANDIDATE_BATCHES}" \
  --probe-like-samples "${PROBE_LIKE_SAMPLES}" \
  --probe-like-repeats "${PROBE_LIKE_REPEATS}" \
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
echo "== action-resampled v4 suite aggregate =="
python -m eval.summary "${RUNS_ROOT}" --out "${SUITE_ROOT}/aggregate.csv" --md "${SUITE_ROOT}/aggregate.md"
echo "Action-resampled v4 suite complete. Results in ${SUITE_ROOT}/"
