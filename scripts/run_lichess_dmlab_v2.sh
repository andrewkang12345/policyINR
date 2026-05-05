#!/usr/bin/env bash
# v2 suite for the two new datasets after the per-dataset cleanup:
#   - lichess_top3 : episodes are now player-only plies (no opponent moves);
#                    7 experiments x 4 models x 2 seeds = 56 jobs.
#   - dmlab_seekavoid : 2 policies (snapshot_0, snapshot_1), each with both
#                    eps=0.0 (ID) and eps=0.25 (OOD) episodes; episodes are
#                    full-length (301 steps); 5 experiments (no_shift,
#                    single_shift, conflation, generalization, specialization)
#                    x 4 models x 2 seeds = 40 jobs.
# Total: 96 jobs across 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"

N_GPUS="${N_GPUS:-4}"
SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/lichess_dmlab_v2}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
HISTORY_K="${HISTORY_K:-16}"

MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent}"
LICHESS_EXPTS="${LICHESS_EXPTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"
# DMLab has only 2 policies after the snapshot/eps restructuring, so
# experiments that reference pid=2 (new_policy, novel_generalization) are
# omitted.
DMLAB_EXPTS="${DMLAB_EXPTS:-no_shift,single_shift,conflation,generalization,specialization}"

mkdir -p "${OUT_ROOT}"
find "${OUT_ROOT}" -mindepth 1 -delete 2>/dev/null || true

echo "=== lichess_top3 (shared_region shift, 42 jobs) ==="
python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "lichess_top3" \
  --models "${MODELS}" \
  --experiments "${LICHESS_EXPTS}" \
  --overrides \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}"

echo "=== dmlab_seekavoid (predefined_split shift, 30 jobs) ==="
python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "dmlab_seekavoid" \
  --models "${MODELS}" \
  --experiments "${DMLAB_EXPTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.history_k=${HISTORY_K}"

echo
echo "== lichess+dmlab v2 aggregate =="
python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
echo "Suite complete. Results in ${OUT_ROOT}/"
