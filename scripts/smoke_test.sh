#!/usr/bin/env bash
# Fast end-to-end smoke test. Exercises the whole pipeline on a tiny
# synthetic dataset with a single model. Expected runtime: ~1-2 min.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="outputs/smoke"
rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

MODELS=(cvae inr_transformer_history_conditioned inr_diffusion_history_conditioned inr_transformer_fitted_latent)
EXP="no_shift"

for MODEL in "${MODELS[@]}"; do
  RUN_NAME="smoke_${MODEL}"
  echo "== smoke test: model=${MODEL} =="
  python -m train.main \
    data=synthetic_grf_small \
    model=${MODEL} \
    experiment=${EXP} \
    train.epochs=2 \
    train.batch_size=32 \
    train.eval_batch_size=64 \
    train.num_workers=0 \
    train.log_every=5 \
    train.history_k=8 \
    train.amp=false \
    eval.per_episode_samples=4 \
    eval.probe_cv_folds=2 \
    seed=0 \
    run_name=${RUN_NAME} \
    output_dir=${OUT_ROOT}/${RUN_NAME}
done

echo
echo "== smoke summary =="
python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
echo
echo "Smoke test complete. Results in ${OUT_ROOT}/"
