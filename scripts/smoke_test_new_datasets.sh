#!/usr/bin/env bash
# Smoke test: end-to-end sanity on the 2 supported new datasets x 4 models x 1 experiment.
# Expected to finish in ~60s on a single GPU. Confirms discrete-action
# plumbing + featurizers + shift + training + eval.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT=outputs/smoke_new_datasets
rm -rf "$OUT"

for data in dmlab_seekavoid lichess_top3; do
  for model in cvae inr_transformer_history_conditioned inr_diffusion_history_conditioned inr_transformer_fitted_latent; do
    echo "== $data × $model =="
    python -m train.main \
      data="$data" model="$model" experiment=no_shift \
      train.epochs=1 train.batch_size=32 train.history_k=8 \
      train.num_workers=0 train.amp=false \
      seed=0 run_name="smoke_${data}_${model}" \
      output_dir="$OUT/${data}_${model}"
  done
done

echo
echo "== new-datasets smoke summary =="
python -m eval.summary "$OUT" --md "$OUT/aggregate.md" --out "$OUT/aggregate.csv"
echo "smoke complete — results in $OUT/"
