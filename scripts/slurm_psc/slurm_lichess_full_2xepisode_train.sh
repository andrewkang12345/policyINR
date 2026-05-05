#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-lichess-2x
#SBATCH --array=0-3
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/lichess-2x-train-%A_%a.out

set -euo pipefail
PROJECT=/ocean/projects/cis260099p
REPO=$PROJECT/akang3/INR
cd "$REPO"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/akang3/polymer/envs/inr_torch"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export INR_LICHESS_CACHE="$PROJECT/akang3/INR_cache/lichess"
export INR_LOG_LEVEL=INFO

mkdir -p "$INR_LICHESS_CACHE" "$INR_LICHESS_CACHE/pgn" "$REPO/outputs/slurm"

if ! python - <<'PY'
import chess.pgn
PY
then
  python -m pip install --user python-chess
fi

MODELS=(cvae inr_transformer_history_conditioned inr_diffusion_history_conditioned inr_transformer_fitted_latent)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="lichess_full_2Xepisode__${MODEL}__no_shift__s0"
OUT_DIR="$REPO/outputs/lichess_full_2Xepisode/$RUN_NAME"
mkdir -p "$OUT_DIR"

run_train() {
  python -m train.main \
    data=lichess_full_2Xepisode \
    model="$MODEL" \
    experiment=no_shift \
    seed=0 \
    run_name="$RUN_NAME" \
    output_dir="$OUT_DIR" \
    ++data.pgn_dir="$INR_LICHESS_CACHE/pgn" \
    train.epochs=10 \
    train.batch_size=128 \
    train.num_workers=0 \
    train.history_k=120
}

if ! run_train; then
  echo "[slurm] first attempt failed for $RUN_NAME; retrying once after cleanup" >&2
  rm -f "$OUT_DIR"/best.pt "$OUT_DIR"/last.pt "$OUT_DIR"/eval.json "$OUT_DIR"/summary.json
  sleep 15
  run_train
fi
