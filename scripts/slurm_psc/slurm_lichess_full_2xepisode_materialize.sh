#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-lichess-2x-mat
#SBATCH --array=0-27
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/lichess-2x-mat-%A_%a.out

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

if ! python - <<'PY'
import chess.pgn
PY
then
  python -m pip install --user python-chess
fi

MODELS=(cvae inr_transformer_history_conditioned inr_diffusion_history_conditioned inr_transformer_fitted_latent)
EXPS=(no_shift new_policy single_shift conflation generalization specialization novel_generalization)

IDX=$SLURM_ARRAY_TASK_ID
EXP_IDX=$(( IDX % 7 ))
MODEL_IDX=$(( IDX / 7 ))
MODEL=${MODELS[$MODEL_IDX]}
EXP=${EXPS[$EXP_IDX]}

python scripts/materialize_shared_lichess_2x_runs.py \
  --root "$REPO/outputs/lichess_full_2Xepisode" \
  --data-name lichess_full_2Xepisode \
  --models "$MODEL" \
  --experiments "$EXP" \
  --seed 0
