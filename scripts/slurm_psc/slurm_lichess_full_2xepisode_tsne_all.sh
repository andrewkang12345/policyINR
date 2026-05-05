#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-lichess-2x-tsne-all
#SBATCH --array=0-27
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/lichess-2x-tsne-all-%A_%a.out

set -euo pipefail
PROJECT=/ocean/projects/cis260099p
REPO=$PROJECT/akang3/INR
cd "$REPO"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/akang3/polymer/envs/inr_torch"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export INR_LOG_LEVEL=INFO
export MUJOCO_GL=egl

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

MODEL_NAME="$MODEL" EXP_NAME="$EXP" python - <<'PY'
import os
import scripts.plot_policy_tsne as mod

target = "lichess_full_2Xepisode"
model = os.environ["MODEL_NAME"]
exp = os.environ["EXP_NAME"]
for subdir, data_label, runs_root, data_cfg in mod.RUN_GROUPS:
    if data_label != target:
        continue
    run_dir = runs_root / f"{data_cfg}__{model}__{exp}__s0"
    result = mod._extract(run_dir)
    if result is None:
        raise RuntimeError(f"missing run artifacts for {run_dir}")
    embs, pids, splits, policy_names = result
    for split in ("ID", "OOD"):
        mask = splits == split
        mod._plot_one(
            subdir,
            f"{target}_{exp}",
            model,
            split,
            embs[mask],
            pids[mask],
            policy_names,
        )
PY
