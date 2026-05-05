#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-lichess-2x-tsne
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/lichess-2x-tsne-%j.out

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

run_tsne() {
  python - <<'PY'
from pathlib import Path
import scripts.plot_policy_tsne as mod

target = "lichess_full_2Xepisode"
for subdir, data_label, runs_root, data_cfg in mod.RUN_GROUPS:
    if data_label != target:
        continue
    for model in mod.MODELS:
        run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
        print(f"[tsne] {run_dir}", flush=True)
        result = mod._extract(run_dir)
        if result is None:
            raise RuntimeError(f"missing run artifacts for {run_dir}")
        embs, pids, splits, policy_names = result
        mod.plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names)
PY
}

if ! run_tsne; then
  echo "[slurm] tsne attempt failed; retrying once" >&2
  sleep 15
  run_tsne
fi
