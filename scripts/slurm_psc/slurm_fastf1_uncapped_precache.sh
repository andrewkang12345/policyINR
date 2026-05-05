#!/bin/bash
#SBATCH -N 1
#SBATCH -A cis260099p
#SBATCH -p RM-shared
#SBATCH --qos=low
#SBATCH -t 04:00:00
#SBATCH -J inr-fastf1-cache
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/fastf1-cache-%j.out

set -euo pipefail
PROJECT=/ocean/projects/cis260099p
REPO=$PROJECT/akang3/INR
cd "$REPO"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/akang3/polymer/envs/inr_torch"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export INR_FASTF1_CACHE="$PROJECT/akang3/INR_cache/fastf1"
export INR_LOG_LEVEL=INFO

mkdir -p "$REPO/outputs/slurm" "$INR_FASTF1_CACHE"

if ! python - <<'PY'
import fastf1
PY
then
  python -m pip install --user fastf1
fi

python - <<'PY'
from omegaconf import OmegaConf
from train.main import _build_base_store
cfg = OmegaConf.load("configs/data/fastf1_stint_full_uncapped.yaml")
cfg.cache_dir = "/ocean/projects/cis260099p/akang3/INR_cache/fastf1"
store = _build_base_store(cfg)
print("episodes", len(store), "state_dim", store.state_dim, "action_dim", store.action_dim)
print("total_pairs", sum(len(s) for s in store.states))
PY
