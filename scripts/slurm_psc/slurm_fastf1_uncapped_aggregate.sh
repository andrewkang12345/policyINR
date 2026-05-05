#!/bin/bash
#SBATCH -N 1
#SBATCH -A cis260099p
#SBATCH -p RM-shared
#SBATCH --qos=low
#SBATCH -t 01:00:00
#SBATCH -J inr-fastf1-agg
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/fastf1-full-agg-%j.out

set -euo pipefail
PROJECT=/ocean/projects/cis260099p
REPO=$PROJECT/akang3/INR
cd "$REPO"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/akang3/polymer/envs/inr_torch"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export INR_LOG_LEVEL=INFO

OUT_ROOT="$REPO/outputs/fastf1_uncapped_full_suite_20260501"
python -m eval.summary "$OUT_ROOT" --out "$OUT_ROOT/aggregate.csv" --md "$OUT_ROOT/aggregate.md"
