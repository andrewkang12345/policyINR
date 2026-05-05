#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 00:45:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-hopper20x-gen
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/gen-%j.out

set -euo pipefail
PROJECT=/ocean/projects/cis260099p
REPO=$PROJECT/akang3/INR
cd "$REPO"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/akang3/polymer/envs/inr_torch"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MINARI_DATASETS_PATH="$PROJECT/akang3/INR_minari/datasets"
export INR_MUJOCO_CHECKPOINT_CACHE="$PROJECT/akang3/INR_cache/mujoco_checkpoints"
export INR_CUSTOM_MUJOCO_CACHE="$PROJECT/akang3/INR_cache/custom_mujoco_store"
export INR_MINARI_CACHE="$PROJECT/akang3/INR_cache/minari"
export HF_HOME="$PROJECT/akang3/INR_cache/hf"
export INR_LOG_LEVEL=INFO
export MUJOCO_GL=egl

mkdir -p "$INR_CUSTOM_MUJOCO_CACHE" "$INR_MINARI_CACHE" "$HF_HOME" "$REPO/outputs/slurm"

python scripts/build_custom_mujoco.py \
  --single-env hopper \
  --generation-mode action_resampled_v5 \
  --dataset-id inr_mujoco_action_resampled_v4_hopper20x/hopper/controlled-v0 \
  --episode-horizon 2560 \
  --id-episodes 256 \
  --ood-episodes 128 \
  --tail-fraction 0.4 \
  --partition-train-frac 0.7 \
  --partition-val-frac 0.15 \
  --acceptance-threshold 2.5 \
  --candidate-batch-size 2048 \
  --max-candidate-batches 128 \
  --probe-like-samples 8 \
  --probe-like-repeats 4 \
  --force-rebuild
