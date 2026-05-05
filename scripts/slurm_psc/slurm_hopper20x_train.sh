#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 08:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-hopper20x
#SBATCH --array=0-55
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/train-%A_%a.out

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

MODELS=(cvae inr_transformer_history_conditioned inr_diffusion_history_conditioned inr_transformer_fitted_latent)
EXPS=(no_shift new_policy single_shift conflation generalization specialization novel_generalization)
SEEDS=(0 1)

IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % 2 ))]}
EXP_IDX=$(( (IDX / 2) % 7 ))
MODEL_IDX=$(( IDX / 14 ))
MODEL=${MODELS[$MODEL_IDX]}
EXP=${EXPS[$EXP_IDX]}

RUN_NAME="custom_mujoco_action_resampled_v4_hopper20x__${MODEL}__${EXP}__s${SEED}"
OUT_DIR="$REPO/outputs/suites/action_resampled_v4_hopper20x/runs/custom_mujoco_action_resampled_v4_hopper20x/$RUN_NAME"
mkdir -p "$OUT_DIR"

python -m train.main \
  data=custom_mujoco_action_resampled_v4_hopper20x \
  model="$MODEL" \
  experiment="$EXP" \
  seed="$SEED" \
  run_name="$RUN_NAME" \
  output_dir="$OUT_DIR" \
  shift.kind=predefined_split \
  train.epochs=10 \
  train.batch_size=32 \
  train.history_k=320 \
  ++data.max_episodes_per_policy=384
