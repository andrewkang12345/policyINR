#!/bin/bash
#SBATCH -N 1
#SBATCH -A cis260099p
#SBATCH -p GPU-shared
#SBATCH --qos=gpu
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J inr-fastf1-full
#SBATCH --array=0-69
#SBATCH -o /ocean/projects/cis260099p/akang3/INR/outputs/slurm/fastf1-full-%A_%a.out

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

MODELS=(
  cvae
  inr_transformer_history_conditioned
  inr_diffusion_history_conditioned
  inr_transformer_fitted_latent
  inr_transformer_infer_latent_maml
)
EXPS=(no_shift new_policy single_shift conflation generalization specialization novel_generalization)
SEEDS=(0 1)

IDX=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( IDX % 2 ))]}
EXP_IDX=$(( (IDX / 2) % 7 ))
MODEL_IDX=$(( IDX / 14 ))
MODEL=${MODELS[$MODEL_IDX]}
EXP=${EXPS[$EXP_IDX]}

RUN_NAME="fastf1_stint_full_uncapped__${MODEL}__${EXP}__s${SEED}"
OUT_DIR="$REPO/outputs/fastf1_uncapped_full_suite_20260501/$RUN_NAME"
mkdir -p "$OUT_DIR"

python -m train.main \
  data=fastf1_stint_full_uncapped \
  model="$MODEL" \
  experiment="$EXP" \
  seed="$SEED" \
  run_name="$RUN_NAME" \
  output_dir="$OUT_DIR" \
  data.cache_dir="$INR_FASTF1_CACHE" \
  shift.kind=predefined_split \
  train.epochs=30 \
  train.batch_size=128 \
  train.num_workers=4 \
  train.history_k=800
