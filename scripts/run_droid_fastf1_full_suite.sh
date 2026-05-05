#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export INR_LOG_LEVEL="${INR_LOG_LEVEL:-INFO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

N_GPUS="${N_GPUS:-0}"
SEEDS="${SEEDS:-0,1}"
OUT_ROOT="${OUT_ROOT:-outputs/droid_fastf1_full_suite}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-4}"
MODELS="${MODELS:-cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent,inr_transformer_infer_latent_maml}"
EXPERIMENTS="${EXPERIMENTS:-no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization}"

mkdir -p "${OUT_ROOT}"

if [[ "${N_GPUS}" == "0" ]]; then
  echo "N_GPUS=0: building/downloading caches only; no training will run."
  python - <<'PY'
from omegaconf import OmegaConf
from train.main import _build_base_store
for name in ["droid_lowdim_full", "fastf1_stint_full"]:
    cfg = OmegaConf.load(f"configs/data/{name}.yaml")
    store = _build_base_store(cfg)
    lens = [len(s) for s in store.states]
    print(name, "episodes", len(store), "state_dim", store.state_dim, "action_dim", store.action_dim,
          "min_len", min(lens), "max_len", max(lens))
    print("policies", sorted(set(m.policy_id for m in store.meta)),
          "ood", sum(int(m.is_ood) for m in store.meta))
PY
  exit 0
fi

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "droid_lowdim_full" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.history_k=300"

python scripts/multi_gpu_launch.py \
  --n-gpus "${N_GPUS}" \
  --seeds "${SEEDS}" \
  --out-root "${OUT_ROOT}" \
  --datasets "fastf1_stint_full" \
  --models "${MODELS}" \
  --experiments "${EXPERIMENTS}" \
  --overrides \
      "shift.kind=predefined_split" \
      "train.epochs=${EPOCHS}" \
      "train.batch_size=${BATCH}" \
      "train.num_workers=${WORKERS}" \
      "train.history_k=800"

python -m eval.summary "${OUT_ROOT}" --out "${OUT_ROOT}/aggregate.csv" --md "${OUT_ROOT}/aggregate.md"
