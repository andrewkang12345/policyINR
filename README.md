# INR — Robust Policy Representation Learning from Offline Data

Offline policy-representation research codebase. The training signal is
purely offline behavioral data — no environment rollouts, no policy
labels at training time (labels are reserved for evaluation only).
The research target is the **robustness of policy representations under
state-distribution shift** between train- and test-time episodes,
measured across continuous-control (MuJoCo, DROID), discrete-action
visual (DMLab), discrete-action symbolic (Lichess) and time-series
(FastF1) domains.

```
INR/
├── configs/         Hydra YAML configs (data / model / experiment / eval)
├── data/            Dataset interfaces — synthetic GRF, Minari, custom
│                    MuJoCo, DMLab, Lichess, DROID, FastF1
├── models/          CVAE, INR-Transformer (history-conditioned, fitted-
│                    latent, infer-latent / MAML), INR-Diffusion
├── train/           Hydra entrypoint + generic Trainer
├── eval/            linear/kNN probe, generative metrics, summary/aggregator
├── scripts/         smoke + suite launchers (run_*.sh) and tooling
│   └── slurm_psc/   PSC Bridges-2 specific SLURM scripts (site-bound)
├── utils/           seed, registries, jsonl logger, checkpoint I/O
├── outputs/         per-run config/metrics/aggregates (checkpoints on HF)
└── paths.txt        env-var bootstrap; source before any run
```

Checkpoints (`best.pt`, `last.pt`) for every run live on Hugging Face:
**[`andrewkang12345/policyINR-checkpoints`](https://huggingface.co/andrewkang12345/policyINR-checkpoints)**.
The repository tree mirrors `outputs/` exactly — see §5.

---

## 1. Install

```bash
git clone https://github.com/andrewkang12345/policyINR.git
cd policyINR

# Python 3.11 recommended. Torch must have CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "gymnasium[mujoco]" minari h5py "hydra-core>=1.3" omegaconf \
            "stable-baselines3>=2.2" "sb3-contrib>=2.2" huggingface-hub \
            scikit-learn matplotlib numpy pandas tqdm fastf1 chess
```

Bootstrap caches and the `PYTHONPATH`. Everything is keyed off
`INR_DATA_ROOT` — the repo itself is location-agnostic.

```bash
# defaults: $INR_ROOT = repo dir, $INR_DATA_ROOT = $HOME
source paths.txt

# or override the data root first
export INR_DATA_ROOT=/scratch/$USER/INR_data
source paths.txt
```

| env var | default | purpose |
|---|---|---|
| `INR_ROOT` | dir of `paths.txt` | repo root (added to `PYTHONPATH`) |
| `INR_DATA_ROOT` | `$HOME` | parent for all caches |
| `MINARI_DATASETS_PATH` | `$INR_DATA_ROOT/.minari/datasets` | Minari dataset store |
| `INR_MUJOCO_CHECKPOINT_CACHE` | `$INR_DATA_ROOT/.cache/INR/mujoco_checkpoints` | Farama policy zips |
| `INR_CUSTOM_MUJOCO_CACHE` | `$INR_DATA_ROOT/.cache/INR/custom_mujoco` | custom-mujoco `.npz` |
| `INR_MINARI_CACHE` | `$INR_DATA_ROOT/.cache/INR/minari` | Minari `.npz` re-pack |
| `INR_DROID_CACHE` | `$INR_DATA_ROOT/.cache/INR/droid` | DROID lowdim arrays |
| `INR_FASTF1_CACHE` | `$INR_DATA_ROOT/.cache/INR/fastf1` | FastF1 session telemetry |
| `INR_LICHESS_CACHE` | `$INR_DATA_ROOT/.cache/INR/lichess` | Lichess PGNs |
| `INR_DMLAB_CACHE` | `$INR_DATA_ROOT/.cache/INR/dmlab` | DMLab episode `.npz` |
| `INR_RLU_DMLAB_CACHE` | `$INR_DATA_ROOT/.cache/INR/rlu_dmlab` | RLU DMLab tfrecord shards |
| `HF_HOME` | `$INR_DATA_ROOT/.cache/huggingface` | HuggingFace hub cache |

The same env vars are honored both by the Python data modules
(`data/{droid,fastf1,dmlab}.py`) and by the Hydra YAMLs
(`configs/data/*.yaml`) via `${oc.env:...}` interpolation, so you can
override any individual cache without touching the rest.

---

## 2. Smoke test (~20 s)

```bash
bash scripts/smoke_test.sh                # synthetic GRF, all 4 model families
bash scripts/smoke_test_new_datasets.sh   # lichess + dmlab smoke
```

The first writes to `outputs/smoke/`, exercises data → train → ckpt →
representation → linear probe → generative eval → summary. Passes when
every run produces a `summary.json` with finite `gen_mse` and a non-null
`probe_acc`.

---

## 3. Replicating the published runs

Every canonical experiment is launched by a single shell script under
`scripts/`. Each script is portable: it reads `INR_*` env vars, pins
one GPU per worker via `CUDA_VISIBLE_DEVICES`, and writes per-run
artifacts to a deterministic path under `outputs/<domain>/<suite>/`.

| Domain | Canonical run | Launcher | Output dir | Notes |
|---|---|---|---|---|
| **lichess** | top-3 GMs, 2× episode horizon, hk=240 | `scripts/run_lichess_full_2Xepisode.sh` | `outputs/lichess/2x_hk240/` | 4 models × 7 exp × 1 seed; complete |
| lichess | fitted-latent ablation, latent=80 | `scripts/run_lichess_full_2Xepisode.sh MODELS=inr_transformer_fitted_latent LATENT_DIM=80` | `outputs/lichess/2x_hk240_latent80/` | 7 exp |
| lichess | fitted-latent ablation, latent=160 | same launcher, `LATENT_DIM=160` | `outputs/lichess/2x_hk240_latent160/` | 7 exp |
| lichess | hk=120 fitted-latent, latent=128 | derived from `run_full_suite_new_datasets.sh` | `outputs/lichess/hk120_latent128/` | 7 exp |
| lichess | infer-latent MAML (10-step inner, ES) | infer-latent MAML model on lichess | `outputs/lichess/hk120_infer_latent_maml_rerun/` | **canonical MAML rerun**; 7 exp |
| lichess | infer-latent meta-64 (partial) | meta variant, hk=120 | `outputs/lichess/hk120_infer_latent_meta64_partial/` | only 4/7 exp ran — kept as published partial |
| lichess | dmlab+lichess v2 combined sweep | `scripts/run_lichess_dmlab_v2.sh` | `outputs/lichess/dmlab_v2_combined/` (+ `dmlabseekavoid/dmlab_v2_combined/`) | 4 models × 7 exp × 2 seeds |
| lichess | baseline (full_suite_new_datasets) | `scripts/run_full_suite_new_datasets.sh` | `outputs/lichess/baseline_top3/` | with `_sa16` ablation |
| **dmlab** | seekavoid baseline | `scripts/run_full_suite_new_datasets.sh` | `outputs/dmlabseekavoid/baseline/` | full + sa16 |
| dmlab | combined v2 sweep | `scripts/run_lichess_dmlab_v2.sh` | `outputs/dmlabseekavoid/dmlab_v2_combined/` | with `_sa16` ablation |
| **droid** | shards707, balanced, min300, hk=300 (incl. MAML) | `scripts/run_droid_balanced_min300_remove_suite.sh` | `outputs/droid/balanced_min300_remove_hk300/` | 5 models × 7 exp × 2 seeds; MAML 12/14 (2 incomplete on `specialization`) |
| droid | shards707, all min300, hk=300 | `scripts/run_droid_full_min300_after_maml.sh` | `outputs/droid/all_min300_hk300/` | 4 models × 7 exp × 2 seeds |
| droid | shards80, 10×, min200, hk=200, mat=512 | `scripts/run_droid_10x_min200_suite_full_util.sh` | `outputs/droid/10x_min200_hk200_mat512/` | 5 models × 7 exp × 2 seeds |
| droid | shards80, 10×, min8, hk=300 | `scripts/run_droid_10x_suite.sh` | `outputs/droid/10x_shards80_min8_hk300/` | 5 models × 7 exp × 2 seeds |
| droid | joint droid+fastf1 small subset | `scripts/run_droid_fastf1_full_suite.sh` | `outputs/droid/joint_droid_fastf1_small_subset/` | early joint sweep |
| **fastf1** | uncapped full suite | `scripts/run_fastf1_uncapped_suite.sh` | `outputs/fastf1/uncapped_full_suite/` | 5 models × 7 exp × 2 seeds |
| **mujoco** | custom_mujoco baseline (5 envs) | `scripts/run_custom_mujoco_suite.sh` | `outputs/mujoco/baseline_custom_mujoco/` | 5 envs × 4 models × 7 exp × 2 seeds |
| mujoco | minari baseline | `scripts/run_full_suite.sh` | `outputs/mujoco/baseline_minari_full_suite/` | 5 envs × 4 models × 7 exp × 2 seeds |
| mujoco | state-resampled v2 | `scripts/run_state_resampled_v2_suite.sh` | `outputs/mujoco/suites/state_resampled_v2/` | 5 envs |
| mujoco | action-resampled v2 / v3 / v4 | `scripts/run_action_resampled_v{2,3,4}_suite.sh` | `outputs/mujoco/suites/action_resampled_v{2,3,4}/` | each across 5 envs |
| mujoco | action-resampled v4 hopper-10× / 20× | `scripts/run_action_resampled_v5_suite.sh` (10×); `slurm_psc/slurm_hopper20x_*.sh` (20×) | `outputs/mujoco/suites/action_resampled_v4_hopper{10x,20x}/` | extended-horizon scaling study |
| **syntheticgrf** | base | `scripts/run_full_suite.sh` (synthetic_grf rows) | `outputs/syntheticgrf/baseline_full_suite/` | sanity / smoke |
| syntheticgrf | 10× horizon | `scripts/run_full_suite_new_datasets.sh` (synthetic_grf_10x rows) | `outputs/syntheticgrf/baseline_10x/` | scaling sanity |

Every per-run directory carries `config.yaml`, `metrics.jsonl`,
`summary.json`, `eval.json`, and an `stdout.log`. Per-suite directories
also have `aggregate.csv` and `aggregate.md`. The `run_dir` column in
each `aggregate.csv` is now repo-relative
(`outputs/<domain>/<suite>/<run>`), so the row points to the actual
artifacts in this checkout.

### Where new runs land

Future runs land in the right place by default — no manual moves
needed. The rules:

| invoked via | output goes to |
|---|---|
| `python -m train.main ...` (single run) | `outputs/oneoff/<timestamp>_<data>_<model>_<exp>_s<seed>/` |
| `python -m train.main -m ...` (Hydra multirun) | `outputs/oneoff/multirun/<timestamp>/<job>/` |
| any single-domain `scripts/run_*_suite.sh` | the matching `outputs/<domain>/<suite>/` listed in §3 |
| cross-domain sweep (`run_full_suite.sh`, `run_full_suite_new_datasets.sh`, `run_lichess_dmlab_v2.sh`, `run_droid_fastf1_full_suite.sh`) | `outputs/_combined/<sweep>/` — split into the domain tree afterward via `python scripts/split_combined_outputs.py outputs/_combined/<sweep>` (use `--dry-run` first) |
| `scripts/multi_gpu_launch.py` directly without `--out-root` | `outputs/_combined/manual_launch/` (override `--out-root` to a domain path for canonical sweeps) |

`outputs/_combined/` is gitignored — it's a staging area. Run the
splitter to promote those runs into the canonical domain tree before
they're picked up by `eval/summary.py` or aggregated into a
suite-level `aggregate.csv`.

### One-off Hydra runs

```bash
# minimal
python -m train.main data=minari_hopper model=inr_transformer_history_conditioned \
    experiment=single_shift seed=0 train.epochs=30

# multirun: seeds × models, sequential
python -m train.main -m data=minari_hopper \
    model=cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent \
    experiment=no_shift,single_shift seed=0,1

# DROID
python -m train.main data=droid_lowdim_full_balanced_min300_remove \
    model=inr_transformer_infer_latent_maml experiment=novel_generalization seed=0

# FastF1 stints
python -m train.main data=fastf1_stint_full_uncapped \
    model=inr_transformer_history_conditioned experiment=single_shift seed=0
```

---

## 4. Custom MuJoCo benchmark — design and suites

The repository ships five domains worth of generators / loaders. The
flagship benchmark for state-shift robustness is the **custom MuJoCo
shared-sequence suite** (`outputs/mujoco/suites/`).

Common generation path (per env):

1. download published checkpoints `farama-minari/<Env>-v5-<ALGO>-<policy>`
   (e.g. Hopper-v5-SAC-{simple,medium,expert})
2. collect pooled simulator states `[qpos, qvel]`
3. fit a suite-specific state / action sampler
4. roll out from sampled initial states for every policy
5. write a Minari-format dataset with per-episode `ID` / `OOD` and (v3/v4/v5)
   `predefined_partition` ∈ {train, val, test} tags

Build for one env+suite:

```bash
python scripts/build_custom_mujoco.py \
  --single-env hopper --generation-mode action_resampled_v5 \
  --dataset-id inr_mujoco_action_resampled_v5/hopper/controlled-v0 \
  --episode-horizon 1280 --force-rebuild
```

All 5 envs across N GPUs:

```bash
python scripts/build_custom_mujoco.py \
  --envs ant,halfcheetah,hopper,humanoid,walker2d \
  --generation-mode action_resampled_v5 --episode-horizon 1280 \
  --n-gpus 4 --out-root outputs/mujoco/suites/action_resampled_v5/build \
  --force-rebuild
```

### Suite variants

| suite | scoring | tail kept as ID | tail kept as OOD | horizon | source-state pool |
|---|---|---|---|---|---|
| `state_resampled_v2` | kNN-density in state space | low-distance | high-distance | 128 | shared, fixed-sequence |
| `action_resampled_v2` | per-state cross-policy action disagreement | low-disagreement | high-disagreement | 128 | shared, fixed-sequence |
| `action_resampled_v3` | per-episode mean action disagreement | high-disagreement | low-disagreement | 128 | **disjoint** train/val/test pools |
| `action_resampled_v4` | per-episode disagreement w/ absolute threshold | real Minari (`simple`/`medium`/`expert`) | synthetic, low-disagreement tail | 128 | disjoint |
| `action_resampled_v4_hopper10x` | as v4 | as v4 | as v4 | **1280** | disjoint |
| `action_resampled_v4_hopper20x` | as v4 | as v4 | as v4 | **2560** | disjoint |

The shared-sequence reuse property: within each env+split, every policy
is queried on the same sampled states in the same order — removing the
policy-specific state sampler used by v1.

---

## 5. Checkpoints on Hugging Face

Repo: **[`andrewkang12345/policyINR-checkpoints`](https://huggingface.co/andrewkang12345/policyINR-checkpoints)**

The repo mirrors `outputs/` exactly. To download every checkpoint for
the canonical lichess 2×-episode sweep, for example:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="andrewkang12345/policyINR-checkpoints",
    repo_type="model",
    allow_patterns=["lichess/2x_hk240/**/*.pt"],
    local_dir="outputs",  # files land at outputs/lichess/2x_hk240/<run>/best.pt
)
```

Or a single run:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="andrewkang12345/policyINR-checkpoints",
    filename="droid/balanced_min300_remove_hk300/"
             "droid_lowdim_full_balanced_min300_remove__"
             "inr_transformer_infer_latent_maml__novel_generalization__s0/best.pt",
)
```

Every leaf run carries `best.pt` (lowest val loss) and `last.pt`
(final epoch). To rebuild a representation given a downloaded
checkpoint and the run's `config.yaml`:

```python
from omegaconf import OmegaConf
from train.main import _build_base_store, _device, MODELS
import torch
cfg = OmegaConf.load("outputs/.../config.yaml")
model = MODELS.build(cfg.model.name, cfg=cfg.model).to(_device(cfg))
model.load_state_dict(torch.load("outputs/.../best.pt", map_location="cpu"))
model.eval()
```

`outputs/_archive/` (smoke / debug / superseded variants) and `outputs/_logs/`
(driver / rerun logs) are **not** uploaded to HF and are gitignored.

---

## 6. Metrics

Each run's `summary.json` carries:

- `train.best_val_loss`, `train.wallclock_s`
- `shift_strength` — per-policy std-normalized ID↔OOD centroid distance
- `eval.probe_acc` / `eval.probe_acc_seen` — linear-probe policy ID accuracy
- `eval.knn_acc1`, `eval.knn_acc5` — kNN companion to the probe
- `eval.novel_mean_embed_dist` — L2 of novel-policy episode embedding to
  nearest seen-policy centroid
- `eval.gen_mse`, `eval.gen_rmse`, `eval.gen_nmse`, `eval.gen_median_se` —
  generative regression metrics. Read `gen_median_se` alongside `gen_nmse`;
  CVAE outputs are unbounded and a few large-but-finite predictions can
  inflate the mean.
- `eval.gen_acc`, `eval.gen_nll` — generative classification metrics for
  discrete-action runs (lichess, dmlab). MSE/NMSE are NaN by design here.
- `eval.finite_fraction`, `eval.target_var` — sanity / scale references.

Probe / kNN labels (`policy_id`) are **never** seen during training — they
exist purely for evaluation.

---

## 7. Design overview (condensed)

### Data interface (`data/base.py::PolicyDataset`)

Every dataset yields:
```
past_states (K, S), past_actions (K, A), current_state (S,),
next_action (A,), episode_id, unit_id, has_unit_latent,
policy_id, is_ood
```

- CVAE training: `shuffle_history=True` (bag-of-pairs encoder).
- INR training: `shuffle_history=False` (ordered preceding window of K).

Discrete-action datasets carry `action_kind="discrete"` end-to-end:
`PairEmbed` uses `nn.Embedding`, `ActionHead` produces logits +
cross-entropy, and the eval path reports `gen_acc` / `gen_nll`.

### Models (subclass `RepresentationModel`, exposing `forward`,
`extract_representation`, `predict_action`)

- **CVAE** — permutation-invariant Transformer encoder over past (s,a),
  conditioned on current state; `z=mu`.
- **INR-Transformer, history-conditioned** — encoder sees past history
  only; latent `z` drives a FiLM-modulated MLP that takes `current_state`.
- **INR-Diffusion, history-conditioned** — same factorization with a
  conditional epsilon-predictor (DDPM).
- **INR-Transformer, fitted-latent** — one learnable code per behavior
  unit; FiLM head predicts actions from `(state, z_unit)`. Unseen units
  optimize `z` only at eval time, with shared INR weights frozen.
- **INR-Transformer, infer-latent / MAML** — meta-learns a per-unit code
  via inner-loop adaptation on support past (s,a) pairs. The 10-step
  early-stopping variant (`maml10_es`) is the canonical lichess/droid
  configuration in `outputs/{lichess,droid}/.../`.

### Shifts (`data/shifts.py`)

Default: `shared_region`. kNN over pooled per-step states; high-
entropy steps (default top 25%) are tagged `shared`; OOD episodes are
restricted to shared-region steps. If `effective_shared < 0.25` the
shift transparently falls back to `per_policy_cluster`.

For checkpoint-generated MuJoCo and DROID datasets that already carry
explicit `predefined_partition` metadata, use
`shift.kind=predefined_split` — no shift reconstruction is needed.

### Experiments (`configs/experiment/<name>.yaml`)

Each lists per-policy `{train, test}` placements ∈ `{ID, OOD, NONE}`.
`data/splits.py::build_experiment_loaders` stitches train/val/test
loaders from each policy's ID/OOD partitions, honoring
`predefined_partition` tags when present.

The seven experiments swept across every domain:
`no_shift`, `new_policy`, `single_shift`, `conflation`,
`generalization`, `specialization`, `novel_generalization`.

---

## 8. Extending

- **New synthetic family**: `@STATE_GENERATORS.register(...)` /
  `@ACTION_POLICIES.register(...)` in `data/synthetic.py`,
  add `configs/data/<name>.yaml`.
- **New representation architecture**: subclass
  `models.base.RepresentationModel`, decorate with `@MODELS.register("foo")`,
  add `configs/model/foo.yaml`.
- **New shift type**: `@SHIFTS.register("foo")` in `data/shifts.py`, then
  `shift.kind=foo` on any run.
- **New eval metric**: add to `eval/`, call from `eval/runner.py::run_full_eval`.
- **New real dataset**: mirror `data/minari_data.py` / `data/lichess.py`
  — build an `EpisodeStore`, plug in via
  `train/main.py::_build_base_store` and a new `configs/data/<name>.yaml`.

---

## 9. Known caveats from the published runs

- Minari `simple/medium/expert` checkpoints for hopper, walker2d, and
  humanoid occupy near-disjoint state regions — no meaningful shared
  region exists at the step level. The shift honestly reports
  `effective_shared << 0.25` and falls back to `per_policy_cluster`;
  probe saturation on those rows is a **data property**, not a
  representation property. For ant, halfcheetah, synthetic, and the
  DROID/FastF1 / lichess / dmlab domains, probe accuracy drops
  meaningfully under `specialization` and other OOD-test experiments.
- INR-Transformer beats INR-Diffusion on `gen_nmse` almost everywhere.
  The diffusion subpolicy is intentionally a small conditional
  ε-predictor with few sampling steps; raising `n_diffusion_steps` /
  `n_sample_steps` would close the gap.
- The infer-latent **MAML** runs on DROID `balanced_min300_remove` are
  missing 2/14 cells (`specialization` × seeds {0, 1}); the rest of that
  suite is complete. This is documented in `aggregate.csv` (12 maml rows
  vs. 14 for other models). Fully reproducible with the matching launcher
  if rerun.
- `gen_nmse` under hard test-time OOD can still be inflated by a few
  large-but-finite CVAE outputs (no clip). Always pair with
  `gen_median_se`. INR variants are output-clipped.

---

## 10. What's not done yet

- Bigger scaling sweeps. Data configs expose `n_policies`,
  `episodes_per_policy`, `episode_length`; the `*_hopper10x` /
  `*_hopper20x` variants are the first step.
- Sample-diversity metrics for the diffusion policy. `predict_action`
  is currently deterministic for all models.
- Joint cross-domain (DROID + FastF1) representation transfer — only a
  small joint subset is published under `outputs/droid/joint_droid_fastf1_small_subset/`.
