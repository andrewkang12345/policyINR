# INR: Robust Policy Representation Learning from Offline Data

Offline policy-representation research codebase. No environment, no
policy labels during training (labels are used only at eval time for
sanity checks). The main research target is robustness of policy
representations under **state-distribution shift** between train and
test episodes.

```
INR/
├── configs/         Hydra YAML configs
│   ├── config.yaml           # base config
│   ├── data/                 # synthetic_grf{,_small}, minari_{hopper,...}
│   ├── model/                # cvae, history-conditioned INR, fitted-latent INR
│   ├── experiment/           # no_shift, new_policy, single_shift,
│   │                         #   conflation, generalization,
│   │                         #   specialization, novel_generalization
│   └── eval/default.yaml
├── data/            dataset interface, synthetic GRF, Minari, shifts, splits
├── models/          CVAE and INR-family architectures
├── train/           Hydra entrypoint + generic Trainer
├── eval/            linear probe, generative metrics, summary/aggregator
├── scripts/         smoke-test + full-suite launchers (4-GPU)
├── utils/           seed, registries, jsonl logger, checkpoint I/O
└── outputs/         runtime: per-run checkpoints, metrics.jsonl,
                       eval.json, summary.json, aggregate.csv/.md
```

## 1. Install

```bash
cd /mnt/data/INR
pip install torch torchvision hydra-core omegaconf scikit-learn \
            minari gymnasium h5py pyyaml stable-baselines3 \
            sb3-contrib huggingface_hub huggingface_sb3 "gymnasium[mujoco]"
```

Torch is expected to have CUDA. GPUs are discovered through
`CUDA_VISIBLE_DEVICES`; the multi-GPU launcher pins one GPU per job.
Minari HDF5 files cache under `~/.minari/`; our own numpy bundle
caches under `~/.cache/INR/minari/`.

## 2. Smoke test (fast, first thing to run)

Verifies the whole pipeline — data → training → checkpoint → repr
extraction → linear probe → generative eval → summary.

```bash
bash scripts/smoke_test.sh
```

Runtime: ~20 s on a single GPU. Trains all four models (CVAE /
history-conditioned INR-transformer / history-conditioned INR-diffusion /
fitted-latent INR-transformer) on a tiny synthetic dataset for 2
epochs and aggregates the results.

Expected final output (the `== smoke summary ==` table):

```
| data                | model           | experiment | n | probe_acc | probe_acc_seen | novel_dist | gen_mse |
|---------------------|-----------------|------------|---|-----------|-----------------|-----------|---------|
| synthetic_grf_small | cvae            | no_shift   | 1 | 0.500     | 0.500           | -         | 0.477   |
| synthetic_grf_small | inr_diffusion_history_conditioned   | no_shift   | 1 | 0.500     | 0.500           | -         | 1.166   |
| synthetic_grf_small | inr_transformer_history_conditioned | no_shift   | 1 | 0.500     | 0.500           | -         | 0.303   |
```

Numbers are illustrative and will vary slightly; what matters is that
every run produces `summary.json` with finite `gen_mse` and a
non-null `probe_acc`. Individual run artifacts live under
`outputs/smoke/<run_name>/`.

## 3. Full suite (all 4 GPUs)

```bash
bash scripts/run_full_suite.sh
```

By default this sweeps:

- **datasets**: `synthetic_grf`, `minari_{hopper,halfcheetah,walker2d,ant,humanoid}`,
  `custom_mujoco_{hopper,halfcheetah,walker2d,ant,humanoid}`
- **models**: `cvae`, `inr_transformer_history_conditioned`,
  `inr_diffusion_history_conditioned`, `inr_transformer_fitted_latent`
- **experiments**: `no_shift`, `new_policy`, `single_shift`, `conflation`,
  `generalization`, `specialization`, `novel_generalization`
- **seeds**: `0,1`

Jobs are scheduled round-robin across `N_GPUS=4` via a worker-queue in
`scripts/multi_gpu_launch.py`, each as a subprocess with
`CUDA_VISIBLE_DEVICES` pinned to one GPU. Tunable via env:

```bash
N_GPUS=4 SEEDS=0,1,2 EPOCHS=20 BATCH=256 HISTORY_K=16 MAX_EPS=60 \
    bash scripts/run_full_suite.sh
```

Or restrict the sweep:

```bash
DATASETS=minari_hopper MODELS=cvae,inr_transformer_history_conditioned \
EXPERIMENTS=no_shift,single_shift SEEDS=0 \
    bash scripts/run_full_suite.sh
```

After the sweep completes, an aggregate CSV + markdown table is
written to `outputs/full_suite/aggregate.{csv,md}`. Each run's own
artifacts are under `outputs/full_suite/<data>__<model>__<exp>__s<seed>/`.

## 4. Running one-off experiments

```bash
# single Hydra run
python -m train.main data=minari_hopper model=inr_transformer_history_conditioned \
    experiment=single_shift seed=0 train.epochs=30

# hydra multirun — seeds x models, sequential
python -m train.main -m data=minari_hopper \
    model=cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent \
    experiment=no_shift,single_shift seed=0,1,2

# custom checkpoint-generated MuJoCo data
python -m train.main data=custom_mujoco_hopper model=inr_transformer_history_conditioned \
    experiment=no_shift shift.kind=predefined_split seed=0
```

## 4.1 Custom checkpoint-generated MuJoCo datasets

The repository can now build local Minari datasets from the published
Farama Minari policy checkpoints for:

- `Ant-v5` (`SAC`)
- `HalfCheetah-v5` (`TQC`)
- `Hopper-v5` (`SAC`)
- `Humanoid-v5` (`TQC`)
- `Walker2d-v5` (`SAC`)

Generation path:

1. download the published checkpoints from `farama-minari/<Env>-v5-<ALGO>-<policy>`
2. collect pooled reference simulator states from those checkpoints
3. fit a configurable simulator-state sampler
4. generate short rollouts from sampled initial states for every policy
5. write a local Minari dataset with per-episode `ID` / `OOD` tags

State-sampler details for the step-resampled custom datasets:

- The calibration pass records simulator states as concatenated `[qpos, qvel]`.
- Those pooled states are clustered with k-means.
- Each cluster gets:
  - a center
  - a per-dimension std
  - a cluster weight equal to its empirical mass, meaning the fraction of
    calibration states assigned to that cluster
- Clusters are ordered along the first principal direction of the cluster-center
  cloud, and by default:
  - the first half of the ordered clusters become `ID`
  - the remaining half become `OOD`
- Sampling then:
  - picks an `ID` or `OOD` cluster according to the normalized cluster weights
    inside that split
  - samples around that cluster center with configurable `qpos` / `qvel` noise
  - clips back to a padded min/max box from the calibration pool

That is the logic behind the `custom_mujoco_step_resampled_*` datasets.

There is also an action-resampled variant:

- start from the same shared reference state bag
- for each policy separately, evaluate its action on every reference state
- cluster those actions in action space
- split the action clusters into `ID` / `OOD` the same way as above
- sample reference states from the chosen action-cluster subset for that policy

That is the logic behind the `custom_mujoco_action_resampled_*` datasets.

Build all five custom datasets across 4 GPUs:

```bash
python scripts/build_custom_mujoco.py --n-gpus 4 --force-rebuild
```

Run the standard suite on the custom datasets using the generator-defined
ID/OOD tags instead of re-clustering the original trajectories:

```bash
bash scripts/run_custom_mujoco_suite.sh
```

Run the state-resampled state-distribution suite:

```bash
bash scripts/run_step_resampled_suite.sh
```

Run the action-resampled suite:

```bash
bash scripts/run_action_resampled_suite.sh
```

## 5. Metrics

Each run emits `summary.json` with:

- `train.best_val_loss`, `train.wallclock_s`
- `shift_strength`            — per-policy std-normalized ID↔OOD centroid
                                  distance (0 = no shift; typically 0.3–1.2)
- `eval.probe_acc`            — policy classification accuracy, all test policies
- `eval.probe_acc_seen`       — same, restricted to policies seen in training
- `eval.novel_mean_embed_dist`— mean L2 distance from novel-policy episode
                                  embeddings to the nearest seen-policy centroid
- `eval.gen_mse`, `eval.gen_rmse`   — raw MSE in normalized-action space
- `eval.gen_nmse`             — MSE / Var(target), scale-free:
                                  0 = perfect, 1 ≈ "predict the mean" baseline.
                                  Comparable across environments.
- `eval.gen_median_se`        — outlier-robust per-sample median SE.
- `eval.finite_fraction`      — fraction of predictions that were finite.
- `eval.target_var`           — denominator for gen_nmse (for reference).

All probe metrics use `policy_id` labels that are **never** used during
training — only for evaluation. **Read `gen_median_se` alongside
`gen_nmse`** — CVAE has no output bound, so under hard OOD it can
produce a few large-but-finite predictions that skew `gen_nmse`'s mean;
`gen_median_se` is the uninflated companion.

## 6. Design overview

### 6.1 Data interface

`data/base.py::PolicyDataset` yields a uniform dict:

```
past_states      (K, state_dim)
past_actions     (K, action_dim)
current_state    (state_dim,)
next_action      (action_dim,)
episode_id, unit_id, has_unit_latent, policy_id, is_ood
```

- **CVAE** training uses `shuffle_history=True` → K past pairs are
  sampled uniformly from the episode (the bag-of-pairs view specified
  for the encoder).
- **INR** training uses `shuffle_history=False` → past history is the
  ordered preceding window of length K; randomness comes from standard
  batch shuffling.

One episode generates up to `T-1` training examples (every step is a
possible "current" step). The shuffle/history behavior is controlled
from the model config field `shuffle_history_train`.

### 6.2 Models

All models subclass `RepresentationModel` (models/base.py) and expose:
```
forward(batch) -> dict(loss, ...)
extract_representation(batch) -> Tensor (B, latent_dim)
predict_action(batch) -> Tensor (B, action_dim)
```

- **CVAE** (`models/cvae.py`) — encoder=permutation-invariant
  Transformer on past (s,a); condition=current state; decoder=MLP on
  `(z, state)`; `z=mu` is the policy representation.
- **INR-Transformer, history-conditioned** (`models/inr_transformer.py`) —
  encoder sees past history only; latent code `z` drives a FiLM-modulated
  MLP head, while the current state is fed directly to that head.
- **INR-Diffusion, history-conditioned** (`models/inr_diffusion.py`) —
  same factorization: history-only encoder for `z`, then a conditional
  epsilon-predictor trained with DDPM on `(current_state, z)`.
- **INR-Transformer, fitted-latent** (`models/inr_transformer.py`) —
  one learned latent code is assigned per behavior unit (episode or fixed
  window). The FiLM-modulated INR head predicts actions from
  `(current_state, z_unit)`. For unseen units at eval time, the shared INR
  weights stay frozen and only the unit latent is optimized against support
  past state-action pairs from that same unit. In comments and metrics this
  is treated as a behavior-function code / trajectory-level latent, not a
  guaranteed canonical policy representation.

### 6.3 State-distribution shift

`data/shifts.py` registers shift strategies under a small `Registry`.
The default is **`shared_region`** — a step-level shared state region
across all policies, designed so the OOD partition does not leak
policy identity via state distribution.

How it works:
1. Pool every timestep's state across all policies; build a kNN index.
2. For each step, measure the entropy of its k neighbors' policy-id
   distribution. High entropy ⇒ near-uniform across policies ⇒
   state doesn't reveal policy.
3. Mark the top `density_quantile` (default 25%) of steps by entropy
   as *shared* — and store the per-episode bool mask.
4. An episode is OOD-eligible if it has enough shared steps; the
   `PolicyDataset` then restricts past-history and current-state
   sampling for OOD episodes to **only shared-region steps**.

When a dataset has genuinely shared state regions across policies
(e.g. Minari ant, halfcheetah, or the synthetic GRF), this produces
an OOD region where an encoder can only separate policies via the
action mapping, not via state distribution. When policies are nearly
state-disjoint (e.g. Minari hopper/walker2d/humanoid — simple/medium/
expert trajectories occupy different parts of state space), the
shift reports `effective_shared < 0.25` and transparently falls
back to `per_policy_cluster` (per-policy ID/OOD), rather than
fabricating a nominally-shared region that isn't actually shared.

Logged per run:
- `shift_strength[pid]`      per-policy ID-vs-OOD Mahalanobis distance
- `shift_overlap`             cross-policy dispersion of OOD centroids
- `shift_overlap_ratio`       OOD disp / ID disp (< 1 = more shared OOD)
- `effective_shared`          fraction of "shared" steps that actually
                               have cross-policy kNN neighbors
- `shift_fallback`            `none` or `per_policy_cluster`

`per_policy_cluster`, `per_policy_quantile`, and `mean_cluster` are
kept for ablation. Future **action**-distribution shift plugs in by
registering a new strategy and swapping `shift.kind`.

For the custom checkpoint-generated datasets, use
`shift.kind=predefined_split`. Those datasets already carry explicit
per-episode `ID` / `OOD` tags in their metadata, so no extra shift
reconstruction is needed.

### 6.4 Custom MuJoCo v2 datasets

The v2 MuJoCo suites are the cleaner constructions used for the
checkpoint-generated benchmark:

- `state_resampled_v2`
- `action_resampled_v2`
- `action_resampled_v3`
- `action_resampled_v4`

Both start from the same shared reference state pool:

1. Download the published `simple`, `medium`, and `expert` MuJoCo
   checkpoints for an env.
2. Run short calibration rollouts and collect simulator states
   `[qpos, qvel]`.
3. Freeze one shared state bag for the environment.

The split logic then differs by suite:

- **State-resampled v2** (`shared_state_density_v2`)
  - Score each shared reference state by kNN mean-distance in state
    space.
  - Define `ID` as the low-distance density tail and `OOD` as the
    high-distance density tail.
  - Sample exact fixed state sequences once per split and reuse those
    same sequences for every policy.

- **Action-resampled v2** (`shared_state_action_disagreement_v2`)
  - Evaluate all policies on the same shared reference state bag.
  - For each state, score action disagreement by mean pairwise action
    distance across policies.
  - Define `ID` as the low-disagreement tail and `OOD` as the
    high-disagreement tail.
  - Again, sample exact fixed state sequences once per split and reuse
    those same sequences for every policy.

- **Action-resampled v3** (`shared_state_action_mean_matched_v3`)
  - Evaluate all policies on the same shared reference state bag.
  - For each state, score action disagreement by mean pairwise action
    distance across policies.
  - Define `ID` as the high-disagreement tail and `OOD` as the
    low-disagreement tail.
  - Split each split-specific source-state pool into disjoint
    `train` / `val` / `test` partitions before any episode generation.
  - Sample many candidate fixed state sequences inside each partition,
    score them by cross-policy disagreement of the episode mean action,
    keep the lowest-disagreement sequences, and reuse those accepted
    sequences for every policy.

- **Action-resampled v4** (`minari_id_action_matched_ood_v4`)
  - `ID` episodes come directly from the original Minari MuJoCo datasets
    (`simple`, `medium`, `expert`) rather than from synthetic state
    sampling.
  - Only the `OOD` partition is synthetic: build a shared checkpoint
    reference-state bag, score per-state action disagreement across
    policies, keep the low-disagreement tail as the difficult OOD source
    pool, and split that pool into disjoint `train` / `val` / `test`
    partitions.
  - Generate many candidate OOD state sequences and accept only those
    whose cross-policy episode-summary disagreement is below an absolute
    threshold; continue searching until the desired OOD count is reached
    or fail loudly.

That exact shared-sequence reuse is the key design constraint: within
each env and split, every policy is queried on the same sampled
states, in the same order. This removes the policy-specific state
sampler used by the earlier v1 constructions. In v3, the underlying
source-state pools are also disjoint across train/val/test, so held-out
evaluation does not reuse the same source states.

Suite entrypoints:

- `scripts/run_state_resampled_v2_suite.sh`
- `scripts/run_action_resampled_v2_suite.sh`
- `scripts/run_action_resampled_v3_suite.sh`
- `scripts/run_action_resampled_v4_suite.sh`

Output roots:

- `outputs/suites/state_resampled_v2`
- `outputs/suites/action_resampled_v2`
- `outputs/suites/action_resampled_v3`
- `outputs/suites/action_resampled_v4`

### 6.5 Experiment composition

Each `configs/experiment/<name>.yaml` lists per-policy
`{train, test}` placements with values `ID | OOD | NONE`.
`data/splits.py::build_experiment_loaders` reads that spec and stitches
the final train/val/test stores from each policy's ID / OOD partitions.
If a dataset already carries `predefined_partition` metadata, those
episode-level train/val/test assignments are honored instead of
randomly re-splitting episodes.

## 7. Extending

- **New synthetic family**: register a generator in
  `data/synthetic.py` via `@STATE_GENERATORS.register(...)` or
  `@ACTION_POLICIES.register(...)`, then reference it in a new YAML
  under `configs/data/`.
- **New representation architecture**: subclass
  `models.base.RepresentationModel`, decorate with
  `@MODELS.register("my_model")`, and add `configs/model/my_model.yaml`.
- **New shift type**: `@SHIFTS.register("my_shift")` in
  `data/shifts.py`, then `shift.kind=my_shift` in any run.
- **New evaluation metric**: add a function in `eval/` and call it
  from `eval/runner.py::run_full_eval`.
- **New real dataset**: mirror `data/minari_data.py` — build an
  `EpisodeStore` and plug in through a new `configs/data/<name>.yaml`
  with `kind: <your_kind>`; add a branch in
  `train/main.py::_build_base_store`.

## 8. Gotchas

- Minari datasets download on first use to `~/.minari/`. The
  `humanoid` set in particular is large (~8 GB). Point elsewhere via
  `MINARI_DATASETS_PATH`, or move `~/.minari` to a bigger filesystem
  and symlink — this repo's smoke test uses a small synthetic config
  instead to avoid the download.
- `nan` for `probe_acc_seen` or `novel_mean_embed_dist` just means the
  experiment has no seen / no novel policies in the test partition.
- The smoke configs intentionally use very few episodes, so
  linear-probe accuracy is noisy. Meaningful separation between models
  shows up only in the full suite's aggregate.

## 9. Known caveats from the full-suite run

- Minari `simple`, `medium`, `expert` checkpoints for hopper,
  walker2d, and humanoid occupy near-disjoint state regions — there
  is no meaningful shared state distribution we can carve out at the
  step level. On these envs the shift honestly reports
  `effective_shared << 0.25` and falls back to `per_policy_cluster`;
  probe saturation on those rows is a **data property**, not a
  representation property. For envs where a shared region does exist
  (ant, halfcheetah, synthetic), probe accuracy drops meaningfully
  under `specialization` and other OOD-test experiments.
- INR-Transformer out-performs INR-Diffusion on `gen_nmse` almost
  everywhere. The diffusion subpolicy was deliberately a small
  conditional epsilon-predictor with few sampling steps; higher
  `n_diffusion_steps` / `n_sample_steps` would likely close the gap.
- `gen_nmse` under strong test-time OOD can still be skewed by a
  handful of large-but-finite CVAE outputs (no clip); always pair it
  with `gen_median_se`. INR variants are output-clipped.

## 10. Additional datasets (discrete action, featurized state)

Two new datasets extend the same `EpisodeStore`/`PolicyDataset` API
to image and symbolic observations with discrete action spaces:

- `dmlab_seekavoid` — RL Unplugged DMLab `seekavoid_arena_01`, 3
  policies = {snapshot_0_eps_0.0, snapshot_1_eps_0.0, snapshot_0_eps_0.25}.
  State: 72x96 RGB + last_action (onehot) + last_reward → 128-d random
  CNN + 15 + 1 = 144-d. Action: 15 discrete. Shards are pulled
  directly from the public `gs://rl_unplugged/dmlab` bucket over HTTPS
  (skipping TFDS's Beam-backed local rebuild), so first use costs only
  a few hundred MB.
- `lichess_top3` — Lichess games of three active-export GMs
  (lance5500, Zhigalko_Sergei, penguingim1). State: 8×8×12 piece-plane
  bit encoding + side/castling/ep/move-count metadata → 783-d. Action:
  UCI-move vocabulary derived from the union of observed moves (~1700
  classes). PGNs are fetched via Lichess's public games export API.
Discrete actions are carried end-to-end: `EpisodeStore.action_kind`,
`PairEmbed` uses an `nn.Embedding`, `ActionHead` produces logits +
cross-entropy loss, `predict_action` returns argmax, and the
generative eval path reports `gen_acc` and `gen_nll` (mean cross-entropy)
with per-class breakdown. `gen_mse`/`gen_nmse` are NaN for discrete
runs by design and the aggregator does not flag them as degenerate.

Entry points:

```bash
# ~60s smoke test: 2 datasets × 3 models × 1 experiment
bash scripts/smoke_test_new_datasets.sh

# full sweep: 2 datasets × 3 models × 7 experiments × 2 seeds
bash scripts/run_full_suite_new_datasets.sh
```

## 11. What's not done yet

- Action-distribution shift (only state shift for now) — hooks in
  place via the shift registry.
- Larger scaling studies — data configs expose `n_policies`,
  `episodes_per_policy`, `episode_length` so this slots in.
- Richer generative metrics (e.g. sample diversity for the diffusion
  policy). Currently `predict_action` is deterministic for all models.
