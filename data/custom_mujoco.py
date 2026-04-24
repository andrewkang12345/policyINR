"""Custom MuJoCo dataset generation from published policy checkpoints.

This module builds local Minari datasets by:
  1. downloading published Farama Minari policy checkpoints,
  2. collecting a pooled reference set of simulator states,
  3. fitting a configurable simulator-state or action-conditioned sampler,
  4. collecting either rollout episodes or per-step resampled states, and
  5. writing a local Minari dataset with explicit per-episode ID/OOD tags.

The resulting dataset avoids the original simple/medium/expert trajectory
support mismatch by enforcing the same state-distribution family across all
policies. Downstream experiments can then consume the predefined ID/OOD split
via shift.kind=predefined_split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import os

import gymnasium as gym
import minari
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from minari import create_dataset_from_buffers
from minari.data_collector.episode_buffer import EpisodeBuffer
from sb3_contrib import TQC
from stable_baselines3 import SAC

from .base import EpisodeMeta, EpisodeStore
from .minari_data import MINARI_ENVS


@dataclass(frozen=True)
class MujocoEnvSpec:
    env_key: str
    env_id: str
    repo_env: str
    algo: str
    policies: Tuple[str, ...]


MUJOCO_ENVS: Dict[str, MujocoEnvSpec] = {
    "ant": MujocoEnvSpec(
        env_key="ant",
        env_id="Ant-v5",
        repo_env="Ant",
        algo="SAC",
        policies=("simple", "medium", "expert"),
    ),
    "halfcheetah": MujocoEnvSpec(
        env_key="halfcheetah",
        env_id="HalfCheetah-v5",
        repo_env="HalfCheetah",
        algo="TQC",
        policies=("simple", "medium", "expert"),
    ),
    "hopper": MujocoEnvSpec(
        env_key="hopper",
        env_id="Hopper-v5",
        repo_env="Hopper",
        algo="SAC",
        policies=("simple", "medium", "expert"),
    ),
    "humanoid": MujocoEnvSpec(
        env_key="humanoid",
        env_id="Humanoid-v5",
        repo_env="Humanoid",
        algo="TQC",
        policies=("simple", "medium", "expert"),
    ),
    "walker2d": MujocoEnvSpec(
        env_key="walker2d",
        env_id="Walker2d-v5",
        repo_env="Walker2d",
        algo="SAC",
        policies=("simple", "medium", "expert"),
    ),
}

ALGO_LOADERS = {
    "SAC": SAC,
    "TQC": TQC,
}


def _safe_dataset_slug(dataset_id: str) -> str:
    digest = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest()[:10]
    return dataset_id.replace("/", "__") + "__" + digest


def _store_cache_path(dataset_id: str) -> Path:
    root = Path(os.environ.get("INR_CUSTOM_MUJOCO_CACHE", Path.home() / ".cache" / "INR" / "custom_mujoco"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_safe_dataset_slug(dataset_id)}.npz"


def _checkpoint_cache_root() -> Path:
    root = Path(os.environ.get("INR_MUJOCO_CHECKPOINT_CACHE", Path.home() / ".cache" / "INR" / "mujoco_checkpoints"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_dataset_id(env_key: str, mode: str = "rollout_episode") -> str:
    mode = str(mode)
    if mode == "rollout_episode":
        return f"inr_mujoco/{env_key}/controlled-v0"
    if mode == "resampled_steps":
        return f"inr_mujoco_step_resampled/{env_key}/controlled-v0"
    if mode == "action_resampled_steps":
        return f"inr_mujoco_action_resampled/{env_key}/controlled-v0"
    if mode == "state_resampled_v2":
        return f"inr_mujoco_state_resampled_v2/{env_key}/controlled-v0"
    if mode == "state_resampled_v3":
        return f"inr_mujoco_state_resampled_v3/{env_key}/controlled-v0"
    if mode == "action_resampled_v2":
        return f"inr_mujoco_action_resampled_v2/{env_key}/controlled-v0"
    if mode == "action_resampled_v3":
        return f"inr_mujoco_action_resampled_v3/{env_key}/controlled-v0"
    if mode == "action_resampled_v4":
        return f"inr_mujoco_action_resampled_v4/{env_key}/controlled-v0"
    if mode == "action_resampled_v5":
        return f"inr_mujoco_action_resampled_v5/{env_key}/controlled-v0"
    raise KeyError(f"Unknown custom MuJoCo mode '{mode}'")


def _as_serializable(x):
    if isinstance(x, dict):
        return {str(k): _as_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_serializable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _policy_order(env_spec: MujocoEnvSpec, policies: Optional[Sequence[str]]) -> List[str]:
    if policies is None:
        return list(env_spec.policies)
    wanted = [str(p) for p in policies]
    for p in wanted:
        if p not in env_spec.policies:
            raise KeyError(f"Unsupported policy '{p}' for {env_spec.env_key}. Choices: {env_spec.policies}")
    return wanted


def _repo_id(env_spec: MujocoEnvSpec, policy_name: str) -> str:
    return f"farama-minari/{env_spec.repo_env}-v5-{env_spec.algo}-{policy_name}"


def _checkpoint_filenames(env_spec: MujocoEnvSpec, policy_name: str) -> List[str]:
    stem = f"{env_spec.repo_env.lower()}-v5"
    algo_upper = env_spec.algo.upper()
    algo_lower = env_spec.algo.lower()
    return [
        f"{stem}-{algo_upper}-{policy_name}.zip",
        f"{stem}-{algo_lower}-{policy_name}.zip",
    ]


def download_policy_checkpoint(env_key: str, policy_name: str) -> Path:
    env_spec = MUJOCO_ENVS[env_key]
    repo_id = _repo_id(env_spec, policy_name)
    cache_dir = _checkpoint_cache_root() / env_key / policy_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for filename in _checkpoint_filenames(env_spec, policy_name):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(cache_dir))
            return Path(path)
        except Exception as e:  # hub mixes concrete error types
            errors.append(f"{filename}: {e}")

    files = list_repo_files(repo_id)
    zip_files = [f for f in files if f.endswith(".zip")]
    if not zip_files:
        raise RuntimeError(f"No .zip checkpoint found in repo {repo_id}. Errors: {errors}")
    path = hf_hub_download(repo_id=repo_id, filename=zip_files[0], local_dir=str(cache_dir))
    return Path(path)


def load_policy_model(env_key: str, policy_name: str, *, device: str = "cpu"):
    env_spec = MUJOCO_ENVS[env_key]
    cls = ALGO_LOADERS[env_spec.algo]
    ckpt = download_policy_checkpoint(env_key, policy_name)
    return cls.load(str(ckpt), device=device)


def _make_env(env_key: str):
    return gym.make(MUJOCO_ENVS[env_key].env_id)


def _sim_state(env) -> np.ndarray:
    u = env.unwrapped
    return np.concatenate([u.data.qpos.copy(), u.data.qvel.copy()]).astype(np.float32)


def _qpos_qvel_sizes(env) -> Tuple[int, int]:
    u = env.unwrapped
    return int(u.model.nq), int(u.model.nv)


def _set_sim_state(env, sim_state: np.ndarray):
    nq, nv = _qpos_qvel_sizes(env)
    qpos = np.asarray(sim_state[:nq], dtype=np.float64)
    qvel = np.asarray(sim_state[nq:nq + nv], dtype=np.float64)
    u = env.unwrapped
    u.set_state(qpos, qvel)
    if hasattr(u.data, "time"):
        u.data.time = 0.0
    if hasattr(u, "_get_obs"):
        obs = u._get_obs()
    else:
        obs = env.observation_space.sample()
    return np.asarray(obs, dtype=np.float32), qpos.astype(np.float32), qvel.astype(np.float32)


def _predict_action(model, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32)


def _ordered_components(centers: np.ndarray) -> np.ndarray:
    centered = centers - centers.mean(axis=0, keepdims=True)
    if centered.shape[0] >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vh[0]
        ordered = np.argsort(proj)
    else:
        ordered = np.arange(centers.shape[0])
    return np.asarray(ordered, dtype=np.int64)


def _resolve_split_components(
    centers: np.ndarray,
    *,
    id_components: Optional[Sequence[int]],
    ood_components: Optional[Sequence[int]],
) -> Dict[str, np.ndarray]:
    ordered = _ordered_components(centers)
    n_comp = centers.shape[0]
    if id_components is None or len(id_components) == 0:
        split = max(1, n_comp // 2)
        id_components = ordered[:split].tolist()
    if ood_components is None or len(ood_components) == 0:
        ood_components = [int(i) for i in ordered if int(i) not in set(int(x) for x in id_components)]
    if not ood_components:
        ood_components = [int(ordered[-1])]
    if not id_components:
        id_components = [int(ordered[0])]
    return {
        "ID": np.array(sorted(set(int(i) for i in id_components)), dtype=np.int64),
        "OOD": np.array(sorted(set(int(i) for i in ood_components)), dtype=np.int64),
    }


def _kmeans_centers(X: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError("sklearn is required for clustered_reference state sampling") from e

    n_clusters = max(2, min(int(n_clusters), int(X.shape[0])))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return labels.astype(np.int64), km.cluster_centers_.astype(np.float32)


class ClusteredReferenceSampler:
    def __init__(
        self,
        states: np.ndarray,
        qpos_dim: int,
        *,
        n_components: int = 16,
        seed: int = 0,
        id_components: Optional[Sequence[int]] = None,
        ood_components: Optional[Sequence[int]] = None,
        qpos_noise_scale: float = 0.15,
        qvel_noise_scale: float = 0.25,
        std_floor: float = 0.05,
        clip_margin_scale: float = 0.25,
    ):
        if states.ndim != 2 or states.shape[0] < 4:
            raise ValueError("Need at least 4 reference simulator states to fit a sampler")
        self.states = states.astype(np.float32)
        self.qpos_dim = int(qpos_dim)
        self.seed = int(seed)
        self.qpos_noise_scale = float(qpos_noise_scale)
        self.qvel_noise_scale = float(qvel_noise_scale)

        labels, centers = _kmeans_centers(self.states, n_clusters=n_components, seed=self.seed)
        n_comp = centers.shape[0]
        global_std = self.states.std(axis=0) + 1e-6
        global_min = self.states.min(axis=0)
        global_max = self.states.max(axis=0)
        clip_margin = clip_margin_scale * global_std
        self.clip_low = global_min - clip_margin
        self.clip_high = global_max + clip_margin

        self.centers = centers
        self.stds = np.zeros_like(centers)
        self.weights = np.zeros(n_comp, dtype=np.float32)
        for c in range(n_comp):
            members = self.states[labels == c]
            if members.shape[0] == 0:
                self.stds[c] = global_std
                self.weights[c] = 1.0
            else:
                self.stds[c] = np.maximum(members.std(axis=0), std_floor * global_std)
                self.weights[c] = float(members.shape[0])
        self.weights /= self.weights.sum()

        self.split_components = _resolve_split_components(
            self.centers,
            id_components=id_components,
            ood_components=ood_components,
        )

    def sample(self, split: str, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
        split = str(split).upper()
        if split not in self.split_components:
            raise KeyError(f"Unknown split '{split}'. Choices: {sorted(self.split_components)}")
        comp_ids = self.split_components[split]
        comp_weights = self.weights[comp_ids]
        comp_weights = comp_weights / comp_weights.sum()
        comp = int(rng.choice(comp_ids, p=comp_weights))
        scale = np.ones(self.centers.shape[1], dtype=np.float32)
        scale[:self.qpos_dim] *= self.qpos_noise_scale
        scale[self.qpos_dim:] *= self.qvel_noise_scale
        sample = self.centers[comp] + rng.normal(size=self.centers.shape[1]).astype(np.float32) * self.stds[comp] * scale
        sample = np.clip(sample, self.clip_low, self.clip_high)
        return sample.astype(np.float32), comp

    def summary(self) -> dict:
        return {
            "kind": "clustered_reference",
            "n_components": int(self.centers.shape[0]),
            "split_components": {k: v.tolist() for k, v in self.split_components.items()},
            "qpos_noise_scale": self.qpos_noise_scale,
            "qvel_noise_scale": self.qvel_noise_scale,
        }


class PolicyActionClusteredSampler:
    def __init__(
        self,
        states: np.ndarray,
        *,
        policy_actions: Dict[str, np.ndarray],
        qpos_dim: int,
        n_components: int = 16,
        seed: int = 0,
        qpos_noise_scale: float = 0.0,
        qvel_noise_scale: float = 0.0,
        std_floor: float = 0.05,
        clip_margin_scale: float = 0.25,
        id_components: Optional[Dict[str, Sequence[int]]] = None,
        ood_components: Optional[Dict[str, Sequence[int]]] = None,
    ):
        if states.ndim != 2 or states.shape[0] < 4:
            raise ValueError("Need at least 4 reference simulator states to fit an action-conditioned sampler")
        self.states = states.astype(np.float32)
        self.qpos_dim = int(qpos_dim)
        self.seed = int(seed)
        self.qpos_noise_scale = float(qpos_noise_scale)
        self.qvel_noise_scale = float(qvel_noise_scale)

        global_std = self.states.std(axis=0) + 1e-6
        global_min = self.states.min(axis=0)
        global_max = self.states.max(axis=0)
        clip_margin = clip_margin_scale * global_std
        self.clip_low = global_min - clip_margin
        self.clip_high = global_max + clip_margin

        self.policy_data: Dict[str, dict] = {}
        for offset, (policy_name, actions) in enumerate(sorted(policy_actions.items())):
            labels, centers = _kmeans_centers(actions.astype(np.float32), n_clusters=n_components, seed=self.seed + offset)
            n_comp = centers.shape[0]
            weights = np.zeros(n_comp, dtype=np.float32)
            state_stds = np.zeros((n_comp, self.states.shape[1]), dtype=np.float32)
            members: Dict[int, np.ndarray] = {}
            for c in range(n_comp):
                idx = np.flatnonzero(labels == c)
                members[c] = idx.astype(np.int64)
                if idx.size == 0:
                    state_stds[c] = global_std
                    weights[c] = 1.0
                else:
                    state_stds[c] = np.maximum(self.states[idx].std(axis=0), std_floor * global_std)
                    weights[c] = float(idx.size)
            weights /= weights.sum()
            split_components = _resolve_split_components(
                centers,
                id_components=(id_components or {}).get(policy_name),
                ood_components=(ood_components or {}).get(policy_name),
            )
            self.policy_data[policy_name] = {
                "centers": centers.astype(np.float32),
                "weights": weights.astype(np.float32),
                "state_stds": state_stds.astype(np.float32),
                "split_components": split_components,
                "members": members,
            }

    def sample(self, policy_name: str, split: str, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
        policy_name = str(policy_name)
        split = str(split).upper()
        pdata = self.policy_data[policy_name]
        comp_ids = pdata["split_components"][split]
        comp_weights = pdata["weights"][comp_ids]
        comp_weights = comp_weights / comp_weights.sum()
        comp = int(rng.choice(comp_ids, p=comp_weights))
        member_ids = pdata["members"][comp]
        if member_ids.size == 0:
            raise RuntimeError(f"No reference states for policy={policy_name} component={comp}")
        idx = int(rng.choice(member_ids))
        sample = self.states[idx].copy()
        if self.qpos_noise_scale > 0.0 or self.qvel_noise_scale > 0.0:
            scale = np.ones(self.states.shape[1], dtype=np.float32)
            scale[:self.qpos_dim] *= self.qpos_noise_scale
            scale[self.qpos_dim:] *= self.qvel_noise_scale
            sample = sample + rng.normal(size=self.states.shape[1]).astype(np.float32) * pdata["state_stds"][comp] * scale
            sample = np.clip(sample, self.clip_low, self.clip_high)
        return sample.astype(np.float32), comp

    def summary(self) -> dict:
        return {
            "kind": "policy_action_clustered_reference",
            "qpos_noise_scale": self.qpos_noise_scale,
            "qvel_noise_scale": self.qvel_noise_scale,
            "policies": {
                policy_name: {
                    "n_components": int(pdata["centers"].shape[0]),
                    "split_components": {k: v.tolist() for k, v in pdata["split_components"].items()},
                }
                for policy_name, pdata in self.policy_data.items()
            },
        }


def _fit_state_sampler(reference_states: np.ndarray, qpos_dim: int, state_dist_cfg: Optional[dict]) -> ClusteredReferenceSampler:
    cfg = dict(state_dist_cfg or {})
    kind = str(cfg.pop("kind", "clustered_reference"))
    if kind != "clustered_reference":
        raise KeyError(f"Unsupported custom MuJoCo state sampler '{kind}'")
    return ClusteredReferenceSampler(reference_states, qpos_dim, **cfg)


def _reference_observations_from_states(env_key: str, states: np.ndarray) -> np.ndarray:
    env = _make_env(env_key)
    obs_list: List[np.ndarray] = []
    for i, sim_state in enumerate(states):
        env.reset(seed=10_000 + i)
        obs, _, _ = _set_sim_state(env, sim_state)
        obs_list.append(obs.astype(np.float32))
    env.close()
    return np.stack(obs_list, axis=0).astype(np.float32)


def _collect_reference_actions(
    env_key: str,
    policy_names: Sequence[str],
    observations: np.ndarray,
    *,
    device: str,
    deterministic: bool,
) -> Dict[str, np.ndarray]:
    policy_actions: Dict[str, np.ndarray] = {}
    for policy_name in policy_names:
        model = load_policy_model(env_key, policy_name, device=device)
        acts = [_predict_action(model, obs, deterministic=deterministic) for obs in observations]
        policy_actions[policy_name] = np.stack(acts, axis=0).astype(np.float32)
    return policy_actions


def _knn_density_scores(states: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise RuntimeError("sklearn is required for v2 state-density splitting") from e

    n = int(states.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 reference states for kNN density scores")
    k = max(1, min(int(k), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(states)
    dists, _ = nbrs.kneighbors(states)
    return dists[:, 1:].mean(axis=1).astype(np.float32)


def _mean_pairwise_action_distance(policy_actions: Dict[str, np.ndarray], policy_names: Sequence[str]) -> np.ndarray:
    acts = np.stack([policy_actions[p] for p in policy_names], axis=0).astype(np.float32)
    n_policies = acts.shape[0]
    if n_policies < 2:
        return np.zeros(acts.shape[1], dtype=np.float32)
    pair_terms = []
    for i in range(n_policies):
        for j in range(i + 1, n_policies):
            pair_terms.append(np.linalg.norm(acts[i] - acts[j], axis=1))
    return np.stack(pair_terms, axis=0).mean(axis=0).astype(np.float32)


def _tail_split_indices(scores: np.ndarray, *, tail_fraction: float) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    n = int(scores.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 scores to define v2 ID/OOD tails")
    tail_fraction = float(tail_fraction)
    if not (0.0 < tail_fraction < 0.5):
        raise ValueError(f"tail_fraction must be in (0, 0.5), got {tail_fraction}")
    tail_n = max(1, int(round(n * tail_fraction)))
    tail_n = min(tail_n, (n - 1) // 2 if n > 2 else 1)
    order = np.argsort(scores)
    return {
        "ID": np.asarray(order[:tail_n], dtype=np.int64),
        "OOD": np.asarray(order[-tail_n:], dtype=np.int64),
    }


def _tail_split_indices_reversed(scores: np.ndarray, *, tail_fraction: float) -> Dict[str, np.ndarray]:
    base = _tail_split_indices(scores, tail_fraction=tail_fraction)
    return {
        "ID": base["OOD"].copy(),
        "OOD": base["ID"].copy(),
    }


def _partition_counts(total: int, *, train_frac: float, val_frac: float) -> Dict[str, int]:
    n_train = int(round(train_frac * total))
    n_val = int(round(val_frac * total))
    n_test = int(total - n_train - n_val)
    if total >= 3:
        if n_train <= 0:
            n_train, n_test = 1, max(0, n_test - 1)
        if n_val <= 0:
            n_val, n_test = 1, max(0, n_test - 1)
        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)
    return {"train": int(n_train), "val": int(n_val), "test": int(n_test)}


def _partition_source_indices(
    split_indices: Dict[str, np.ndarray],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, idxs in split_indices.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        split_offset = sum(ord(ch) for ch in str(split_name))
        rng = np.random.default_rng(seed + split_offset)
        order = rng.permutation(idxs)
        counts = _partition_counts(int(order.size), train_frac=train_frac, val_frac=val_frac)
        n_train = counts["train"]
        n_val = counts["val"]
        out[split_name] = {
            "train": order[:n_train].astype(np.int64),
            "val": order[n_train:n_train + n_val].astype(np.int64),
            "test": order[n_train + n_val:].astype(np.int64),
        }
    return out


def _build_candidate_sequences(
    idxs: np.ndarray,
    *,
    n_sequences: int,
    episode_horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idxs = np.asarray(idxs, dtype=np.int64)
    if idxs.size == 0:
        raise RuntimeError("Cannot build candidate sequences from an empty source-state pool")
    return rng.choice(idxs, size=(int(n_sequences), int(episode_horizon)), replace=True).astype(np.int64)


def _sequence_mean_action_disagreement(
    policy_actions: Dict[str, np.ndarray],
    policy_names: Sequence[str],
    candidate_indices: np.ndarray,
) -> np.ndarray:
    means = np.stack(
        [policy_actions[p][candidate_indices].mean(axis=1) for p in policy_names],
        axis=0,
    ).astype(np.float32)
    pair_terms = []
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            pair_terms.append(np.linalg.norm(means[i] - means[j], axis=1))
    if not pair_terms:
        return np.zeros(candidate_indices.shape[0], dtype=np.float32)
    return np.stack(pair_terms, axis=0).mean(axis=0).astype(np.float32)


def _sequence_summary_disagreement(
    policy_actions: Dict[str, np.ndarray],
    policy_names: Sequence[str],
    candidate_indices: np.ndarray,
    *,
    probe_like_samples: int,
    probe_like_repeats: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    horizon = int(candidate_indices.shape[1])
    rng = np.random.default_rng(seed)
    subset_idxs = []
    for _ in range(max(0, probe_like_repeats)):
        k = min(max(1, int(probe_like_samples)), horizon)
        subset_idxs.append(np.sort(rng.choice(horizon, size=k, replace=False)))

    features = []
    for policy_name in policy_names:
        seq_actions = policy_actions[policy_name][candidate_indices]  # (N, H, A)
        parts = [seq_actions.mean(axis=1)]
        for subset in subset_idxs:
            parts.append(seq_actions[:, subset, :].mean(axis=1))
        features.append(np.concatenate(parts, axis=1).astype(np.float32))

    pair_terms = []
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            pair_terms.append(np.linalg.norm(features[i] - features[j], axis=1))
    if not pair_terms:
        z = np.zeros(candidate_indices.shape[0], dtype=np.float32)
        return z, z
    pair_stack = np.stack(pair_terms, axis=0).astype(np.float32)
    return pair_stack.max(axis=0), pair_stack.mean(axis=0)


def _build_shared_sequences(
    states: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    *,
    counts: Dict[str, int],
    episode_horizon: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sequences: Dict[str, np.ndarray] = {}
    source_indices: Dict[str, np.ndarray] = {}
    for split_name, idxs in split_indices.items():
        target = int(counts.get(split_name, 0))
        if target <= 0:
            continue
        if idxs.size == 0:
            raise RuntimeError(f"No source states available for split '{split_name}'")
        choice = rng.choice(idxs, size=(target, episode_horizon), replace=True)
        sequences[split_name] = states[choice].astype(np.float32)
        source_indices[split_name] = choice.astype(np.int64)
    return sequences, source_indices


def _prepare_state_resampled_v2(
    reference_states: np.ndarray,
    *,
    counts: Dict[str, int],
    episode_horizon: int,
    state_dist_cfg: Optional[dict],
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    cfg = dict(state_dist_cfg or {})
    kind = str(cfg.pop("kind", "shared_state_density_v2"))
    if kind != "shared_state_density_v2":
        raise KeyError(f"Unsupported custom MuJoCo v2 state sampler '{kind}'")
    knn_k = int(cfg.pop("knn_k", 32))
    tail_fraction = float(cfg.pop("tail_fraction", 0.4))
    if cfg:
        raise KeyError(f"Unknown v2 state sampler config keys: {sorted(cfg)}")
    scores = _knn_density_scores(reference_states, k=knn_k)
    split_indices = _tail_split_indices(scores, tail_fraction=tail_fraction)
    sequences, source_indices = _build_shared_sequences(
        reference_states,
        split_indices,
        counts=counts,
        episode_horizon=episode_horizon,
        seed=seed,
    )
    summary = {
        "kind": kind,
        "knn_k": knn_k,
        "tail_fraction": tail_fraction,
        "n_reference_states": int(reference_states.shape[0]),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "split_pool_sizes": {k: int(v.size) for k, v in split_indices.items()},
    }
    return sequences, source_indices, summary


def _prepare_action_resampled_v2(
    env_key: str,
    reference_states: np.ndarray,
    policy_names: Sequence[str],
    *,
    counts: Dict[str, int],
    episode_horizon: int,
    action_dist_cfg: Optional[dict],
    device: str,
    deterministic: bool,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    cfg = dict(action_dist_cfg or {})
    kind = str(cfg.pop("kind", "shared_state_action_disagreement_v2"))
    if kind != "shared_state_action_disagreement_v2":
        raise KeyError(f"Unsupported custom MuJoCo v2 action sampler '{kind}'")
    tail_fraction = float(cfg.pop("tail_fraction", 0.4))
    if cfg:
        raise KeyError(f"Unknown v2 action sampler config keys: {sorted(cfg)}")
    reference_obs = _reference_observations_from_states(env_key, reference_states)
    policy_actions = _collect_reference_actions(
        env_key,
        policy_names,
        reference_obs,
        device=device,
        deterministic=deterministic,
    )
    scores = _mean_pairwise_action_distance(policy_actions, policy_names)
    split_indices = _tail_split_indices(scores, tail_fraction=tail_fraction)
    sequences, source_indices = _build_shared_sequences(
        reference_states,
        split_indices,
        counts=counts,
        episode_horizon=episode_horizon,
        seed=seed,
    )
    summary = {
        "kind": kind,
        "tail_fraction": tail_fraction,
        "n_reference_states": int(reference_states.shape[0]),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "split_pool_sizes": {k: int(v.size) for k, v in split_indices.items()},
        "policy_names": list(policy_names),
    }
    return sequences, source_indices, summary


def _prepare_action_resampled_v3(
    env_key: str,
    reference_states: np.ndarray,
    policy_names: Sequence[str],
    *,
    counts: Dict[str, int],
    episode_horizon: int,
    action_dist_cfg: Optional[dict],
    device: str,
    deterministic: bool,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    cfg = dict(action_dist_cfg or {})
    kind = str(cfg.pop("kind", "shared_state_action_mean_matched_v3"))
    if kind != "shared_state_action_mean_matched_v3":
        raise KeyError(f"Unsupported custom MuJoCo v3 action sampler '{kind}'")
    tail_fraction = float(cfg.pop("tail_fraction", 0.4))
    train_frac = float(cfg.pop("train_frac", 0.7))
    val_frac = float(cfg.pop("val_frac", 0.15))
    candidate_multiplier = int(cfg.pop("candidate_multiplier", 12))
    if candidate_multiplier < 1:
        raise ValueError(f"candidate_multiplier must be >= 1, got {candidate_multiplier}")
    if cfg:
        raise KeyError(f"Unknown v3 action sampler config keys: {sorted(cfg)}")

    reference_obs = _reference_observations_from_states(env_key, reference_states)
    policy_actions = _collect_reference_actions(
        env_key,
        policy_names,
        reference_obs,
        device=device,
        deterministic=deterministic,
    )
    state_scores = _mean_pairwise_action_distance(policy_actions, policy_names)
    # Requested semantics for v3:
    #   ID  = high action-disagreement states (easier policy classification)
    #   OOD = low action-disagreement states  (harder policy classification)
    split_indices = _tail_split_indices_reversed(state_scores, tail_fraction=tail_fraction)
    partitioned = _partition_source_indices(
        split_indices,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    sequences: Dict[str, np.ndarray] = {}
    source_indices: Dict[str, np.ndarray] = {}
    partitions: Dict[str, np.ndarray] = {}
    partition_pool_sizes: Dict[str, Dict[str, int]] = {}
    kept_score_summary: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(seed)

    for split_name in ("ID", "OOD"):
        split_sequences = []
        split_source_indices = []
        split_partitions = []
        partition_pool_sizes[split_name] = {}
        kept_score_summary[split_name] = {}
        part_counts = _partition_counts(int(counts.get(split_name, 0)), train_frac=train_frac, val_frac=val_frac)
        for part_name in ("train", "val", "test"):
            target = int(part_counts.get(part_name, 0))
            if target <= 0:
                continue
            pool = partitioned[split_name][part_name]
            partition_pool_sizes[split_name][part_name] = int(pool.size)
            if pool.size == 0:
                raise RuntimeError(
                    f"No source states for split='{split_name}' partition='{part_name}' in action_resampled_v3"
                )
            n_candidates = max(target, target * candidate_multiplier)
            candidate_idx = _build_candidate_sequences(
                pool,
                n_sequences=n_candidates,
                episode_horizon=episode_horizon,
                rng=rng,
            )
            candidate_scores = _sequence_mean_action_disagreement(policy_actions, policy_names, candidate_idx)
            keep = np.argsort(candidate_scores)[:target]
            kept_idx = candidate_idx[keep].astype(np.int64)
            split_sequences.append(reference_states[kept_idx].astype(np.float32))
            split_source_indices.append(kept_idx)
            split_partitions.extend([part_name] * target)
            kept = candidate_scores[keep]
            kept_score_summary[split_name][part_name] = float(kept.mean())
        if split_sequences:
            sequences[split_name] = np.concatenate(split_sequences, axis=0).astype(np.float32)
            source_indices[split_name] = np.concatenate(split_source_indices, axis=0).astype(np.int64)
            partitions[split_name] = np.asarray(split_partitions)

    summary = {
        "kind": kind,
        "tail_fraction": tail_fraction,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "candidate_multiplier": candidate_multiplier,
        "n_reference_states": int(reference_states.shape[0]),
        "score_min": float(state_scores.min()),
        "score_max": float(state_scores.max()),
        "split_pool_sizes": {k: int(v.size) for k, v in split_indices.items()},
        "partition_pool_sizes": partition_pool_sizes,
        "kept_mean_sequence_disagreement": kept_score_summary,
        "policy_names": list(policy_names),
        "id_semantics": "high_action_disagreement",
        "ood_semantics": "low_action_disagreement",
    }
    return sequences, source_indices, partitions, summary


def _prepare_action_resampled_v4(
    env_key: str,
    reference_states: np.ndarray,
    policy_names: Sequence[str],
    *,
    counts: Dict[str, int],
    episode_horizon: int,
    action_dist_cfg: Optional[dict],
    device: str,
    deterministic: bool,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    cfg = dict(action_dist_cfg or {})
    kind = str(cfg.pop("kind", "minari_id_action_matched_ood_v4"))
    if kind != "minari_id_action_matched_ood_v4":
        raise KeyError(f"Unsupported custom MuJoCo v4 action sampler '{kind}'")
    tail_fraction = float(cfg.pop("tail_fraction", 0.4))
    train_frac = float(cfg.pop("train_frac", 0.7))
    val_frac = float(cfg.pop("val_frac", 0.15))
    acceptance_threshold = float(cfg.pop("acceptance_threshold", 2.5))
    candidate_batch_size = int(cfg.pop("candidate_batch_size", 2048))
    max_candidate_batches = int(cfg.pop("max_candidate_batches", 128))
    probe_like_samples = int(cfg.pop("probe_like_samples", 8))
    probe_like_repeats = int(cfg.pop("probe_like_repeats", 4))
    if cfg:
        raise KeyError(f"Unknown v4 action sampler config keys: {sorted(cfg)}")

    reference_obs = _reference_observations_from_states(env_key, reference_states)
    policy_actions = _collect_reference_actions(
        env_key,
        policy_names,
        reference_obs,
        device=device,
        deterministic=deterministic,
    )
    state_scores = _mean_pairwise_action_distance(policy_actions, policy_names)
    # For v4, the synthetic OOD pool is the difficult low-disagreement tail.
    low_disagreement = _tail_split_indices(state_scores, tail_fraction=tail_fraction)["ID"]
    partitioned = _partition_source_indices(
        {"OOD": low_disagreement},
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )["OOD"]

    sequences: Dict[str, np.ndarray] = {}
    source_indices: Dict[str, np.ndarray] = {}
    partitions: Dict[str, np.ndarray] = {}
    part_counts = _partition_counts(int(counts.get("OOD", 0)), train_frac=train_frac, val_frac=val_frac)
    accepted_sequences = []
    accepted_indices = []
    accepted_partitions = []
    part_pool_sizes: Dict[str, int] = {}
    acceptance_stats: Dict[str, dict] = {}

    for part_name in ("train", "val", "test"):
        target = int(part_counts.get(part_name, 0))
        if target <= 0:
            continue
        pool = partitioned[part_name]
        part_pool_sizes[part_name] = int(pool.size)
        if pool.size == 0:
            raise RuntimeError(f"No OOD source states for partition='{part_name}' in action_resampled_v4")

        kept_idx_batches: List[np.ndarray] = []
        kept_score_batches: List[np.ndarray] = []
        n_accepted = 0
        n_seen = 0
        for batch_id in range(max_candidate_batches):
            candidate_idx = _build_candidate_sequences(
                pool,
                n_sequences=candidate_batch_size,
                episode_horizon=episode_horizon,
                rng=np.random.default_rng(seed + 10_000 * (1 + batch_id) + sum(ord(c) for c in part_name)),
            )
            max_dist, mean_dist = _sequence_summary_disagreement(
                policy_actions,
                policy_names,
                candidate_idx,
                probe_like_samples=probe_like_samples,
                probe_like_repeats=probe_like_repeats,
                seed=seed + batch_id,
            )
            mask = max_dist <= acceptance_threshold
            n_seen += int(candidate_idx.shape[0])
            if np.any(mask):
                kept_idx_batches.append(candidate_idx[mask])
                kept_score_batches.append(mean_dist[mask])
                n_accepted += int(mask.sum())
            if n_accepted >= target:
                break
        if n_accepted < target:
            raise RuntimeError(
                f"Only accepted {n_accepted}/{target} OOD sequences for env={env_key} partition={part_name} "
                f"under acceptance_threshold={acceptance_threshold}. Increase max_candidate_batches or relax the threshold."
            )

        kept_idx = np.concatenate(kept_idx_batches, axis=0).astype(np.int64)
        kept_scores = np.concatenate(kept_score_batches, axis=0).astype(np.float32)
        order = np.argsort(kept_scores)[:target]
        chosen_idx = kept_idx[order]
        accepted_sequences.append(reference_states[chosen_idx].astype(np.float32))
        accepted_indices.append(chosen_idx)
        accepted_partitions.extend([part_name] * target)
        acceptance_stats[part_name] = {
            "target": target,
            "seen_candidates": n_seen,
            "accepted_before_truncation": int(kept_idx.shape[0]),
            "retained_mean_score": float(kept_scores[order].mean()),
            "retained_max_score": float(kept_scores[order].max()),
        }

    if accepted_sequences:
        sequences["OOD"] = np.concatenate(accepted_sequences, axis=0).astype(np.float32)
        source_indices["OOD"] = np.concatenate(accepted_indices, axis=0).astype(np.int64)
        partitions["OOD"] = np.asarray(accepted_partitions)

    summary = {
        "kind": kind,
        "tail_fraction": tail_fraction,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "acceptance_threshold": acceptance_threshold,
        "candidate_batch_size": candidate_batch_size,
        "max_candidate_batches": max_candidate_batches,
        "probe_like_samples": probe_like_samples,
        "probe_like_repeats": probe_like_repeats,
        "n_reference_states": int(reference_states.shape[0]),
        "ood_source_pool_size": int(low_disagreement.size),
        "ood_partition_pool_sizes": part_pool_sizes,
        "acceptance_stats": acceptance_stats,
        "policy_names": list(policy_names),
        "id_source": "minari",
        "ood_source": "synthetic_low_action_disagreement",
    }
    return sequences, source_indices, partitions, summary


def _fit_action_sampler(
    env_key: str,
    reference_states: np.ndarray,
    qpos_dim: int,
    policy_names: Sequence[str],
    *,
    action_dist_cfg: Optional[dict],
    device: str,
    deterministic: bool,
) -> PolicyActionClusteredSampler:
    cfg = dict(action_dist_cfg or {})
    kind = str(cfg.pop("kind", "policy_action_clustered_reference"))
    if kind != "policy_action_clustered_reference":
        raise KeyError(f"Unsupported custom MuJoCo action sampler '{kind}'")
    reference_obs = _reference_observations_from_states(env_key, reference_states)
    policy_actions = _collect_reference_actions(
        env_key,
        policy_names,
        reference_obs,
        device=device,
        deterministic=deterministic,
    )
    return PolicyActionClusteredSampler(
        reference_states,
        policy_actions=policy_actions,
        qpos_dim=qpos_dim,
        **cfg,
    )


def _collect_reference_states(
    env_key: str,
    policy_names: Sequence[str],
    *,
    calibration_cfg: Optional[dict],
    device: str,
    deterministic: bool,
    seed: int,
) -> Tuple[np.ndarray, int]:
    cfg = dict(calibration_cfg or {})
    episodes_per_policy = int(cfg.pop("episodes_per_policy", 4))
    horizon = int(cfg.pop("horizon", 256))
    sample_stride = max(1, int(cfg.pop("sample_stride", 4)))
    if cfg:
        raise KeyError(f"Unknown calibration config keys: {sorted(cfg)}")

    env = _make_env(env_key)
    qpos_dim, _ = _qpos_qvel_sizes(env)
    samples: List[np.ndarray] = []
    for policy_offset, policy_name in enumerate(policy_names):
        model = load_policy_model(env_key, policy_name, device=device)
        for ep in range(episodes_per_policy):
            obs, _ = env.reset(seed=seed + 1000 * policy_offset + ep)
            samples.append(_sim_state(env))
            for step in range(horizon):
                action = _predict_action(model, obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                if step % sample_stride == 0:
                    samples.append(_sim_state(env))
                if terminated or truncated:
                    break
    env.close()
    return np.stack(samples, axis=0).astype(np.float32), qpos_dim


def _episode_buffer(
    episode_id: int,
    observations: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[float],
    terminations: List[bool],
    truncations: List[bool],
    *,
    seed: int,
    info: dict,
) -> EpisodeBuffer:
    return EpisodeBuffer(
        id=episode_id,
        seed=seed,
        observations=np.stack(observations, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        rewards=[float(r) for r in rewards],
        terminations=[bool(x) for x in terminations],
        truncations=[bool(x) for x in truncations],
        infos=info,
    )


def _collect_minari_id_episode_buffers(
    env_key: str,
    policy_names: Sequence[str],
    *,
    per_policy_count: int,
    episode_horizon: int,
    min_episode_length: int,
    generation_mode: str,
    seed: int,
    start_episode_id: int,
) -> Tuple[List[EpisodeBuffer], int]:
    import minari

    ds_ids = MINARI_ENVS[env_key]
    if len(ds_ids) != len(MUJOCO_ENVS[env_key].policies):
        raise RuntimeError(f"Minari dataset mapping mismatch for env='{env_key}'")

    policy_to_ds = {
        policy_name: ds_ids[i]
        for i, policy_name in enumerate(MUJOCO_ENVS[env_key].policies)
    }
    rng = np.random.default_rng(seed)
    buffers: List[EpisodeBuffer] = []
    episode_id = int(start_episode_id)

    for pid, policy_name in enumerate(policy_names):
        ds = minari.load_dataset(policy_to_ds[policy_name], download=True)
        candidates = []
        for ep in ds.iterate_episodes():
            T = min(len(ep.actions), len(ep.observations) - 1)
            if T < min_episode_length:
                continue
            T = min(T, int(episode_horizon))
            obs = np.asarray(ep.observations[: T + 1], dtype=np.float32)
            actions = np.asarray(ep.actions[:T], dtype=np.float32)
            rewards = np.asarray(ep.rewards[:T], dtype=np.float32).tolist()
            terminations = np.asarray(ep.terminations[:T], dtype=bool).tolist()
            truncations = np.asarray(ep.truncations[:T], dtype=bool).tolist()
            candidates.append((obs, actions, rewards, terminations, truncations))
        if len(candidates) < per_policy_count:
            raise RuntimeError(
                f"Requested {per_policy_count} Minari ID episodes for env={env_key} policy={policy_name}, "
                f"but only found {len(candidates)} qualifying episodes."
            )
        chosen = rng.choice(len(candidates), size=per_policy_count, replace=False)
        for idx in chosen.tolist():
            obs, actions, rewards, terminations, truncations = candidates[idx]
            info = {
                "policy_id": np.asarray([int(pid)], dtype=np.int64),
                "policy_name": [str(policy_name)],
                "state_split": ["ID"],
                "generation_mode": [str(generation_mode)],
                "source_dataset": [policy_to_ds[policy_name]],
            }
            buffers.append(
                _episode_buffer(
                    episode_id,
                    [np.asarray(x, dtype=np.float32) for x in obs],
                    [np.asarray(x, dtype=np.float32) for x in actions],
                    rewards,
                    terminations,
                    truncations,
                    seed=seed + episode_id,
                    info=info,
                )
            )
            episode_id += 1
    return buffers, episode_id


def _collect_rollout_episode(
    *,
    env,
    model,
    sim_state: np.ndarray,
    seed: int,
    episode_horizon: int,
    deterministic: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], List[bool], np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    obs0, qpos0, qvel0 = _set_sim_state(env, sim_state)
    observations = [obs0]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    obs = obs0
    for _ in range(episode_horizon):
        action = _predict_action(model, obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(action.astype(np.float32))
        rewards.append(float(reward))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        observations.append(np.asarray(next_obs, dtype=np.float32))
        obs = np.asarray(next_obs, dtype=np.float32)
        if terminated or truncated:
            break
    return observations, actions, rewards, terminations, truncations, qpos0, qvel0


def _collect_resampled_step_episode(
    *,
    env,
    model,
    sampler: "ClusteredReferenceSampler",
    split_name: str,
    seed: int,
    episode_horizon: int,
    deterministic: bool,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], List[bool], np.ndarray, np.ndarray, np.ndarray, int]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    components: List[int] = []
    init_qpos = None
    init_qvel = None
    init_state = None

    for step in range(episode_horizon):
        sim_state, component = sampler.sample(split_name, rng)
        env.reset(seed=seed + step)
        obs, qpos, qvel = _set_sim_state(env, sim_state)
        action = _predict_action(model, obs, deterministic=deterministic)
        observations.append(obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(0.0)
        terminations.append(False)
        truncations.append(False)
        components.append(int(component))
        if init_qpos is None:
            init_qpos = qpos
            init_qvel = qvel
            init_state = sim_state.astype(np.float32)

    assert observations
    truncations[-1] = True
    observations.append(observations[-1].copy())
    assert init_qpos is not None and init_qvel is not None and init_state is not None
    return observations, actions, rewards, terminations, truncations, init_qpos, init_qvel, init_state, int(components[0])


def _collect_action_resampled_step_episode(
    *,
    env,
    model,
    sampler: "PolicyActionClusteredSampler",
    policy_name: str,
    split_name: str,
    seed: int,
    episode_horizon: int,
    deterministic: bool,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], List[bool], np.ndarray, np.ndarray, np.ndarray, int]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    components: List[int] = []
    init_qpos = None
    init_qvel = None
    init_state = None

    for step in range(episode_horizon):
        sim_state, component = sampler.sample(policy_name, split_name, rng)
        env.reset(seed=seed + step)
        obs, qpos, qvel = _set_sim_state(env, sim_state)
        action = _predict_action(model, obs, deterministic=deterministic)
        observations.append(obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(0.0)
        terminations.append(False)
        truncations.append(False)
        components.append(int(component))
        if init_qpos is None:
            init_qpos = qpos
            init_qvel = qvel
            init_state = sim_state.astype(np.float32)

    assert observations
    truncations[-1] = True
    observations.append(observations[-1].copy())
    assert init_qpos is not None and init_qvel is not None and init_state is not None
    return observations, actions, rewards, terminations, truncations, init_qpos, init_qvel, init_state, int(components[0])


def _collect_fixed_state_sequence_episode(
    *,
    env,
    model,
    sim_states: np.ndarray,
    seed: int,
    deterministic: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], List[bool], np.ndarray, np.ndarray, np.ndarray]:
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    init_qpos = None
    init_qvel = None
    init_state = None

    for step, sim_state in enumerate(np.asarray(sim_states, dtype=np.float32)):
        env.reset(seed=seed + step)
        obs, qpos, qvel = _set_sim_state(env, sim_state)
        action = _predict_action(model, obs, deterministic=deterministic)
        observations.append(obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(0.0)
        terminations.append(False)
        truncations.append(False)
        if init_qpos is None:
            init_qpos = qpos
            init_qvel = qvel
            init_state = sim_state.astype(np.float32)

    assert observations
    truncations[-1] = True
    observations.append(observations[-1].copy())
    assert init_qpos is not None and init_qvel is not None and init_state is not None
    return observations, actions, rewards, terminations, truncations, init_qpos, init_qvel, init_state


def generate_custom_mujoco_dataset(
    env_key: str,
    *,
    dataset_id: Optional[str] = None,
    policies: Optional[Sequence[str]] = None,
    device: str = "cpu",
    deterministic: bool = True,
    episode_horizon: int = 128,
    min_episode_length: int = 32,
    episodes_per_policy: Optional[dict] = None,
    calibration: Optional[dict] = None,
    state_distribution: Optional[dict] = None,
    action_distribution: Optional[dict] = None,
    generation_mode: str = "rollout_episode",
    force_rebuild: bool = False,
    seed: int = 0,
) -> str:
    if env_key not in MUJOCO_ENVS:
        raise KeyError(f"Unknown custom MuJoCo env '{env_key}'. Choices: {sorted(MUJOCO_ENVS)}")
    env_spec = MUJOCO_ENVS[env_key]
    generation_mode = str(generation_mode)
    if generation_mode not in {"rollout_episode", "resampled_steps", "action_resampled_steps", "state_resampled_v2", "state_resampled_v3", "action_resampled_v2", "action_resampled_v3", "action_resampled_v4", "action_resampled_v5"}:
        raise KeyError(f"Unknown generation_mode '{generation_mode}'")
    if state_distribution is not None and action_distribution is not None:
        raise ValueError("Specify at most one of state_distribution or action_distribution")
    dataset_id = str(dataset_id or _default_dataset_id(env_key, generation_mode))
    policy_names = _policy_order(env_spec, policies)
    counts = {"ID": 48, "OOD": 24}
    if episodes_per_policy is not None:
        for k, v in dict(episodes_per_policy).items():
            counts[str(k).upper()] = int(v)

    if force_rebuild and dataset_id in minari.list_local_datasets():
        minari.delete_dataset(dataset_id)

    ref_states, qpos_dim = _collect_reference_states(
        env_key,
        policy_names,
        calibration_cfg=calibration,
        device=device,
        deterministic=deterministic,
        seed=seed,
    )
    shared_sequences = None
    shared_sequence_indices = None
    shared_sequence_partitions = None
    prebuilt_id_buffers: List[EpisodeBuffer] = []
    if generation_mode == "action_resampled_steps":
        sampler = _fit_action_sampler(
            env_key,
            ref_states,
            qpos_dim,
            policy_names,
            action_dist_cfg=action_distribution,
            device=device,
            deterministic=deterministic,
        )
        sampler_summary = _as_serializable(sampler.summary())
    elif generation_mode in {"state_resampled_v2", "state_resampled_v3"}:
        shared_sequences, shared_sequence_indices, sampler_summary = _prepare_state_resampled_v2(
            ref_states,
            counts=counts,
            episode_horizon=episode_horizon,
            state_dist_cfg=state_distribution,
            seed=seed,
        )
        sampler = None
    elif generation_mode == "action_resampled_v2":
        shared_sequences, shared_sequence_indices, sampler_summary = _prepare_action_resampled_v2(
            env_key,
            ref_states,
            policy_names,
            counts=counts,
            episode_horizon=episode_horizon,
            action_dist_cfg=action_distribution,
            device=device,
            deterministic=deterministic,
            seed=seed,
        )
        sampler = None
    elif generation_mode == "action_resampled_v3":
        shared_sequences, shared_sequence_indices, shared_sequence_partitions, sampler_summary = _prepare_action_resampled_v3(
            env_key,
            ref_states,
            policy_names,
            counts=counts,
            episode_horizon=episode_horizon,
            action_dist_cfg=action_distribution,
            device=device,
            deterministic=deterministic,
            seed=seed,
        )
        sampler = None
    elif generation_mode in {"action_resampled_v4", "action_resampled_v5"}:
        shared_sequences, shared_sequence_indices, shared_sequence_partitions, sampler_summary = _prepare_action_resampled_v4(
            env_key,
            ref_states,
            policy_names,
            counts=counts,
            episode_horizon=episode_horizon,
            action_dist_cfg=action_distribution,
            device=device,
            deterministic=deterministic,
            seed=seed,
        )
        prebuilt_id_buffers, _ = _collect_minari_id_episode_buffers(
            env_key,
            policy_names,
            per_policy_count=int(counts.get("ID", 0)),
            episode_horizon=episode_horizon,
            min_episode_length=min_episode_length,
            generation_mode=generation_mode,
            seed=seed,
            start_episode_id=0,
        )
        sampler = None
    else:
        sampler = _fit_state_sampler(ref_states, qpos_dim=qpos_dim, state_dist_cfg=state_distribution)
        sampler_summary = _as_serializable(sampler.summary())

    env = _make_env(env_key)
    buffers: List[EpisodeBuffer] = list(prebuilt_id_buffers)
    episode_id = len(buffers)
    rng = np.random.default_rng(seed)

    for pid, policy_name in enumerate(policy_names):
        model = load_policy_model(env_key, policy_name, device=device)
        for split_name in ("ID", "OOD"):
            target = int(counts.get(split_name, 0))
            if target <= 0:
                continue
            if generation_mode in {"action_resampled_v4", "action_resampled_v5"} and split_name == "ID":
                continue
            if generation_mode in {"state_resampled_v2", "state_resampled_v3", "action_resampled_v2", "action_resampled_v3", "action_resampled_v4", "action_resampled_v5"}:
                assert shared_sequences is not None and shared_sequence_indices is not None
                split_sequences = shared_sequences[split_name]
                split_indices = shared_sequence_indices[split_name]
                split_partitions = shared_sequence_partitions[split_name] if shared_sequence_partitions is not None else None
                if split_sequences.shape[0] < target:
                    raise RuntimeError(f"Insufficient shared sequences for split '{split_name}'")
                for seq_id in range(target):
                    episode_seed = seed + pid * 10000 + seq_id
                    observations, actions, rewards, terminations, truncations, qpos0, qvel0, init_state = _collect_fixed_state_sequence_episode(
                        env=env,
                        model=model,
                        sim_states=split_sequences[seq_id],
                        seed=episode_seed,
                        deterministic=deterministic,
                    )
                    component = int(split_indices[seq_id, 0])
                    info = {
                        "policy_id": np.asarray([int(pid)], dtype=np.int64),
                        "policy_name": [str(policy_name)],
                        "state_split": [str(split_name)],
                        "sampler_component": np.asarray([int(component)], dtype=np.int64),
                        "init_qpos": qpos0.astype(np.float32),
                        "init_qvel": qvel0.astype(np.float32),
                        "init_state": init_state.astype(np.float32),
                        "generation_mode": [generation_mode],
                        "shared_sequence_id": np.asarray([int(seq_id)], dtype=np.int64),
                    }
                    if split_partitions is not None:
                        info["predefined_partition"] = [str(split_partitions[seq_id])]
                    buffers.append(
                        _episode_buffer(
                            episode_id,
                            observations,
                            actions,
                            rewards,
                            terminations,
                            truncations,
                            seed=seed + episode_id,
                            info=info,
                        )
                    )
                    episode_id += 1
                continue
            accepted = 0
            attempts = 0
            max_attempts = max(20, 10 * target)
            while accepted < target and attempts < max_attempts:
                attempts += 1
                episode_seed = seed + pid * 10000 + accepted * 31 + attempts
                if generation_mode == "rollout_episode":
                    sim_state, component = sampler.sample(split_name, rng)
                    observations, actions, rewards, terminations, truncations, qpos0, qvel0 = _collect_rollout_episode(
                        env=env,
                        model=model,
                        sim_state=sim_state,
                        seed=episode_seed,
                        episode_horizon=episode_horizon,
                        deterministic=deterministic,
                    )
                    init_state = sim_state.astype(np.float32)
                elif generation_mode == "resampled_steps":
                    observations, actions, rewards, terminations, truncations, qpos0, qvel0, init_state, component = _collect_resampled_step_episode(
                        env=env,
                        model=model,
                        sampler=sampler,
                        split_name=split_name,
                        seed=episode_seed,
                        episode_horizon=episode_horizon,
                        deterministic=deterministic,
                        rng=rng,
                    )
                else:
                    observations, actions, rewards, terminations, truncations, qpos0, qvel0, init_state, component = _collect_action_resampled_step_episode(
                        env=env,
                        model=model,
                        sampler=sampler,
                        policy_name=policy_name,
                        split_name=split_name,
                        seed=episode_seed,
                        episode_horizon=episode_horizon,
                        deterministic=deterministic,
                        rng=rng,
                    )

                if len(actions) < min_episode_length:
                    continue

                info = {
                    "policy_id": np.asarray([int(pid)], dtype=np.int64),
                    "policy_name": [str(policy_name)],
                    "state_split": [str(split_name)],
                    "sampler_component": np.asarray([int(component)], dtype=np.int64),
                    "init_qpos": qpos0.astype(np.float32),
                    "init_qvel": qvel0.astype(np.float32),
                    "init_state": init_state.astype(np.float32),
                    "generation_mode": [generation_mode],
                }
                buffers.append(
                    _episode_buffer(
                        episode_id,
                        observations,
                        actions,
                        rewards,
                        terminations,
                        truncations,
                        seed=seed + episode_id,
                        info=info,
                    )
                )
                accepted += 1
                episode_id += 1
            if accepted < target:
                raise RuntimeError(
                    f"Only generated {accepted}/{target} episodes for env={env_key} policy={policy_name} split={split_name}. "
                    f"Try lowering min_episode_length or noise scales."
                )
    env.close()

    dataset = create_dataset_from_buffers(
        dataset_id=dataset_id,
        buffer=buffers,
        env=env_spec.env_id,
        eval_env=env_spec.env_id,
        algorithm_name="sb3-policy-bank",
        author="OpenAI Codex",
        author_email="support@openai.com",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts/tree/main/scripts/mujoco/create_dataset.py",
        description=(
            f"Custom locally generated MuJoCo dataset for {env_spec.env_id} using published "
            f"Farama Minari checkpoints and explicit simulator-state ID/OOD samplers."
        ),
        requirements=["gymnasium[mujoco]", "stable-baselines3", "sb3-contrib", "minari"],
    )
    dataset.storage.update_metadata(
        {
            "inr_custom_mujoco": True,
            "inr_env_key": env_key,
            "policy_names": list(policy_names),
            "policy_repo_ids": [_repo_id(env_spec, p) for p in policy_names],
            "policy_algorithm": env_spec.algo,
            "episode_horizon": int(episode_horizon),
            "min_episode_length": int(min_episode_length),
            "episodes_per_policy": counts,
            "seed": int(seed),
            "generation_mode": generation_mode,
            "state_distribution": _as_serializable(_as_serializable(state_distribution or {})),
            "action_distribution": _as_serializable(_as_serializable(action_distribution or {})),
            "sampling_summary": sampler_summary,
            "calibration": _as_serializable(calibration or {}),
        }
    )
    return dataset_id


def _load_dataset_episodes(dataset_id: str, *, min_length: int, max_length: Optional[int]):
    ds = minari.load_dataset(dataset_id)
    states_all: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    pids_all: List[int] = []
    extras_all: List[dict] = []
    for ep in ds.iterate_episodes():
        T = min(len(ep.actions), len(ep.observations) - 1)
        if T < min_length:
            continue
        s = np.asarray(ep.observations[:T], dtype=np.float32)
        a = np.asarray(ep.actions[:T], dtype=np.float32)
        if max_length is not None and s.shape[0] > max_length:
            s = s[:max_length]
            a = a[:max_length]
        info = ep.infos if isinstance(ep.infos, dict) else {}
        pid = int(_decode_scalar(info.get("policy_id", 0)))
        split_name = str(_decode_scalar(info.get("state_split", "ID"))).upper()
        policy_name = str(_decode_scalar(info.get("policy_name", f"pid_{pid}")))
        sampler_component = int(_decode_scalar(info.get("sampler_component", -1)))
        predefined_partition = str(_decode_scalar(info.get("predefined_partition", ""))).lower()
        extras = {
            "predefined_split": split_name,
            "policy_name": policy_name,
            "sampler_component": sampler_component,
        }
        if predefined_partition:
            extras["predefined_partition"] = predefined_partition
        states_all.append(s)
        actions_all.append(a)
        pids_all.append(pid)
        extras_all.append(extras)
    return states_all, actions_all, pids_all, extras_all


def build_custom_mujoco_store(
    env_key: str,
    *,
    dataset_id: Optional[str] = None,
    max_episodes_per_policy: Optional[int] = None,
    min_length: int = 32,
    max_length: Optional[int] = None,
    use_cache: bool = True,
    generation_cfg: Optional[dict] = None,
) -> EpisodeStore:
    if env_key not in MUJOCO_ENVS:
        raise KeyError(f"Unknown env '{env_key}'. Choices: {sorted(MUJOCO_ENVS)}")

    gen = dict(generation_cfg or {})
    generation_mode = str(gen.get("generation_mode", "rollout_episode"))
    dataset_id = str(dataset_id or _default_dataset_id(env_key, generation_mode))
    cache = _store_cache_path(dataset_id)
    generate_if_missing = bool(gen.pop("generate_if_missing", True))
    force_rebuild = bool(gen.pop("force_rebuild", False))

    if force_rebuild:
        if dataset_id in minari.list_local_datasets():
            minari.delete_dataset(dataset_id)
        if cache.exists():
            cache.unlink()

    if dataset_id not in minari.list_local_datasets():
        if not generate_if_missing:
            raise RuntimeError(f"Custom MuJoCo dataset '{dataset_id}' is not available locally")
        generate_custom_mujoco_dataset(env_key, dataset_id=dataset_id, **gen)
        if cache.exists():
            cache.unlink()

    raw = None
    if use_cache and cache.exists():
        try:
            blob = np.load(cache, allow_pickle=True)
            raw = (
                list(blob["states"]),
                list(blob["actions"]),
                list(blob["pids"]),
                list(blob["extras"]),
            )
        except Exception:
            raw = None
    if raw is None:
        raw = _load_dataset_episodes(dataset_id, min_length=min_length, max_length=max_length)
        if use_cache:
            np.savez_compressed(
                cache,
                states=np.array(raw[0], dtype=object),
                actions=np.array(raw[1], dtype=object),
                pids=np.array(raw[2], dtype=np.int64),
                extras=np.array(raw[3], dtype=object),
            )

    states_all, actions_all, pids_all, extras_all = raw

    selected_idx = list(range(len(states_all)))
    if max_episodes_per_policy is not None:
        selected_idx = []
        by_pid: Dict[int, List[int]] = {}
        for i, pid in enumerate(pids_all):
            by_pid.setdefault(int(pid), []).append(i)
        for pid in sorted(by_pid):
            idxs = by_pid[pid]
            if len(idxs) <= max_episodes_per_policy:
                selected_idx.extend(idxs)
                continue
            split_to_idxs: Dict[str, List[int]] = {}
            for i in idxs:
                split = str(extras_all[i].get("predefined_split", "")).upper()
                split_to_idxs.setdefault(split, []).append(i)
            valid_splits = [s for s in ("ID", "OOD") if split_to_idxs.get(s)]
            if len(valid_splits) < 2:
                selected_idx.extend(idxs[:max_episodes_per_policy])
                continue

            total = sum(len(split_to_idxs[s]) for s in valid_splits)
            alloc = {s: int(round(max_episodes_per_policy * len(split_to_idxs[s]) / total)) for s in valid_splits}
            for s in valid_splits:
                alloc[s] = min(len(split_to_idxs[s]), alloc[s])
            if max_episodes_per_policy >= len(valid_splits):
                for s in valid_splits:
                    alloc[s] = max(1, alloc[s])

            while sum(alloc.values()) > max_episodes_per_policy:
                donor = max(valid_splits, key=lambda s: alloc[s])
                if alloc[donor] <= 1:
                    break
                alloc[donor] -= 1
            while sum(alloc.values()) < max_episodes_per_policy:
                receiver = max(valid_splits, key=lambda s: len(split_to_idxs[s]) - alloc[s])
                if alloc[receiver] >= len(split_to_idxs[receiver]):
                    break
                alloc[receiver] += 1

            for s in valid_splits:
                selected_idx.extend(split_to_idxs[s][:alloc[s]])

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    meta: List[EpisodeMeta] = []
    counts: Dict[int, int] = {}
    eid = 0
    for i in selected_idx:
        s = states_all[i]
        a = actions_all[i]
        pid = int(pids_all[i])
        extras = extras_all[i]
        pid = int(pid)
        states.append(np.asarray(s, dtype=np.float32))
        actions.append(np.asarray(a, dtype=np.float32))
        meta.append(
            EpisodeMeta(
                episode_id=eid,
                policy_id=pid,
                is_ood=False,
                source=f"custom_mujoco/{env_key}",
                extras=dict(extras),
            )
        )
        counts[pid] = counts.get(pid, 0) + 1
        eid += 1

    if not states:
        raise RuntimeError(f"No custom MuJoCo episodes loaded for env '{env_key}' and dataset '{dataset_id}'")
    return EpisodeStore(
        states=states,
        actions=actions,
        meta=meta,
        state_dim=states[0].shape[1],
        action_dim=actions[0].shape[1],
        source=f"custom_mujoco/{env_key}",
    )
