"""Minari MuJoCo loader.

For each env we use the three available policies in Minari (simple /
medium / expert versions) as our 3 policy identities. Policy labels
are used ONLY for evaluation (linear probe); they never enter the
training loss.

The loader caches raw trajectories as a numpy `.npz` bundle under
`~/.cache/INR/minari/<env>.npz` so repeated runs avoid reparsing the
Minari HDF5 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os
import numpy as np

from .base import EpisodeStore, EpisodeMeta


MINARI_ENVS = {
    "hopper":       ["mujoco/hopper/simple-v0",       "mujoco/hopper/medium-v0",       "mujoco/hopper/expert-v0"],
    "humanoid":     ["mujoco/humanoid/simple-v0",     "mujoco/humanoid/medium-v0",     "mujoco/humanoid/expert-v0"],
    "halfcheetah":  ["mujoco/halfcheetah/simple-v0",  "mujoco/halfcheetah/medium-v0",  "mujoco/halfcheetah/expert-v0"],
    "walker2d":     ["mujoco/walker2d/simple-v0",     "mujoco/walker2d/medium-v0",     "mujoco/walker2d/expert-v0"],
    "ant":          ["mujoco/ant/simple-v0",          "mujoco/ant/medium-v0",          "mujoco/ant/expert-v0"],
}


def _cache_path(env_key: str) -> Path:
    root = Path(os.environ.get("INR_MINARI_CACHE", Path.home() / ".cache" / "INR" / "minari"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{env_key}.npz"


def _fetch_minari(env_key: str, max_episodes_per_policy: Optional[int], min_length: int):
    import minari

    ds_ids = MINARI_ENVS[env_key]
    states_all: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    pids_all: List[int] = []
    for pid, ds_id in enumerate(ds_ids):
        try:
            ds = minari.load_dataset(ds_id, download=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Minari dataset '{ds_id}': {e}") from e
        taken = 0
        for ep in ds.iterate_episodes():
            T = min(len(ep.actions), len(ep.observations) - 1)
            if T < min_length:
                continue
            s = np.asarray(ep.observations[:T], dtype=np.float32)
            a = np.asarray(ep.actions[:T], dtype=np.float32)
            states_all.append(s)
            actions_all.append(a)
            pids_all.append(pid)
            taken += 1
            if max_episodes_per_policy is not None and taken >= max_episodes_per_policy:
                break
    return states_all, actions_all, pids_all


def build_minari_store(
    env_key: str,
    *,
    max_episodes_per_policy: Optional[int] = None,
    min_length: int = 32,
    max_length: Optional[int] = None,
    use_cache: bool = True,
) -> EpisodeStore:
    if env_key not in MINARI_ENVS:
        raise KeyError(f"Unknown env '{env_key}'. Choices: {list(MINARI_ENVS)}")

    cache = _cache_path(env_key)
    raw = None
    if use_cache and cache.exists():
        try:
            blob = np.load(cache, allow_pickle=True)
            raw = (list(blob["states"]), list(blob["actions"]), list(blob["pids"]))
        except Exception:
            raw = None
    if raw is None:
        raw = _fetch_minari(env_key, max_episodes_per_policy=None, min_length=min_length)
        if use_cache:
            try:
                np.savez_compressed(
                    cache,
                    states=np.array(raw[0], dtype=object),
                    actions=np.array(raw[1], dtype=object),
                    pids=np.array(raw[2], dtype=np.int64),
                )
            except Exception:
                pass

    states_all, actions_all, pids_all = raw

    # apply per-policy episode cap and optional truncation
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    meta: List[EpisodeMeta] = []
    counts = {}
    eid = 0
    for s, a, pid in zip(states_all, actions_all, pids_all):
        if max_episodes_per_policy is not None and counts.get(int(pid), 0) >= max_episodes_per_policy:
            continue
        if s.shape[0] < min_length:
            continue
        if max_length is not None and s.shape[0] > max_length:
            s = s[:max_length]
            a = a[:max_length]
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        states.append(s)
        actions.append(a)
        meta.append(EpisodeMeta(
            episode_id=eid, policy_id=int(pid), is_ood=False,
            source=f"mujoco/{env_key}",
        ))
        counts[int(pid)] = counts.get(int(pid), 0) + 1
        eid += 1

    if not states:
        raise RuntimeError(f"No episodes loaded for env '{env_key}'")
    state_dim = states[0].shape[1]
    action_dim = actions[0].shape[1]
    return EpisodeStore(
        states=states, actions=actions, meta=meta,
        state_dim=state_dim, action_dim=action_dim,
        source=f"mujoco/{env_key}",
    )
