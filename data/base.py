"""Unified offline-episode data interface.

Every dataset — synthetic or real — materializes an `EpisodeStore`:
a list of fixed-length trajectories with per-episode metadata. A
`PolicyDataset` wraps a store and yields training examples of the
form used by both CVAE and INR models:

    past_history_states   : (K, state_dim)     bag of past (s) values
    past_history_actions  : (K, action_dim)    bag of past (a) values
    current_state         : (state_dim,)
    next_action           : (action_dim,)      action taken at current step
    episode_id            : int
    unit_id               : int                fitted-latent behavior unit id
    has_unit_latent       : bool               whether this batch item has a learned unit code
    policy_id             : int                for evaluation only
    is_ood                : bool               in state-shift OOD partition

The key training property (per the spec):
  - CVAE: `shuffle_history=True` -> K past pairs are sampled uniformly
    from the episode (bag-of-pairs view). One episode can produce as
    many training examples as it has (s, a) pairs, because we iterate
    over every step as a possible "current" step.
  - INR: `shuffle_history=False` -> past history uses the preceding
    window up to K steps (or wraps from the start if t<K). Ordering is
    preserved so positional info is meaningful; randomness comes from
    standard batch shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeMeta:
    episode_id: int
    policy_id: int            # 0..P-1 within the dataset; used for eval only
    is_ood: bool = False      # state-distribution partition
    source: str = ""          # e.g., "synthetic" or "mujoco/hopper"
    extras: dict = field(default_factory=dict)


class EpisodeStore:
    """Holds a dataset's full set of episodes.

    states:  list of np.ndarray (T_i, state_dim)
    actions: list of np.ndarray (T_i, action_dim)
    meta:    list of EpisodeMeta (len = #episodes)
    """

    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        meta: List[EpisodeMeta],
        state_dim: int,
        action_dim: int,
        source: str = "",
        action_kind: str = "continuous",   # "continuous" or "discrete"
        n_actions: Optional[int] = None,   # required if action_kind == "discrete"
    ):
        assert len(states) == len(actions) == len(meta)
        assert action_kind in ("continuous", "discrete")
        if action_kind == "discrete":
            assert n_actions is not None and n_actions > 0, \
                "n_actions must be set for discrete action spaces"
        self.states = states
        self.actions = actions
        self.meta = meta
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.source = source
        self.action_kind = action_kind
        self.n_actions = int(n_actions) if n_actions is not None else None

    # ---- subset helpers -------------------------------------------------
    def subset(self, indices: List[int]) -> "EpisodeStore":
        indices = [int(i) for i in indices]
        return EpisodeStore(
            states=[self.states[i] for i in indices],
            actions=[self.actions[i] for i in indices],
            meta=[self.meta[i] for i in indices],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            source=self.source,
            action_kind=self.action_kind,
            n_actions=self.n_actions,
        )

    def filter(self, predicate) -> "EpisodeStore":
        idx = [i for i, m in enumerate(self.meta) if predicate(m)]
        return self.subset(idx)

    def __len__(self) -> int:
        return len(self.meta)

    # ---- normalization --------------------------------------------------
    def state_stats(self):
        all_s = np.concatenate(self.states, axis=0) if self.states else np.zeros((1, self.state_dim))
        return all_s.mean(axis=0), all_s.std(axis=0) + 1e-6

    def action_stats(self):
        # For discrete action spaces, normalization doesn't apply — return
        # zeros / ones so PolicyDataset is a no-op on actions. The models
        # consume discrete actions as int indices, not normalized floats.
        if self.action_kind == "discrete":
            return (np.zeros(self.action_dim, dtype=np.float32),
                    np.ones(self.action_dim, dtype=np.float32))
        all_a = np.concatenate(self.actions, axis=0) if self.actions else np.zeros((1, self.action_dim))
        return all_a.mean(axis=0), all_a.std(axis=0) + 1e-6


class PolicyDataset(Dataset):
    """Flat index of (episode, step) pairs for training and evaluation."""

    def __init__(
        self,
        store: EpisodeStore,
        history_k: int = 16,
        shuffle_history: bool = False,
        behavior_unit: str = "episode",
        unit_window_size: int = 0,
        known_unit_map: Optional[Dict[Tuple[int, int], int]] = None,
        register_units: bool = False,
        min_t: int = 1,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        seed: int = 0,
    ):
        self.store = store
        self.history_k = history_k
        self.shuffle_history = shuffle_history
        self.behavior_unit = str(behavior_unit)
        self.unit_window_size = int(unit_window_size) if unit_window_size else int(history_k)
        self.min_t = min_t
        self._rng = np.random.default_rng(seed)
        if self.behavior_unit not in {"episode", "window"}:
            raise ValueError(f"Unknown behavior_unit='{self.behavior_unit}'")
        if self.behavior_unit == "window" and self.unit_window_size <= 0:
            raise ValueError("unit_window_size must be positive for behavior_unit='window'")

        sm = state_mean if state_mean is not None else np.zeros(store.state_dim, dtype=np.float32)
        ss = state_std if state_std is not None else np.ones(store.state_dim, dtype=np.float32)
        am = action_mean if action_mean is not None else np.zeros(store.action_dim, dtype=np.float32)
        asd = action_std if action_std is not None else np.ones(store.action_dim, dtype=np.float32)
        self.state_mean = sm.astype(np.float32)
        self.state_std = ss.astype(np.float32)
        self.action_mean = am.astype(np.float32)
        self.action_std = asd.astype(np.float32)

        # Build flat index of (episode, step) pairs. For OOD episodes,
        # restrict `t` to steps flagged as "shared" (state region shared
        # across policies) so both current_state and the past-history bag
        # we sample stay in the shared region. For ID episodes, use all
        # steps as before.
        self.index: List[tuple] = []
        self._unit_key_by_index: List[Tuple[int, int]] = []
        for ei, states in enumerate(store.states):
            T = states.shape[0]
            mask = store.meta[ei].extras.get("shared_mask") if store.meta[ei].is_ood else None
            if mask is not None:
                for t in range(min_t, T):
                    if bool(mask[t]):
                        self.index.append((ei, t))
                        self._unit_key_by_index.append(self._unit_key(ei, t))
            else:
                for t in range(min_t, T):
                    self.index.append((ei, t))
                    self._unit_key_by_index.append(self._unit_key(ei, t))

        self.unit_map: Optional[Dict[Tuple[int, int], int]] = None
        if register_units:
            self.unit_map = {}
            for key in self._unit_key_by_index:
                if key not in self.unit_map:
                    self.unit_map[key] = len(self.unit_map)
        elif known_unit_map is not None:
            self.unit_map = dict(known_unit_map)
        self.n_units = len(self.unit_map) if self.unit_map is not None else 0

    @property
    def state_dim(self) -> int:
        return self.store.state_dim

    @property
    def action_dim(self) -> int:
        return self.store.action_dim

    def __len__(self) -> int:
        return len(self.index)

    def _norm_s(self, s):
        return (s - self.state_mean) / self.state_std

    def _norm_a(self, a):
        return (a - self.action_mean) / self.action_std

    def _pick_history(self, ep_idx: int, t: int):
        S = self.store.states[ep_idx]
        A = self.store.actions[ep_idx]
        K = self.history_k
        T = S.shape[0]
        meta = self.store.meta[ep_idx]
        if self.behavior_unit == "window":
            _, window_idx = self._unit_key(ep_idx, t)
            win_start = window_idx * self.unit_window_size
            win_end = min(T, (window_idx + 1) * self.unit_window_size)
            unit_steps = np.arange(win_start, win_end, dtype=np.int64)
        else:
            unit_steps = np.arange(T, dtype=np.int64)
        # For OOD episodes, the past-history (both shuffled bag and ordered
        # window) must stay inside the shared state-region mask.
        shared_steps = None
        if meta.is_ood:
            mask = meta.extras.get("shared_mask")
            if mask is not None and mask.any():
                shared_steps = unit_steps[mask[unit_steps]]
        candidate_steps = shared_steps if (shared_steps is not None and shared_steps.size > 0) else unit_steps
        if self.shuffle_history:
            idx = self._rng.choice(candidate_steps, size=K, replace=True)
        else:
            # ordered past window of length K. If OOD, filter to shared steps
            # before `t` and left-pad if fewer than K.
            past_idx = candidate_steps[candidate_steps < t]
            if past_idx.size >= K:
                idx = past_idx[-K:]
            elif past_idx.size > 0:
                pad = np.full(K - past_idx.size, past_idx[0], dtype=np.int64)
                idx = np.concatenate([pad, past_idx], axis=0)
            else:
                fallback = candidate_steps[0] if candidate_steps.size else t
                idx = np.full(K, fallback, dtype=np.int64)
        return S[idx], A[idx]

    def _unit_key(self, ep_idx: int, t: int) -> Tuple[int, int]:
        meta = self.store.meta[ep_idx]
        base_episode_id = int(meta.extras.get("base_episode_id", meta.episode_id))
        if self.behavior_unit == "episode":
            return (base_episode_id, 0)
        return (base_episode_id, int(t // self.unit_window_size))

    def __getitem__(self, i: int):
        ei, t = self.index[i]
        S = self.store.states[ei]
        A = self.store.actions[ei]
        meta = self.store.meta[ei]

        unit_key = self._unit_key_by_index[i]
        if self.unit_map is None:
            unit_id = -1
            has_unit_latent = 0
        else:
            unit_id = int(self.unit_map.get(unit_key, -1))
            has_unit_latent = int(unit_id >= 0)

        past_s, past_a = self._pick_history(ei, t)
        cur_s = S[t]
        next_a = A[t]

        # State is always continuous/featurized → normalized float.
        # Actions: continuous → normalized float; discrete → integer index.
        # Storage convention: discrete actions still stored as (T, 1) arrays
        # of integer ids; we squeeze the trailing 1 here.
        if self.store.action_kind == "discrete":
            past_a_t = torch.as_tensor(past_a.reshape(-1).astype(np.int64), dtype=torch.long)
            next_a_t = torch.as_tensor(int(next_a.reshape(-1)[0]), dtype=torch.long)
        else:
            past_a_t = torch.as_tensor(self._norm_a(past_a), dtype=torch.float32)
            next_a_t = torch.as_tensor(self._norm_a(next_a), dtype=torch.float32)

        return {
            "past_states": torch.as_tensor(self._norm_s(past_s), dtype=torch.float32),
            "past_actions": past_a_t,
            "current_state": torch.as_tensor(self._norm_s(cur_s), dtype=torch.float32),
            "next_action": next_a_t,
            "episode_id": torch.as_tensor(meta.episode_id, dtype=torch.long),
            "unit_id": torch.as_tensor(unit_id, dtype=torch.long),
            "has_unit_latent": torch.as_tensor(has_unit_latent, dtype=torch.long),
            "policy_id": torch.as_tensor(meta.policy_id, dtype=torch.long),
            "is_ood": torch.as_tensor(int(meta.is_ood), dtype=torch.long),
        }


class MaterializedPolicyDataset(Dataset):
    """In-memory tensor cache for small continuous low-dimensional datasets.

    This removes the Python/NumPy per-sample history construction cost from
    every training epoch. For shuffled-history datasets, the sampled history is
    fixed at materialization time instead of being re-sampled each epoch.
    """

    def __init__(self, dataset: PolicyDataset):
        self.dataset = dataset
        columns = {}
        for i in range(len(dataset)):
            item = dataset[i]
            if not columns:
                columns = {k: [] for k in item}
            for k, v in item.items():
                columns[k].append(v)
        self.tensors = {k: torch.stack(v, dim=0) for k, v in columns.items()}

        # Preserve attributes used by evaluation/probe code.
        self.store = dataset.store
        self.history_k = dataset.history_k
        self.shuffle_history = dataset.shuffle_history
        self.behavior_unit = dataset.behavior_unit
        self.unit_window_size = dataset.unit_window_size
        self.min_t = dataset.min_t
        self.index = dataset.index
        self._unit_key_by_index = dataset._unit_key_by_index
        self.unit_map = dataset.unit_map
        self.n_units = dataset.n_units
        self.state_mean = dataset.state_mean
        self.state_std = dataset.state_std
        self.action_mean = dataset.action_mean
        self.action_std = dataset.action_std

    @property
    def state_dim(self) -> int:
        return self.dataset.state_dim

    @property
    def action_dim(self) -> int:
        return self.dataset.action_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int):
        return {k: v[i] for k, v in self.tensors.items()}
