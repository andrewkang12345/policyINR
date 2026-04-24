"""Synthetic offline-RL-style datasets.

`build_synthetic_store(...)` creates an `EpisodeStore` where each
policy is a frozen random function and each episode is a rollout
under that policy, generated purely in feature space (no env).

The primary state/action generator is a **Gaussian Random Field** over
spatial frequencies (smooth, continuous, roughly bounded), but the
generator is swapped via `state_generator` / `action_policy` so future
families (random MLPs, linear maps, hashing, mixtures) slot in without
touching the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.registry import Registry
from .base import EpisodeStore, EpisodeMeta


STATE_GENERATORS = Registry("synth_state_generators")
ACTION_POLICIES = Registry("synth_action_policies")


# ---------------- Gaussian random field helpers --------------------------

def _grf_sample(n_points: int, dim: int, n_modes: int = 8, length_scale: float = 1.0,
                rng: np.random.Generator = None) -> np.ndarray:
    """Sample a smooth GRF-like vector field of shape (n_points, dim).

    We realize it as a sum of sinusoidal basis modes with random
    frequencies and phases, which gives a continuous, bounded, roughly
    Gaussian field over a 1D time-like index. `dim` outputs are
    produced with independent coefficient matrices.
    """
    rng = rng or np.random.default_rng()
    t = np.linspace(0.0, 2.0 * np.pi, n_points, dtype=np.float32)[:, None]  # (T,1)
    freqs = rng.normal(0, 1.0 / length_scale, size=(1, n_modes)).astype(np.float32)
    phases = rng.uniform(0, 2 * np.pi, size=(1, n_modes)).astype(np.float32)
    basis = np.sin(t * freqs + phases)  # (T, n_modes)
    W = rng.normal(0, 1.0 / np.sqrt(n_modes), size=(n_modes, dim)).astype(np.float32)
    return basis @ W  # (T, dim)


@STATE_GENERATORS.register("grf")
def grf_state_generator(T: int, state_dim: int, *, seed: int, n_modes: int = 8,
                         length_scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _grf_sample(T, state_dim, n_modes=n_modes, length_scale=length_scale, rng=rng)


@STATE_GENERATORS.register("random_mlp")
def random_mlp_state_generator(T: int, state_dim: int, *, seed: int,
                                 hidden: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, T, dtype=np.float32)[:, None]
    W1 = rng.normal(0, 1.0, size=(1, hidden)).astype(np.float32)
    b1 = rng.normal(0, 0.1, size=(hidden,)).astype(np.float32)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, state_dim)).astype(np.float32)
    h = np.tanh(t @ W1 + b1)
    return h @ W2


@ACTION_POLICIES.register("grf_policy")
def grf_action_policy(policy_seed: int, state_dim: int, action_dim: int,
                       n_modes: int = 8, length_scale: float = 1.0):
    """Return a callable pi(state)->action implementing a smooth random
    function of state via a sinusoidal random-Fourier-feature map."""
    rng = np.random.default_rng(policy_seed)
    F = rng.normal(0, 1.0 / length_scale, size=(state_dim, n_modes)).astype(np.float32)
    ph = rng.uniform(0, 2 * np.pi, size=(n_modes,)).astype(np.float32)
    W = rng.normal(0, 1.0 / np.sqrt(n_modes), size=(n_modes, action_dim)).astype(np.float32)

    def pi(state: np.ndarray) -> np.ndarray:
        feats = np.sin(state @ F + ph)
        return feats @ W

    return pi


@ACTION_POLICIES.register("linear_policy")
def linear_action_policy(policy_seed: int, state_dim: int, action_dim: int):
    rng = np.random.default_rng(policy_seed)
    W = rng.normal(0, 1.0 / np.sqrt(state_dim), size=(state_dim, action_dim)).astype(np.float32)
    b = rng.normal(0, 0.1, size=(action_dim,)).astype(np.float32)

    def pi(state):
        return state @ W + b

    return pi


# ---------------- main builder -------------------------------------------

def build_synthetic_store(
    *,
    n_policies: int,
    episodes_per_policy: int,
    episode_length: int,
    state_dim: int = 8,
    action_dim: int = 4,
    state_generator: str = "grf",
    action_policy: str = "grf_policy",
    noise_std: float = 0.05,
    seed: int = 0,
    state_gen_kwargs: Optional[dict] = None,
    action_gen_kwargs: Optional[dict] = None,
) -> EpisodeStore:
    """Build a synthetic offline dataset.

    Each policy has a different random action function. Each episode
    has a different random state trajectory produced by the chosen
    state generator. Actions are (policy(state) + small Gaussian noise).
    """
    state_gen_kwargs = state_gen_kwargs or {}
    action_gen_kwargs = action_gen_kwargs or {}
    rng = np.random.default_rng(seed)

    gen_state = STATE_GENERATORS.get(state_generator)
    make_policy = ACTION_POLICIES.get(action_policy)

    policies = []
    for p in range(n_policies):
        policies.append(make_policy(
            policy_seed=int(rng.integers(0, 2**31 - 1)),
            state_dim=state_dim,
            action_dim=action_dim,
            **action_gen_kwargs,
        ))

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    meta: List[EpisodeMeta] = []
    eid = 0
    for pid in range(n_policies):
        pi = policies[pid]
        for _ in range(episodes_per_policy):
            s = gen_state(T=episode_length, state_dim=state_dim,
                          seed=int(rng.integers(0, 2**31 - 1)), **state_gen_kwargs)
            a_clean = pi(s)
            a = a_clean + noise_std * rng.normal(size=a_clean.shape).astype(np.float32)
            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))
            meta.append(EpisodeMeta(
                episode_id=eid, policy_id=pid, is_ood=False, source="synthetic",
            ))
            eid += 1

    return EpisodeStore(
        states=states, actions=actions, meta=meta,
        state_dim=state_dim, action_dim=action_dim, source="synthetic",
    )
