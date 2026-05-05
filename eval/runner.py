"""Evaluation orchestrator used by train/main.py.

Given a trained `RepresentationModel` and the experiment's loaders,
(1) extract a single representation per test episode, averaged over a
few samples, (2) run the linear probe using policy_id labels, and
(3) compute generative metrics on the test loader.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.base import EpisodeStore, PolicyDataset
from .linear_probe import cosine_knn_probe, linear_probe
from .generative import generative_metrics


@torch.no_grad()
def _per_episode_representations(
    model, store: EpisodeStore, *, history_k: int, per_episode_samples: int,
    state_mean, state_std, action_mean, action_std, shuffle_history: bool,
    device: str, batch_size: int = 256, min_len: int = 4,
    behavior_unit: str = "episode", unit_window_size: int = 0,
    known_unit_map=None,
):
    """Return (embeddings [N, D], labels [N], is_ood [N])."""
    embs: List[torch.Tensor] = []
    pids: List[int] = []
    oods: List[int] = []
    # build a dataset; take K random per-episode picks and average
    # use shuffle_history=True at eval so the bag view works for both model families
    # (deterministic per-index seed).
    ds = PolicyDataset(
        store, history_k=history_k, shuffle_history=shuffle_history,
        behavior_unit=behavior_unit, unit_window_size=unit_window_size,
        known_unit_map=known_unit_map,
        state_mean=state_mean, state_std=state_std,
        action_mean=action_mean, action_std=action_std, seed=0,
    )
    # index by (episode_id, sample_i) -> pick a random step per episode
    step_by_ep: Dict[int, List[int]] = {}
    for flat_i, (ei, t) in enumerate(ds.index):
        step_by_ep.setdefault(ei, []).append(flat_i)

    eps_order = list(step_by_ep.keys())
    flat_picks: List[int] = []
    ep_marker: List[int] = []  # which episode each flat pick came from
    rng = np.random.default_rng(0)
    for ei in eps_order:
        pool = step_by_ep[ei]
        if len(pool) < min_len:
            continue
        k = min(per_episode_samples, len(pool))
        picks = rng.choice(pool, size=k, replace=False)
        flat_picks.extend(picks.tolist())
        ep_marker.extend([ei] * k)

    if not flat_picks:
        return np.zeros((0, model.latent_dim)), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    # iterate in mini-batches over flat_picks
    per_ep_sum: Dict[int, torch.Tensor] = {}
    per_ep_cnt: Dict[int, int] = {}
    bs = batch_size
    model.eval()
    for start in range(0, len(flat_picks), bs):
        chunk = flat_picks[start:start + bs]
        chunk_eids = ep_marker[start:start + bs]
        batch = [ds[i] for i in chunk]
        coll = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        coll = {k: v.to(device, non_blocking=True) for k, v in coll.items()}
        z = model.extract_representation(coll)  # (B, D)
        for zi, ei in zip(z.cpu(), chunk_eids):
            per_ep_sum[ei] = per_ep_sum.get(ei, torch.zeros_like(zi)) + zi
            per_ep_cnt[ei] = per_ep_cnt.get(ei, 0) + 1

    eids = sorted(per_ep_sum.keys())
    E = torch.stack([per_ep_sum[e] / max(1, per_ep_cnt[e]) for e in eids]).numpy()
    pid_arr = np.array([store.meta[e].policy_id for e in eids], dtype=np.int64)
    ood_arr = np.array([int(store.meta[e].is_ood) for e in eids], dtype=np.int64)
    return E, pid_arr, ood_arr


def run_full_eval(model, loaders, eval_cfg, device: str) -> Dict:
    test_store: EpisodeStore = loaders["test_store"]
    out: Dict = {}
    if test_store is None or len(test_store) == 0:
        return {"probe_acc": float("nan"), "gen_mse": float("nan"), "note": "empty test set"}

    # representation extraction (bag-of-pairs view, which works for both CVAE & INR)
    train_embs, train_labels, _ = _per_episode_representations(
        model, loaders["train_store"],
        history_k=int(loaders.get("history_k", 16)),
        per_episode_samples=int(eval_cfg.per_episode_samples),
        state_mean=loaders["state_mean"], state_std=loaders["state_std"],
        action_mean=loaders["action_mean"], action_std=loaders["action_std"],
        shuffle_history=True, device=device,
        min_len=int(eval_cfg.min_episode_length_for_eval),
        behavior_unit=str(loaders.get("behavior_unit", "episode")),
        unit_window_size=int(loaders.get("unit_window_size", 0)),
        known_unit_map=loaders.get("train_unit_map") if loaders.get("use_unit_latents", False) else None,
    )
    test_embs, test_labels, oods = _per_episode_representations(
        model, test_store,
        history_k=int(loaders.get("history_k", 16)),
        per_episode_samples=int(eval_cfg.per_episode_samples),
        state_mean=loaders["state_mean"], state_std=loaders["state_std"],
        action_mean=loaders["action_mean"], action_std=loaders["action_std"],
        shuffle_history=True, device=device,
        min_len=int(eval_cfg.min_episode_length_for_eval),
        behavior_unit=str(loaders.get("behavior_unit", "episode")),
        unit_window_size=int(loaders.get("unit_window_size", 0)),
        known_unit_map=loaders.get("train_unit_map") if loaders.get("use_unit_latents", False) else None,
    )

    # seen policies == those present in train loader
    train_pids = sorted({int(m.policy_id) for m in loaders["train_store"].meta})
    probe = linear_probe(
        train_embs, train_labels,
        test_embs, test_labels,
        train_seen_pids=train_pids,
        C=float(eval_cfg.probe_C),
        max_iter=int(eval_cfg.probe_max_iter),
    )
    probe.update(cosine_knn_probe(train_embs, train_labels, test_embs, test_labels, k=5))
    out.update(probe)

    # generative metrics on the test loader
    gen = generative_metrics(
        model, loaders["test_loader"], device=device,
        max_batches=eval_cfg.get("gen_n_batches"),
    )
    out.update(gen)

    # a few bookkeeping fields
    out["train_pids"] = [int(p) for p in train_pids]
    out["test_pids"] = [int(p) for p in sorted(set(test_labels.tolist()))]
    out["n_train_episodes_used"] = int(train_embs.shape[0])
    out["n_test_episodes_used"] = int(test_embs.shape[0])
    out["n_test_ood"] = int(oods.sum())
    out["probe_protocol"] = "train_probe_on_train_embeddings_eval_on_test_embeddings"
    return out
