"""Experiment composition: turn a base `EpisodeStore` + a compositional
spec (per-policy train/test shift placement) into concrete train/val/test
loaders.

An experiment YAML config has a list of `policies`, each an item:

    policies:
      - pid: 0
        train: ID         # ID | OOD | NONE     (NONE = not in training)
        test:  ID          # ID | OOD
      - pid: 1
        train: ID
        test:  ID
      - pid: 2
        train: NONE
        test:  ID

Our pipeline:
  1. partition each policy's episodes into ID / OOD via shift assignment
  2. for each policy, assign the ID subset and OOD subset to train/val/test
     buckets according to the spec
  3. construct `PolicyDataset` + `DataLoader` for each bucket
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .base import EpisodeStore, PolicyDataset, EpisodeMeta
from .shifts import assign_state_shift


@dataclass
class PolicySpec:
    pid: int
    train: str   # "ID", "OOD", or "NONE"
    test: str    # "ID" or "OOD"


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_idx = order[:n_train].tolist()
    val_idx = order[n_train:n_train + n_val].tolist()
    test_idx = order[n_train + n_val:].tolist()
    # guarantee each non-empty split has >=1 item when n>=3
    if n >= 3:
        if not train_idx:
            train_idx = [val_idx.pop() if val_idx else test_idx.pop()]
        if not val_idx:
            val_idx = [test_idx.pop() if test_idx else train_idx.pop()]
        if not test_idx:
            test_idx = [val_idx.pop() if val_idx else train_idx.pop()]
    return train_idx, val_idx, test_idx


def _predefined_partition_indices(store: EpisodeStore, idxs: Sequence[int]):
    buckets = {"train": [], "val": [], "test": []}
    saw_any = False
    for i in idxs:
        part = str(store.meta[i].extras.get("predefined_partition", "")).lower()
        if not part:
            return None
        saw_any = True
        if part not in buckets:
            raise RuntimeError(
                f"Unknown predefined_partition='{part}' on episode_id={store.meta[i].episode_id}. "
                "Expected one of {'train', 'val', 'test'}."
            )
        buckets[part].append(i)
    return buckets if saw_any else None


def _concat_stores(stores: Sequence[EpisodeStore]) -> Optional[EpisodeStore]:
    stores = [s for s in stores if len(s) > 0]
    if not stores:
        return None
    states, actions, meta = [], [], []
    eid = 0
    for s in stores:
        for i in range(len(s)):
            states.append(s.states[i])
            actions.append(s.actions[i])
            m = s.meta[i]
            meta.append(EpisodeMeta(
                episode_id=eid, policy_id=m.policy_id, is_ood=m.is_ood,
                source=m.source,
                extras={"base_episode_id": int(m.episode_id), **dict(m.extras)},
            ))
            eid += 1
    return EpisodeStore(
        states=states, actions=actions, meta=meta,
        state_dim=stores[0].state_dim, action_dim=stores[0].action_dim,
        source="+".join(sorted({s.source for s in stores})),
        action_kind=stores[0].action_kind, n_actions=stores[0].n_actions,
    )


def build_experiment_loaders(
    base_store: EpisodeStore,
    *,
    policies: List[PolicySpec],
    history_k: int,
    shift_kind: str = "mean_cluster",
    shift_kwargs: Optional[dict] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    batch_size: int = 128,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    shuffle_history_train: bool = False,
    behavior_unit: str = "episode",
    unit_window_size: int = 0,
    use_unit_latents: bool = False,
    seed: int = 0,
):
    """Compose experiment splits. Returns a dict with:
        train_store, val_store, test_store          (EpisodeStore)
        train_loader, val_loader, test_loader       (DataLoader)
        state_mean, state_std, action_mean, action_std
        n_train_policies (int)

    `shuffle_history_train` flips per-model: CVAE=True, INR=False.
    """
    shift_kwargs = shift_kwargs or {}
    eval_batch_size = eval_batch_size or batch_size

    # 1) state-shift assignment over the whole store
    shifted, shift_diag = assign_state_shift(base_store, kind=shift_kind, **shift_kwargs)
    shift_strength = shift_diag.get("shift_strength", {})
    shift_overlap = shift_diag.get("shift_overlap", float("nan"))
    shift_overlap_ratio = shift_diag.get("shift_overlap_ratio", float("nan"))
    effective_shared = shift_diag.get("effective_shared", float("nan"))
    shift_fallback = shift_diag.get("shift_fallback", "none")

    # 2) per-policy, per-partition split into train/val/test indices
    train_stores: List[EpisodeStore] = []
    val_stores: List[EpisodeStore] = []
    test_stores: List[EpisodeStore] = []

    pid_policy = {p.pid: p for p in policies}
    train_pids = sorted({p.pid for p in policies if p.train != "NONE"})
    n_train_policies = len(train_pids)
    degenerate_buckets: List[str] = []

    for pid, spec in pid_policy.items():
        pol_idx = [i for i, m in enumerate(shifted.meta) if m.policy_id == pid]
        if not pol_idx:
            raise RuntimeError(
                f"Experiment composition requires policy_id={pid} but the base store has no episodes for it."
            )
        id_idx = [i for i in pol_idx if not shifted.meta[i].is_ood]
        ood_idx = [i for i in pol_idx if shifted.meta[i].is_ood]
        # guard: the partition this policy needs must be non-empty
        if spec.train in ("ID", "OOD"):
            needed = id_idx if spec.train == "ID" else ood_idx
            if not needed:
                degenerate_buckets.append(f"pid={pid} train={spec.train} partition empty")
        if spec.test in ("ID", "OOD"):
            needed = id_idx if spec.test == "ID" else ood_idx
            if not needed:
                degenerate_buckets.append(f"pid={pid} test={spec.test} partition empty")
        id_predef = _predefined_partition_indices(shifted, id_idx)
        ood_predef = _predefined_partition_indices(shifted, ood_idx)

        def take(src, subset):
            return [src[i] for i in subset]

        def partition_triplet(src, predef, seed_offset):
            if predef is not None:
                return predef["train"], predef["val"], predef["test"]
            tr, va, te = _split_indices(len(src), train_frac, val_frac, seed + seed_offset)
            return take(src, tr), take(src, va), take(src, te)

        # Per-policy, per-ID/OOD partitioning. If the dataset already carries
        # explicit train/val/test episode assignments, honor them; otherwise
        # fall back to the original random episode split.
        id_tr, id_va, id_te = partition_triplet(id_idx, id_predef, pid)
        ood_tr, ood_va, ood_te = partition_triplet(ood_idx, ood_predef, 1000 + pid)

        buckets = {
            ("ID", "train"): shifted.subset(id_tr),
            ("ID", "val"):   shifted.subset(id_va),
            ("ID", "test"):  shifted.subset(id_te),
            ("OOD", "train"): shifted.subset(ood_tr),
            ("OOD", "val"):   shifted.subset(ood_va),
            ("OOD", "test"):  shifted.subset(ood_te),
        }

        if spec.train == "ID":
            train_stores.append(buckets[("ID", "train")])
            val_stores.append(buckets[("ID", "val")])
        elif spec.train == "OOD":
            train_stores.append(buckets[("OOD", "train")])
            val_stores.append(buckets[("OOD", "val")])
        # NONE -> no train/val

        if spec.test == "ID":
            test_stores.append(buckets[("ID", "test")])
        elif spec.test == "OOD":
            test_stores.append(buckets[("OOD", "test")])

    if degenerate_buckets:
        raise RuntimeError(
            "Shift-partition was empty for required buckets: "
            + "; ".join(degenerate_buckets)
            + ". Increase ood_fraction, use shift.kind=per_policy_cluster, or add more episodes per policy."
        )

    train_store = _concat_stores(train_stores)
    val_store = _concat_stores(val_stores)
    test_store = _concat_stores(test_stores)

    if train_store is None or len(train_store) == 0:
        raise RuntimeError("Experiment composition produced empty training set")
    if test_store is None or len(test_store) == 0:
        raise RuntimeError("Experiment composition produced empty test set")

    # Disjointness invariant: each base-store episode contributes to exactly one
    # of train/val/test because _split_indices returns disjoint index partitions
    # over each (policy, ID/OOD) bucket, and each bucket goes to at most one
    # destination per experiment spec. We rely on that construction rather than
    # re-checking: `_concat_stores` reindexes `episode_id` sequentially so the
    # downstream meta no longer carries base-store identity.

    # 3) normalization stats computed from TRAIN only
    s_mean, s_std = train_store.state_stats()
    a_mean, a_std = train_store.action_stats()

    train_dataset = None
    val_dataset = None
    test_dataset = None
    train_unit_map = None

    if train_store is not None and len(train_store) > 0:
        train_dataset = PolicyDataset(
            train_store,
            history_k=history_k,
            shuffle_history=shuffle_history_train,
            behavior_unit=behavior_unit,
            unit_window_size=unit_window_size,
            register_units=use_unit_latents,
            state_mean=s_mean,
            state_std=s_std,
            action_mean=a_mean,
            action_std=a_std,
            seed=seed,
        )
        train_unit_map = train_dataset.unit_map
    if val_store is not None and len(val_store) > 0:
        val_dataset = PolicyDataset(
            val_store,
            history_k=history_k,
            shuffle_history=False,
            behavior_unit=behavior_unit,
            unit_window_size=unit_window_size,
            known_unit_map=train_unit_map if use_unit_latents else None,
            state_mean=s_mean,
            state_std=s_std,
            action_mean=a_mean,
            action_std=a_std,
            seed=seed,
        )
    if test_store is not None and len(test_store) > 0:
        test_dataset = PolicyDataset(
            test_store,
            history_k=history_k,
            shuffle_history=False,
            behavior_unit=behavior_unit,
            unit_window_size=unit_window_size,
            known_unit_map=train_unit_map if use_unit_latents else None,
            state_mean=s_mean,
            state_std=s_std,
            action_mean=a_mean,
            action_std=a_std,
            seed=seed,
        )

    def make_loader(dataset, shuffle, bs):
        if dataset is None or len(dataset) == 0:
            return None
        return DataLoader(
            dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            drop_last=False, pin_memory=True, persistent_workers=num_workers > 0,
        )

    return {
        "train_store": train_store,
        "val_store": val_store,
        "test_store": test_store,
        "train_loader": make_loader(train_dataset, True, batch_size),
        "val_loader": make_loader(val_dataset, False, eval_batch_size),
        "test_loader": make_loader(test_dataset, False, eval_batch_size),
        "state_mean": s_mean, "state_std": s_std,
        "action_mean": a_mean, "action_std": a_std,
        "n_train_policies": n_train_policies,
        "n_train_units": int(train_dataset.n_units) if train_dataset is not None else 0,
        "train_unit_map": train_unit_map,
        "behavior_unit": behavior_unit,
        "unit_window_size": int(unit_window_size),
        "use_unit_latents": bool(use_unit_latents),
        "state_dim": base_store.state_dim,
        "action_dim": base_store.action_dim,
        "action_kind": base_store.action_kind,
        "n_actions": base_store.n_actions,
        "history_k": history_k,
        "shift_strength": shift_strength,
        "shift_overlap": shift_overlap,
        "shift_overlap_ratio": shift_overlap_ratio,
        "effective_shared": effective_shared,
        "shift_fallback": shift_fallback,
    }
