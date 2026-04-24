"""State-distribution shift assignment.

The policy-representation experiments care about state-support shift
*within* a policy: we want each policy to have both an ID subset
(broad state support) and an OOD subset (smaller, similar state
support — far from the ID centroid). Global clustering over all
policies is ill-conditioned because some policies can end up with
zero ID or zero OOD episodes; all composition experiments that need
both partitions then silently degrade. We therefore default to
**per-policy** clustering.

Registered strategies:
  - per_policy_cluster  : (default) cluster each policy's episodes into
                           K=2 groups on a richer per-episode summary
                           (mean, std, first-state, last-state). The
                           smaller + more-peripheral cluster is OOD.
  - per_policy_quantile : deterministic top-`ood_fraction` farthest-from
                           -median episodes per policy are OOD.
  - mean_cluster        : the original global strategy (kept for ablation).

All strategies also return a `shift_strength` dict:
    {policy_id: mahalanobis_distance_between_ID_and_OOD_centroids}
so the trainer can log how meaningful the shift actually is per run.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np

from utils.registry import SHIFTS
from .base import EpisodeStore, EpisodeMeta


def _episode_features(states: List[np.ndarray]) -> np.ndarray:
    """Per-episode summary: mean, std, first-step, last-step.

    This captures state-support better than mean alone because two
    different regions with the same mean can have very different
    spread and endpoints.
    """
    feats = []
    for s in states:
        mu = s.mean(0)
        sd = s.std(0)
        first = s[0]
        last = s[-1]
        feats.append(np.concatenate([mu, sd, first, last]))
    return np.stack(feats, axis=0) if feats else np.zeros((0, 1))


def _two_means(X: np.ndarray, seed: int = 0, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=np.int64)
    init = rng.choice(X.shape[0], size=2, replace=False)
    c = X[init].copy()
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(iters):
        d2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_c = np.stack([X[labels == k].mean(0) if (labels == k).any() else c[k]
                          for k in range(2)])
        c = new_c
    return labels


def _mahalanobis(a_feats: np.ndarray, b_feats: np.ndarray) -> float:
    """Coarse shift-strength: std-normalized centroid distance."""
    if a_feats.shape[0] == 0 or b_feats.shape[0] == 0:
        return float("nan")
    pooled_std = np.std(np.concatenate([a_feats, b_feats], axis=0), axis=0) + 1e-6
    d = (a_feats.mean(0) - b_feats.mean(0)) / pooled_std
    return float(np.linalg.norm(d) / np.sqrt(len(pooled_std)))


def _shift_diagnostics_from_flags(store: EpisodeStore, flags: Sequence[bool]) -> Tuple[Dict[int, float], float, float]:
    feats = _episode_features(store.states)
    if feats.shape[0] == 0:
        return {}, float("nan"), float("nan")
    feats_s = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-6)
    flags = np.asarray(flags, dtype=bool)
    strength: Dict[int, float] = {}
    pol_to_idx: Dict[int, List[int]] = {}
    for i, m in enumerate(store.meta):
        pol_to_idx.setdefault(int(m.policy_id), []).append(i)
    pids = sorted(pol_to_idx)
    for pid in pids:
        idx = np.array(pol_to_idx[pid], dtype=np.int64)
        fp = flags[idx]
        if fp.any() and (~fp).any():
            strength[pid] = _mahalanobis(feats_s[idx[~fp]], feats_s[idx[fp]])
        else:
            strength[pid] = float("nan")

    def _cross(buckets):
        cs = [feats_s[np.array(idx)].mean(0) for idx in buckets if len(idx)]
        if len(cs) < 2:
            return float("nan")
        cs = np.stack(cs, axis=0)
        return float(np.mean([
            np.linalg.norm(cs[i] - cs[j])
            for i in range(len(cs))
            for j in range(i + 1, len(cs))
        ]))

    id_buckets = [[i for i in pol_to_idx[p] if not flags[i]] for p in pids]
    ood_buckets = [[i for i in pol_to_idx[p] if flags[i]] for p in pids]
    disp_id = _cross(id_buckets)
    disp_ood = _cross(ood_buckets)
    overlap_ratio = float(disp_ood / disp_id) if disp_id and disp_id == disp_id else float("nan")
    return strength, float(disp_ood), overlap_ratio


def _assign_per_policy(
    store: EpisodeStore,
    cluster_fn,
    *,
    ood_fraction: float,
    seed: int,
    min_per_partition: int,
) -> Tuple[List[bool], Dict[int, float]]:
    """Apply `cluster_fn(X_policy, seed, ood_fraction)->is_ood[]` per policy.

    Guarantees (when possible) at least `min_per_partition` episodes in
    each of ID/OOD for every policy.
    """
    n = len(store)
    is_ood = np.zeros(n, dtype=bool)
    shift_strength: Dict[int, float] = {}

    # group episode indices by policy
    pol_to_idx: Dict[int, List[int]] = {}
    for i, m in enumerate(store.meta):
        pol_to_idx.setdefault(int(m.policy_id), []).append(i)

    for pid, idx_list in pol_to_idx.items():
        if len(idx_list) < 2 * min_per_partition:
            # Too few episodes to meaningfully split — put half in OOD
            # by index order so the experiment still runs.
            half = max(1, len(idx_list) // 2)
            for i in idx_list[:half]:
                is_ood[i] = False
            for i in idx_list[half:]:
                is_ood[i] = True
            shift_strength[pid] = float("nan")
            continue

        states = [store.states[i] for i in idx_list]
        X = _episode_features(states)
        # standardize features before clustering — high-dim envs otherwise
        # let a few large-magnitude dims dominate the distance.
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        flags = cluster_fn(X, seed + pid, ood_fraction)
        # rebalance to respect min_per_partition
        n_pol = len(idx_list)
        target_ood = max(min_per_partition, min(n_pol - min_per_partition,
                                                 int(round(ood_fraction * n_pol))))
        if abs(flags.sum() - target_ood) > 0:
            # rank by distance to the majority cluster centroid and take the
            # top target_ood as OOD (stable tie-breaking).
            center = X[~flags].mean(0) if (~flags).any() else X.mean(0)
            d = np.linalg.norm(X - center, axis=1)
            order = np.argsort(-d)
            flags = np.zeros_like(flags)
            flags[order[:target_ood]] = True

        for local_i, global_i in enumerate(idx_list):
            is_ood[global_i] = bool(flags[local_i])

        shift_strength[pid] = _mahalanobis(X[~flags], X[flags])

    return is_ood.tolist(), shift_strength


def _cluster_with_two_means(X: np.ndarray, seed: int, ood_fraction: float) -> np.ndarray:
    labels = _two_means(X, seed=seed)
    # OOD = whichever cluster is smaller AND farther from global centroid
    global_c = X.mean(0)
    c0 = X[labels == 0].mean(0) if (labels == 0).any() else global_c
    c1 = X[labels == 1].mean(0) if (labels == 1).any() else global_c
    d0 = np.linalg.norm(c0 - global_c)
    d1 = np.linalg.norm(c1 - global_c)
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if d0 > d1 and n0 <= n1:
        ood_cluster = 0
    elif d1 > d0 and n1 <= n0:
        ood_cluster = 1
    else:
        ood_cluster = 0 if n0 < n1 else 1
    return (labels == ood_cluster)


def _cluster_with_quantile(X: np.ndarray, seed: int, ood_fraction: float) -> np.ndarray:
    med = np.median(X, axis=0)
    d = np.linalg.norm(X - med, axis=1)
    k = max(1, int(round(ood_fraction * X.shape[0])))
    order = np.argsort(-d)
    flags = np.zeros(X.shape[0], dtype=bool)
    flags[order[:k]] = True
    return flags


def _step_level_shared_mask(
    store: EpisodeStore,
    rng: np.random.Generator,
    knn_k: int = 20,
    sub_per_policy: int = 3000,
    density_quantile: float = 0.25,
) -> Tuple[List[np.ndarray], float]:
    """Find shared state-space region at the STEP level.

    A state is "shared" when its k nearest neighbors in the **pooled**
    (all-policy) subsample have a roughly uniform distribution over
    policy ids — i.e., knowing the state doesn't tell you the policy.
    We use normalized entropy over the neighbor pid distribution in
    [0, 1] (1 = perfectly uniform, 0 = single policy) and threshold by
    quantile so the top `density_quantile` fraction of steps is shared.

    Returns per-episode bool masks of shape (T_i,) plus the dataset-
    wide fraction of shared steps (should equal `density_quantile`).
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise RuntimeError("sklearn is required for shared_region shift") from e

    step_states, step_pids, step_eid, step_t = [], [], [], []
    for ei, s in enumerate(store.states):
        pid = int(store.meta[ei].policy_id)
        T = s.shape[0]
        step_states.append(s)
        step_pids.append(np.full(T, pid, dtype=np.int64))
        step_eid.append(np.full(T, ei, dtype=np.int64))
        step_t.append(np.arange(T, dtype=np.int64))
    X = np.concatenate(step_states, axis=0).astype(np.float32)
    pids = np.concatenate(step_pids)
    eid = np.concatenate(step_eid)
    t_local = np.concatenate(step_t)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    unique_pids = np.unique(pids)
    P = unique_pids.size
    pid_to_ix = {int(p): i for i, p in enumerate(unique_pids)}

    # global pooled subsample balanced across policies
    sub_parts = []
    for p in unique_pids:
        pool = np.where(pids == p)[0]
        sel = pool if pool.size <= sub_per_policy else rng.choice(pool, size=sub_per_policy, replace=False)
        sub_parts.append(sel)
    idx_sub = np.concatenate(sub_parts)
    X_sub = X[idx_sub]
    pid_sub = pids[idx_sub]
    pid_ix_sub = np.array([pid_to_ix[int(p)] for p in pid_sub], dtype=np.int64)

    nn = NearestNeighbors(n_neighbors=min(knn_k, X_sub.shape[0]), algorithm="auto")
    nn.fit(X_sub)

    # chunked kNN for memory
    chunk = 20000
    shared_score = np.empty(X.shape[0], dtype=np.float32)
    log_P = float(np.log(max(P, 2)))
    for s in range(0, X.shape[0], chunk):
        _, idxs = nn.kneighbors(X[s:s + chunk])
        nb_pid_ix = pid_ix_sub[idxs]  # (chunk, k)
        # one-hot-sum -> counts, vectorized
        one_hot = np.zeros((nb_pid_ix.shape[0], P), dtype=np.float32)
        np.add.at(one_hot, (np.arange(nb_pid_ix.shape[0])[:, None], nb_pid_ix), 1.0)
        probs = one_hot / max(1, nb_pid_ix.shape[1])
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / log_P
        shared_score[s:s + chunk] = entropy  # in [0, 1]

    # Argsort rather than quantile — robust to massive ties at entropy==0
    # (which is common: when a policy's state distribution is disjoint from
    # others, most of its steps have all-same-policy neighbors).
    n_target = max(1, int(round(density_quantile * X.shape[0])))
    order = np.argsort(-shared_score, kind="stable")
    is_shared = np.zeros(X.shape[0], dtype=bool)
    is_shared[order[:n_target]] = True
    shared_fraction = float(is_shared.mean())
    # "effective" shared mass = fraction of selected steps that have ANY
    # cross-policy neighbors (entropy > 0). When low, the env's policies
    # are near-disjoint in state space and state-distribution will still
    # leak policy identity; the caller should inspect this diagnostic.
    effective_shared = float((shared_score[order[:n_target]] > 1e-4).mean())
    _step_level_shared_mask._effective_shared = effective_shared

    per_ep: List[np.ndarray] = [np.zeros(s.shape[0], dtype=bool) for s in store.states]
    for i in range(X.shape[0]):
        per_ep[int(eid[i])][int(t_local[i])] = bool(is_shared[i])
    return per_ep, shared_fraction


@SHIFTS.register("shared_region")
def shared_region_shift(
    store: EpisodeStore,
    ood_fraction: float = 0.3,
    seed: int = 0,
    min_per_partition: int = 3,
    knn_k: int = 5,
    sub_per_policy: int = 3000,
    density_quantile: float = 0.25,
    min_shared_steps_frac: float = 0.2,
) -> List[bool]:
    """Construct a shared-across-policy OOD region at the STEP level.

    Motivation: episode-level centroid assignment fails on Minari because
    simple/medium/expert policies occupy near-disjoint state regions —
    there's no per-episode summary that's shared across policies. So we
    pick a shared region at the STEP level, mark per-step which states
    are in it, and then (a) classify each episode as OOD if it has
    enough shared steps, (b) carry the per-step mask through to the
    `PolicyDataset` so OOD past-histories and current-states are drawn
    *only* from shared-region steps.

    Mechanism:
      1. `_step_level_shared_mask` builds a bool mask per episode over
         timesteps where every policy has neighbors in state space.
      2. Per policy, rank episodes by shared-step count; take the top
         `ood_fraction` as OOD-eligible.
      3. Store the per-episode shared mask on meta.extras["shared_mask"]
         so PolicyDataset can restrict step-level sampling.

    Written onto the function for diagnostics:
      last_shift_strength : per-policy ID-vs-OOD Mahalanobis
      last_overlap        : cross-policy OOD centroid dispersion
      last_overlap_ratio  : cross-policy OOD dispersion / ID dispersion
      last_shared_step_fraction : fraction of all steps flagged shared
    """
    n_total = len(store)
    if n_total == 0:
        shared_region_shift.last_shift_strength = {}
        shared_region_shift.last_overlap = float("nan")
        shared_region_shift.last_overlap_ratio = float("nan")
        shared_region_shift.last_shared_step_fraction = float("nan")
        return []

    rng = np.random.default_rng(seed)
    per_ep_mask, shared_step_fraction = _step_level_shared_mask(
        store, rng=rng, knn_k=knn_k, sub_per_policy=sub_per_policy,
        density_quantile=density_quantile,
    )
    effective = getattr(_step_level_shared_mask, "_effective_shared", float("nan"))

    # If the picked "shared" steps are overwhelmingly not cross-policy
    # (common on Minari simple/medium/expert where state supports are
    # near-disjoint), forcing a step-level mask is worse than a clean
    # per-policy cluster. Fall back and log it.
    FALLBACK_THRESHOLD = 0.25
    if not (effective >= FALLBACK_THRESHOLD):
        # blank masks => PolicyDataset samples from whole episodes; OOD
        # assignment below falls through to per-policy cluster.
        for ei in range(n_total):
            store.meta[ei].extras["shared_mask"] = None
        fallback_flags = per_policy_cluster(
            store, ood_fraction=ood_fraction, seed=seed,
            min_per_partition=min_per_partition,
        )
        strength = getattr(per_policy_cluster, "last_shift_strength", {})
        shared_region_shift.last_shift_strength = strength
        shared_region_shift.last_overlap = float("nan")
        shared_region_shift.last_overlap_ratio = float("nan")
        shared_region_shift.last_shared_step_fraction = float(shared_step_fraction)
        shared_region_shift.last_effective_shared = float(effective)
        shared_region_shift.last_fallback = "per_policy_cluster"
        return fallback_flags

    shared_region_shift.last_fallback = "none"
    # write masks onto meta so PolicyDataset can use them
    for ei in range(n_total):
        store.meta[ei].extras["shared_mask"] = per_ep_mask[ei]

    pol_to_idx: Dict[int, List[int]] = {}
    for i, m in enumerate(store.meta):
        pol_to_idx.setdefault(int(m.policy_id), []).append(i)
    pids = sorted(pol_to_idx.keys())

    shared_counts = np.array([int(m.sum()) for m in per_ep_mask])
    min_shared = int(np.ceil(min_shared_steps_frac * max(1, np.median([s.shape[0] for s in store.states]))))

    is_ood = np.zeros(n_total, dtype=bool)
    for pid in pids:
        idx = np.array(pol_to_idx[pid])
        n_pol = idx.size
        # only episodes with enough shared steps are OOD-eligible
        eligible = idx[shared_counts[idx] >= min_shared]
        target = max(min_per_partition, min(n_pol - min_per_partition,
                                             int(round(ood_fraction * n_pol))))
        if n_pol < 2 * min_per_partition:
            target = max(1, n_pol // 2)
        if eligible.size == 0:
            # fall back to highest-shared-count episodes even if under threshold
            eligible = idx[np.argsort(-shared_counts[idx])[:target]]
        if eligible.size < target:
            # pad with next-most-shared from ineligible set
            rest = np.setdiff1d(idx, eligible, assume_unique=False)
            rest = rest[np.argsort(-shared_counts[rest])]
            eligible = np.concatenate([eligible, rest[:target - eligible.size]])
        order = eligible[np.argsort(-shared_counts[eligible])][:target]
        for i in order:
            is_ood[int(i)] = True

    # diagnostics: compute Mahalanobis & cross-policy dispersion using
    # per-episode state *means over shared steps only* for OOD, and over
    # all steps for ID. This reflects what the encoder will actually see.
    def _ep_feat_shared_only(ei: int) -> np.ndarray:
        mask = per_ep_mask[ei]
        s = store.states[ei]
        if mask.any():
            s_slice = s[mask]
        else:
            s_slice = s
        return np.concatenate([s_slice.mean(0), s_slice.std(0),
                               s_slice[0], s_slice[-1]])

    feats_all = []
    for ei in range(n_total):
        if is_ood[ei]:
            feats_all.append(_ep_feat_shared_only(ei))
        else:
            s = store.states[ei]
            feats_all.append(np.concatenate([s.mean(0), s.std(0), s[0], s[-1]]))
    feats_all = np.stack(feats_all, axis=0).astype(np.float32)
    feats_all = (feats_all - feats_all.mean(0)) / (feats_all.std(0) + 1e-6)

    strength: Dict[int, float] = {}
    for pid in pids:
        idx = np.array(pol_to_idx[pid])
        po = is_ood[idx]
        if po.any() and (~po).any():
            strength[pid] = _mahalanobis(feats_all[idx[~po]], feats_all[idx[po]])
        else:
            strength[pid] = float("nan")

    def _cross(buckets):
        cs = [feats_all[np.array(idx)].mean(0) for idx in buckets if len(idx)]
        if len(cs) < 2:
            return float("nan")
        cs = np.stack(cs)
        return float(np.mean([np.linalg.norm(cs[i] - cs[j])
                              for i in range(len(cs)) for j in range(i + 1, len(cs))]))
    id_b = [[i for i in pol_to_idx[p] if not is_ood[i]] for p in pids]
    ood_b = [[i for i in pol_to_idx[p] if is_ood[i]]     for p in pids]
    disp_id = _cross(id_b)
    disp_ood = _cross(ood_b)
    overlap_ratio = float(disp_ood / disp_id) if disp_id and disp_id == disp_id else float("nan")

    shared_region_shift.last_shift_strength = strength
    shared_region_shift.last_overlap = float(disp_ood)
    shared_region_shift.last_overlap_ratio = overlap_ratio
    shared_region_shift.last_shared_step_fraction = shared_step_fraction
    shared_region_shift.last_effective_shared = getattr(
        _step_level_shared_mask, "_effective_shared", float("nan"))
    return is_ood.tolist()


@SHIFTS.register("predefined_split")
def predefined_split(store: EpisodeStore, **_: dict) -> List[bool]:
    """Use per-episode predefined ID/OOD tags carried by the dataset itself."""
    flags: List[bool] = []
    missing = []
    for i, meta in enumerate(store.meta):
        split = str(meta.extras.get("predefined_split", "")).upper()
        if split not in {"ID", "OOD"}:
            missing.append(i)
            flags.append(False)
            continue
        flags.append(split == "OOD")
    if missing:
        raise RuntimeError(
            "predefined_split requires meta.extras['predefined_split'] in {'ID','OOD'} "
            f"for every episode; missing/invalid on indices {missing[:10]}"
        )
    strength, overlap, overlap_ratio = _shift_diagnostics_from_flags(store, flags)
    predefined_split.last_shift_strength = strength
    predefined_split.last_overlap = overlap
    predefined_split.last_overlap_ratio = overlap_ratio
    predefined_split.last_effective_shared = 1.0
    predefined_split.last_fallback = "none"
    return flags


@SHIFTS.register("per_policy_cluster")
def per_policy_cluster(store: EpisodeStore, ood_fraction: float = 0.3,
                        seed: int = 0, min_per_partition: int = 3) -> List[bool]:
    flags, strength = _assign_per_policy(
        store, _cluster_with_two_means,
        ood_fraction=ood_fraction, seed=seed,
        min_per_partition=min_per_partition,
    )
    per_policy_cluster.last_shift_strength = strength
    return flags


@SHIFTS.register("per_policy_quantile")
def per_policy_quantile(store: EpisodeStore, ood_fraction: float = 0.3,
                         seed: int = 0, min_per_partition: int = 3) -> List[bool]:
    flags, strength = _assign_per_policy(
        store, _cluster_with_quantile,
        ood_fraction=ood_fraction, seed=seed,
        min_per_partition=min_per_partition,
    )
    per_policy_quantile.last_shift_strength = strength
    return flags


@SHIFTS.register("mean_cluster")
def mean_cluster_shift(store: EpisodeStore, ood_fraction: float = 0.3, seed: int = 0) -> List[bool]:
    """Legacy global 2-means shift — kept for ablation only."""
    # global feature matrix using the same richer features
    feats = _episode_features(store.states)
    if feats.shape[0] == 0:
        mean_cluster_shift.last_shift_strength = {}
        return []
    feats_s = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    flags = _cluster_with_two_means(feats_s, seed=seed, ood_fraction=ood_fraction)
    target = max(1, int(round(ood_fraction * feats.shape[0])))
    if abs(flags.sum() - target) > max(2, 0.1 * feats.shape[0]):
        d = np.linalg.norm(feats_s - feats_s.mean(0), axis=1)
        order = np.argsort(-d)
        flags = np.zeros(feats.shape[0], dtype=bool)
        flags[order[:target]] = True
    # shift strength per policy for diagnostics
    strength: Dict[int, float] = {}
    for pid in {m.policy_id for m in store.meta}:
        pidx = np.array([i for i, m in enumerate(store.meta) if m.policy_id == pid])
        fp = flags[pidx]
        if fp.any() and (~fp).any():
            strength[int(pid)] = _mahalanobis(feats_s[pidx[~fp]], feats_s[pidx[fp]])
        else:
            strength[int(pid)] = float("nan")
    mean_cluster_shift.last_shift_strength = strength
    return flags.tolist()


def assign_state_shift(store: EpisodeStore, kind: str = "shared_region", **kwargs):
    """Returns (new_store, diagnostics_dict).

    diagnostics_dict keys:
      shift_strength : {pid -> ID-vs-OOD Mahalanobis per policy}
      shift_overlap  : mean pairwise distance between per-policy OOD
                       centroids (only set by shift strategies that
                       produce a shared OOD region; NaN otherwise).
    """
    fn = SHIFTS.get(kind)
    flags = fn(store, **kwargs)
    strength = getattr(fn, "last_shift_strength", {})
    overlap = getattr(fn, "last_overlap", float("nan"))
    new_meta: List[EpisodeMeta] = []
    for m, f in zip(store.meta, flags):
        new_meta.append(EpisodeMeta(
            episode_id=m.episode_id, policy_id=m.policy_id,
            is_ood=bool(f), source=m.source, extras=dict(m.extras),
        ))
    new_store = EpisodeStore(
        states=store.states, actions=store.actions, meta=new_meta,
        state_dim=store.state_dim, action_dim=store.action_dim,
        source=store.source,
        action_kind=store.action_kind, n_actions=store.n_actions,
    )
    overlap_ratio = getattr(fn, "last_overlap_ratio", float("nan"))
    effective_shared = getattr(fn, "last_effective_shared", float("nan"))
    fallback = getattr(fn, "last_fallback", "none")
    return new_store, {
        "shift_strength": strength,
        "shift_overlap": overlap,
        "shift_overlap_ratio": overlap_ratio,
        "effective_shared": effective_shared,
        "shift_fallback": fallback,
    }
