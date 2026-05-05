"""Linear probe on learned representations for policy classification.

Policy labels are used ONLY here, never during training. With small
amounts of per-episode data we still fit the probe only on the train
split embeddings and evaluate it on the held-out test split embeddings.

When the test set contains a 'novel' policy (i.e. a policy not present
in the training set), we report:
  - probe_acc_seen      : probe accuracy restricted to policies that
                          were seen in training
  - novel_mean_embed_dist: mean L2 distance from novel-policy episode
                          embeddings to the nearest seen-policy centroid
                          (higher = more cleanly separable)
"""

from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    *,
    train_seen_pids: Sequence[int],
    C: float = 1.0,
    max_iter: int = 1000,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if train_embeddings.shape[0] == 0 or test_embeddings.shape[0] == 0:
        return {"probe_acc": 0.0, "probe_acc_seen": 0.0, "n_probe": 0}

    seen_mask = np.isin(test_labels, np.array(list(train_seen_pids), dtype=np.int64))
    novel_mask = ~seen_mask

    clf, scaler = _fit_probe(train_embeddings, train_labels, C=C, max_iter=max_iter)
    if clf is None or scaler is None:
        out["probe_acc"] = float("nan")
        out["probe_acc_seen"] = float("nan")
        out["n_probe"] = int(test_embeddings.shape[0])
        out["n_classes"] = int(np.unique(train_labels).size)
        out["novel_mean_embed_dist"] = float("nan")
        out["n_novel"] = int(novel_mask.sum())
        return out

    pred = clf.predict(scaler.transform(test_embeddings))
    out["probe_acc"] = float(accuracy_score(test_labels, pred))
    out["n_probe"] = int(test_embeddings.shape[0])
    out["n_classes"] = int(np.unique(train_labels).size)

    # probe restricted to seen policies (the thing representation learning optimized for)
    if seen_mask.any():
        pred_seen = clf.predict(scaler.transform(test_embeddings[seen_mask]))
        acc_seen = accuracy_score(test_labels[seen_mask], pred_seen)
        out["probe_acc_seen"] = float(acc_seen)
    else:
        out["probe_acc_seen"] = float("nan")

    # novel-policy separability
    if novel_mask.any() and seen_mask.any():
        seen_centroids = {}
        for pid in np.unique(train_labels):
            seen_centroids[int(pid)] = train_embeddings[train_labels == pid].mean(0)
        cmat = np.stack(list(seen_centroids.values()))  # (C_seen, D)
        novel_embs = test_embeddings[novel_mask]
        dists = np.linalg.norm(novel_embs[:, None, :] - cmat[None, :, :], axis=-1)
        out["novel_mean_embed_dist"] = float(dists.min(axis=1).mean())
        out["n_novel"] = int(novel_mask.sum())
    else:
        out["novel_mean_embed_dist"] = float("nan")
        out["n_novel"] = 0

    return out


def cosine_knn_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    *,
    k: int = 5,
) -> Dict[str, float]:
    """Cosine kNN policy classification on train embeddings -> test embeddings."""
    if train_embeddings.shape[0] == 0 or test_embeddings.shape[0] == 0:
        return {"knn_acc1": float("nan"), "knn_acc5": float("nan"), "n_knn": 0}

    train_norm = _l2_normalize(train_embeddings.astype(np.float32, copy=False))
    test_norm = _l2_normalize(test_embeddings.astype(np.float32, copy=False))
    sims = test_norm @ train_norm.T

    k_eff = min(max(1, int(k)), train_norm.shape[0])
    topk = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    top1 = np.argmax(sims, axis=1)
    topk_labels = train_labels[topk]

    return {
        "knn_acc1": float(np.mean(train_labels[top1] == test_labels)),
        "knn_acc5": float(np.mean(np.any(topk_labels == test_labels[:, None], axis=1))),
        "n_knn": int(test_embeddings.shape[0]),
    }


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def _fit_probe(X, y, *, C, max_iter):
    classes = np.unique(y)
    if X.shape[0] == 0 or classes.size < 2:
        return None, None
    sc = StandardScaler().fit(X)
    Xn = sc.transform(X)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(Xn, y)
    return clf, sc
