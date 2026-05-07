"""State-Action BEM (Behavioral Embedding Map) baseline.

For each episode we represent the policy by an empirical histogram of
its (s, a) visitations, quantized into a shared k-means codebook fit
once over the training data. This is the simple non-parametric baseline
referenced in Pacchiano (BEM via trajectory-level visitation embeddings)
and Mutti (discounted state-action occupancy compression).

Per the locked spec for our case: k=64, joint (s,a) clusters, embedding
is the L1-normalized count vector. predict_action returns the mean
action of the cluster assignment of the most-recent (s,a) pair (so
there is a non-trivial generative metric to aggregate).

The module exposes a dummy `nn.Parameter` so the AdamW optimizer in
`train.Trainer` accepts it; the parameter has no role and `forward`
returns a zero loss attached to it. K-means is fit lazily on the first
training batch (subsequent batches do nothing).
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from utils.registry import MODELS
from .base import RepresentationModel


@MODELS.register("state_action_bem")
class StateActionBEM(RepresentationModel):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        n_clusters: int = 64,
        action_kind: str = "continuous",
        n_actions: int | None = None,
        kmeans_max_iter: int = 50,
        kmeans_max_samples: int = 50000,
        seed: int = 0,
    ):
        super().__init__()
        self.latent_dim = n_clusters
        self.n_clusters = n_clusters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_kind = action_kind
        self.n_actions = n_actions
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_max_samples = kmeans_max_samples
        self.seed = seed

        # Codebook centers — shape (K, sa_dim). Discrete actions are
        # one-hot expanded so the (s,a) feature dim is state_dim + n_actions
        # there; continuous actions use raw action_dim.
        sa_dim = state_dim + (n_actions if action_kind == "discrete" else action_dim)
        self.register_buffer("centers", torch.zeros(n_clusters, sa_dim))
        # Per-cluster mean action — (K, action_dim) for continuous, or
        # (K, n_actions) one-hot accumulator for discrete.
        if action_kind == "discrete":
            self.register_buffer("cluster_action", torch.zeros(n_clusters, n_actions))
        else:
            self.register_buffer("cluster_action", torch.zeros(n_clusters, action_dim))
        self.register_buffer("fitted", torch.zeros(1))   # 0/1 flag

        # Dummy parameter so AdamW has something to optimize.
        self.dummy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # k-means / fit
    def _featurize_sa(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(B, K, S) + (B, K, A or B, K) -> (B*K, sa_dim)."""
        B = s.shape[0]
        K = s.shape[1] if s.ndim == 3 else 1
        s_flat = s.reshape(B * K, -1)
        if self.action_kind == "discrete":
            a_long = a.long().reshape(B * K)
            a_oh = torch.zeros(B * K, self.n_actions, device=a.device, dtype=s.dtype)
            a_oh.scatter_(1, a_long.unsqueeze(1), 1.0)
            return torch.cat([s_flat, a_oh], dim=-1)
        a_flat = a.reshape(B * K, -1)
        return torch.cat([s_flat, a_flat], dim=-1)

    def _fit_from_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        # Assemble (s,a) features from past + current
        ps = batch["past_states"]; pa = batch["past_actions"]
        cs = batch["current_state"][:, None, :]
        if self.action_kind == "discrete":
            ca = batch["next_action"][:, None]
        else:
            ca = batch["next_action"][:, None, :]
        s_all = torch.cat([ps, cs], dim=1)
        a_all = torch.cat([pa, ca], dim=1)
        feats = self._featurize_sa(s_all, a_all).detach().cpu().numpy().astype(np.float32)

        if feats.shape[0] > self.kmeans_max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(feats.shape[0], size=self.kmeans_max_samples, replace=False)
            feats = feats[idx]
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.seed,
            max_iter=self.kmeans_max_iter, n_init=3, batch_size=4096,
        ).fit(feats)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        self.centers.data.copy_(centers.to(self.centers.device))

        # Per-cluster mean action: assign all features and average.
        labels = km.labels_
        if self.action_kind == "discrete":
            # action portion of feats is one-hot already; mean is the per-cluster
            # marginal probability over actions.
            action_part = feats[:, self.state_dim:]
            mean_act = np.zeros((self.n_clusters, self.n_actions), dtype=np.float32)
            counts = np.zeros(self.n_clusters, dtype=np.float32)
            for f, l in zip(action_part, labels):
                mean_act[l] += f; counts[l] += 1
            counts = np.maximum(counts, 1.0)
            mean_act = mean_act / counts[:, None]
            self.cluster_action.data.copy_(torch.from_numpy(mean_act).to(self.cluster_action.device))
        else:
            action_part = feats[:, self.state_dim:]
            mean_act = np.zeros((self.n_clusters, self.action_dim), dtype=np.float32)
            counts = np.zeros(self.n_clusters, dtype=np.float32)
            for f, l in zip(action_part, labels):
                mean_act[l] += f; counts[l] += 1
            counts = np.maximum(counts, 1.0)
            mean_act = mean_act / counts[:, None]
            self.cluster_action.data.copy_(torch.from_numpy(mean_act).to(self.cluster_action.device))
        self.fitted.fill_(1.0)

    # ------------------------------------------------------------------
    # core forward / repr / predict
    def _assign(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (N, sa_dim) -> (N,) cluster ids."""
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
        d = (
            (feats * feats).sum(dim=1, keepdim=True)
            - 2.0 * feats @ self.centers.t()
            + (self.centers * self.centers).sum(dim=1)[None]
        )
        return d.argmin(dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.fitted.item() < 0.5:
            self._fit_from_batch(batch)
        # Loss: zero, attached to the dummy parameter so backward succeeds.
        loss = (self.dummy * 0.0).sum()
        # For logging convenience, produce a per-batch reconstruction surrogate
        # (mean L2 of the predicted vs target action).
        with torch.no_grad():
            pred = self.predict_action(batch)
            if self.action_kind == "discrete":
                tgt = batch["next_action"].long()
                acc = (pred == tgt).float().mean()
                surrogate = 1.0 - acc
            else:
                surrogate = (pred - batch["next_action"]).pow(2).mean()
        return {"loss": loss + surrogate.detach() * 0.0,
                "recon": surrogate, "pred": pred}

    @torch.no_grad()
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fitted.item() < 0.5:
            self._fit_from_batch(batch)
        ps = batch["past_states"]; pa = batch["past_actions"]
        cs = batch["current_state"][:, None, :]
        if self.action_kind == "discrete":
            ca = batch["next_action"][:, None]
        else:
            ca = batch["next_action"][:, None, :]
        s_all = torch.cat([ps, cs], dim=1)
        a_all = torch.cat([pa, ca], dim=1)
        B, KP1 = s_all.shape[0], s_all.shape[1]
        feats = self._featurize_sa(s_all, a_all)                  # (B*(K+1), sa_dim)
        ids = self._assign(feats)                                  # (B*(K+1),)
        ids = ids.view(B, KP1)
        # Histogram per row
        hist = torch.zeros(B, self.n_clusters, device=ids.device, dtype=torch.float32)
        hist.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.float32))
        # L1 normalize
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
        return hist

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fitted.item() < 0.5:
            return batch["next_action"] * 0.0  # before fit returns zeros
        # Use the most recent past (s,a) pair to query the cluster, then
        # return that cluster's mean action.
        ps = batch["past_states"][:, -1, :]
        pa = batch["past_actions"][:, -1, :] if batch["past_actions"].ndim == 3 else batch["past_actions"][:, -1]
        if self.action_kind == "discrete":
            a_long = pa.long().reshape(-1)
            a_oh = torch.zeros(a_long.shape[0], self.n_actions, device=ps.device, dtype=ps.dtype)
            a_oh.scatter_(1, a_long.unsqueeze(1), 1.0)
            feats = torch.cat([ps, a_oh], dim=-1)
            ids = self._assign(feats)
            # argmax over per-cluster mean action distribution
            return self.cluster_action[ids].argmax(dim=-1)
        feats = torch.cat([ps, pa], dim=-1)
        ids = self._assign(feats)
        return self.cluster_action[ids]
