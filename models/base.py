"""Common base class + shared modules for representation models.

Every model in this repo exposes:
  - `forward(batch) -> dict`           : training losses and predictions
  - `extract_representation(batch) -> Tensor (B, latent_dim)`

That uniform interface is what lets the eval code treat all models
identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationModel(nn.Module, ABC):
    latent_dim: int

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    @abstractmethod
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic point prediction of next_action given batch."""
        ...


class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU, dropout=0.0, out_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        if out_act is not None:
            layers.append(out_act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PairEmbed(nn.Module):
    """Embed a (state, action) pair into a token.

    Supports both continuous actions (concatenated float vector) and
    discrete actions (learned Embedding over `n_actions`). The flag is
    `action_kind`; for discrete spaces the action tensor is a long
    index of shape (B, K), otherwise a float of shape (B, K, A).
    """
    def __init__(self, state_dim, action_dim, d_model,
                 action_kind: str = "continuous", n_actions: int | None = None,
                 action_emb_dim: int = 64):
        super().__init__()
        self.action_kind = action_kind
        if action_kind == "discrete":
            assert n_actions is not None
            self.action_embed = nn.Embedding(n_actions, action_emb_dim)
            self.proj = nn.Linear(state_dim + action_emb_dim, d_model)
        else:
            self.proj = nn.Linear(state_dim + action_dim, d_model)

    def forward(self, s, a):
        if self.action_kind == "discrete":
            a_emb = self.action_embed(a.long())
            return self.proj(torch.cat([s, a_emb], dim=-1))
        return self.proj(torch.cat([s, a], dim=-1))


class ActionHead(nn.Module):
    """Produce model output for next-action prediction.

    continuous: a simple Linear(hidden, action_dim). Loss = MSE.
    discrete  : a Linear(hidden, n_actions) producing logits. Loss = CE.
    `predict` returns action indices for discrete, values for continuous.
    """
    def __init__(self, hidden_dim: int, action_dim: int,
                 action_kind: str = "continuous", n_actions: int | None = None,
                 out_clip: float = 0.0):
        super().__init__()
        self.action_kind = action_kind
        self.out_clip = out_clip
        if action_kind == "discrete":
            assert n_actions is not None
            self.logits = nn.Linear(hidden_dim, n_actions)
            self.n_actions = n_actions
        else:
            self.out = nn.Linear(hidden_dim, action_dim)
            self.action_dim = action_dim

    def forward(self, h):
        if self.action_kind == "discrete":
            return self.logits(h)
        y = self.out(h)
        if self.out_clip > 0:
            y = y.clamp(-self.out_clip, self.out_clip)
        return y

    def loss(self, out, target):
        if self.action_kind == "discrete":
            return F.cross_entropy(out, target.long())
        return F.mse_loss(out, target)

    def predict(self, out):
        if self.action_kind == "discrete":
            return out.argmax(dim=-1)
        return out


class HistoryEncoder(nn.Module):
    """Encode a set/sequence of K (state,action) pairs into a single vector.

    With `permutation_invariant=True`, we use mean-pooling over tokens
    (set encoder) — suitable for CVAE where history is shuffled.
    With `permutation_invariant=False`, we add learned positional
    embeddings and use a small Transformer encoder — suitable for INR
    where history order is meaningful.
    """
    def __init__(self, state_dim, action_dim, d_model=128, n_heads=4, n_layers=2,
                 history_k=16, permutation_invariant=False, dropout=0.0,
                 action_kind: str = "continuous", n_actions: int | None = None):
        super().__init__()
        self.permutation_invariant = permutation_invariant
        self.pair_embed = PairEmbed(state_dim, action_dim, d_model,
                                    action_kind=action_kind, n_actions=n_actions)
        if not permutation_invariant:
            self.pos = nn.Parameter(0.02 * torch.randn(history_k, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, past_states, past_actions):
        # past_states: (B, K, state_dim)
        tok = self.pair_embed(past_states, past_actions)  # (B, K, D)
        if not self.permutation_invariant:
            tok = tok + self.pos[: tok.size(1)][None]
        h = self.encoder(tok)  # (B, K, D)
        if self.permutation_invariant:
            z = h.mean(dim=1)
        else:
            # CLS-free: mean pool, which also works with transformer+pos
            z = h.mean(dim=1)
        return self.out_proj(z)  # (B, D)
