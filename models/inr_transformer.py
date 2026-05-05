"""Transformer-based INR variants with explicit conditioning semantics.

Two variants live here:

  * history-conditioned / amortized:
      z = encoder(history)
      action = f_theta(current_state; z)

  * fitted-latent:
      z_unit = learned latent for a behavior unit (episode or window)
      action = f_theta(current_state; z_unit)

For fitted-latent evaluation on unseen units, we use the standard INR
protocol: keep the shared INR weights frozen and optimize only the unit
latent against support state-action pairs from that unit. We describe
`z` cautiously as a behavior-function code / unit-level latent, not as a
canonical policy representation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS
from .base import RepresentationModel, PairEmbed, ActionHead


class FiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, gamma, beta):
        x = self.lin(x)
        x = self.norm(x)
        x = x * (1.0 + gamma) + beta
        return F.gelu(x)


class FiLMPolicyHead(nn.Module):
    """Shared INR backbone modulated by a latent code via FiLM."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        film_scale: float = 0.5,
        out_clip: float = 10.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.film_scale = film_scale

        self.in_proj = nn.Linear(state_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([FiLMBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_kind=action_kind,
            n_actions=n_actions,
            out_clip=out_clip,
        )
        self.film = nn.Linear(latent_dim, 2 * hidden_dim * n_blocks)

    def forward(self, state, z):
        params = self.film(z).view(z.size(0), self.n_blocks, 2, self.hidden_dim)
        params = torch.tanh(params) * self.film_scale
        h = F.gelu(self.in_norm(self.in_proj(state)))
        for i, blk in enumerate(self.blocks):
            gamma = params[:, i, 0]
            beta = params[:, i, 1]
            h = blk(h, gamma, beta)
        return self.action_head(h)


class TransformerHistoryEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 64,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.pair_embed = PairEmbed(
            state_dim,
            action_dim,
            d_model,
            action_kind=action_kind,
            n_actions=n_actions,
        )
        self.type_pair = nn.Parameter(0.02 * torch.randn(d_model))
        self.pos = nn.Parameter(0.02 * torch.randn(history_k, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.latent_head = nn.Linear(d_model, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, past_s, past_a):
        pair_tok = self.pair_embed(past_s, past_a) + self.type_pair
        pair_tok = pair_tok + self.pos[: pair_tok.size(1)][None]
        z = self.latent_head(self.encoder(pair_tok).mean(dim=1))
        return self.latent_norm(z)


@MODELS.register("inr_transformer_history_conditioned")
@MODELS.register("inr_transformer")
class INRTransformerHistoryConditioned(RepresentationModel):
    """History-conditioned transformer INR with amortized latent inference."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 64,
        head_hidden: int = 256,
        head_blocks: int = 3,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_k = history_k
        self.action_kind = action_kind
        self.encoder = TransformerHistoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            history_k=history_k,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            latent_dim=latent_dim,
            dropout=dropout,
            action_kind=action_kind,
            n_actions=n_actions,
        )
        self.policy_head = FiLMPolicyHead(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=head_hidden,
            n_blocks=head_blocks,
            action_kind=action_kind,
            n_actions=n_actions,
        )

    def _encode(self, past_s, past_a):
        return self.encoder(past_s, past_a)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self._encode(batch["past_states"], batch["past_actions"])
        out = self.policy_head(batch["current_state"], z)
        loss = self.policy_head.action_head.loss(out, batch["next_action"])
        return {"loss": loss, "pred": out}

    @torch.no_grad()
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(batch["past_states"], batch["past_actions"])

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._encode(batch["past_states"], batch["past_actions"])
        out = self.policy_head(batch["current_state"], z)
        return self.policy_head.action_head.predict(out)


@MODELS.register("inr_transformer_fitted_latent")
class INRTransformerFittedLatent(RepresentationModel):
    """INR with one learned latent per behavior unit and test-time latent fitting."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 64,
        head_hidden: int = 256,
        head_blocks: int = 3,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
        n_train_units: int = 0,
        behavior_unit: str = "episode",
        unit_latent_l2_weight: float = 1.0e-4,
        latent_infer_steps: int = 40,
        latent_infer_lr: float = 5.0e-2,
        latent_infer_l2_weight: float = 1.0e-4,
    ):
        super().__init__()
        if n_train_units <= 0:
            raise ValueError("INRTransformerFittedLatent requires n_train_units > 0")
        self.latent_dim = latent_dim
        self.history_k = history_k
        self.action_kind = action_kind
        self.behavior_unit = str(behavior_unit)
        self.unit_latent_l2_weight = float(unit_latent_l2_weight)
        self.latent_infer_steps = int(latent_infer_steps)
        self.latent_infer_lr = float(latent_infer_lr)
        self.latent_infer_l2_weight = float(latent_infer_l2_weight)
        self.unit_latents = nn.Embedding(n_train_units, latent_dim)
        nn.init.normal_(self.unit_latents.weight, mean=0.0, std=0.02)
        self.policy_head = FiLMPolicyHead(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=head_hidden,
            n_blocks=head_blocks,
            action_kind=action_kind,
            n_actions=n_actions,
        )

    def _fitted(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        has = batch["has_unit_latent"].bool()
        if not bool(has.any()):
            return None
        unit_ids = batch["unit_id"].clamp_min(0)
        return self.unit_latents(unit_ids)

    def _support_loss(
        self,
        z: torch.Tensor,
        past_states: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        bsz, k, state_dim = past_states.shape
        flat_states = past_states.reshape(bsz * k, state_dim)
        flat_z = z[:, None, :].expand(bsz, k, self.latent_dim).reshape(bsz * k, self.latent_dim)
        support_pred = self.policy_head(flat_states, flat_z)
        if self.action_kind == "discrete":
            flat_actions = past_actions.reshape(bsz * k)
        else:
            flat_actions = past_actions.reshape(bsz * k, -1)
        loss = self.policy_head.action_head.loss(support_pred, flat_actions)
        if self.latent_infer_l2_weight > 0:
            loss = loss + self.latent_infer_l2_weight * z.pow(2).mean()
        return loss

    def _infer_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.enable_grad():
            init = self.unit_latents.weight.detach().mean(dim=0, keepdim=True)
            z = init.expand(batch["current_state"].shape[0], -1).clone()
            z = z.to(batch["current_state"].device).detach().requires_grad_(True)
            for _ in range(self.latent_infer_steps):
                loss = self._support_loss(z, batch["past_states"], batch["past_actions"])
                grad = torch.autograd.grad(loss, z, only_inputs=True)[0]
                z = (z - self.latent_infer_lr * grad).detach().requires_grad_(True)
        return z.detach()

    def _select_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_fitted = self._fitted(batch)
        if z_fitted is None:
            return self._infer_latent(batch)
        has = batch["has_unit_latent"].bool()
        if bool(has.all()):
            return z_fitted
        z_used = z_fitted.clone()
        unknown = ~has
        sub_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.shape[0] == has.shape[0]:
                sub_batch[key] = value[unknown]
            else:
                sub_batch[key] = value
        z_used[unknown] = self._infer_latent(sub_batch)
        return z_used

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_used = self._select_latent(batch)
        pred = self.policy_head(batch["current_state"], z_used)
        action_loss = self.policy_head.action_head.loss(pred, batch["next_action"])
        loss = action_loss
        out = {"pred": pred, "action_loss": action_loss}

        z_fitted = self._fitted(batch)
        if z_fitted is not None and self.unit_latent_l2_weight > 0:
            has = batch["has_unit_latent"].bool()
            unit_latent_l2 = z_fitted[has].pow(2).mean()
            loss = loss + self.unit_latent_l2_weight * unit_latent_l2
            out["unit_latent_l2"] = unit_latent_l2

        out["loss"] = loss
        return out

    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For probe extraction, use the same latent-inference procedure for
        # both train and test units so the representation geometry is not
        # confounded by mixing table lookups with optimized latents.
        return self._infer_latent(batch)

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_used = self._select_latent(batch)
        out = self.policy_head(batch["current_state"], z_used)
        return self.policy_head.action_head.predict(out)


@MODELS.register("inr_transformer_infer_latent")
class INRTransformerInferLatent(RepresentationModel):
    """INR whose latent is always inferred from support history.

    This is the table-free fitted-latent variant: there is no per-train-unit
    embedding table, so train and test episodes both obtain z by the same
    support-loss gradient descent procedure.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 64,
        head_hidden: int = 256,
        head_blocks: int = 3,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
        behavior_unit: str = "episode",
        latent_infer_steps: int = 40,
        latent_infer_lr: float = 5.0e-2,
        latent_infer_l2_weight: float = 1.0e-4,
        latent_infer_create_graph: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_k = history_k
        self.action_kind = action_kind
        self.behavior_unit = str(behavior_unit)
        self.latent_infer_steps = int(latent_infer_steps)
        self.latent_infer_lr = float(latent_infer_lr)
        self.latent_infer_l2_weight = float(latent_infer_l2_weight)
        self.latent_infer_create_graph = bool(latent_infer_create_graph)
        self.z_init = nn.Parameter(torch.zeros(latent_dim))
        self.policy_head = FiLMPolicyHead(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=head_hidden,
            n_blocks=head_blocks,
            action_kind=action_kind,
            n_actions=n_actions,
        )

    def _support_loss(
        self,
        z: torch.Tensor,
        past_states: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        bsz, k, state_dim = past_states.shape
        flat_states = past_states.reshape(bsz * k, state_dim)
        flat_z = z[:, None, :].expand(bsz, k, self.latent_dim).reshape(bsz * k, self.latent_dim)
        support_pred = self.policy_head(flat_states, flat_z)
        if self.action_kind == "discrete":
            flat_actions = past_actions.reshape(bsz * k)
        else:
            flat_actions = past_actions.reshape(bsz * k, -1)
        loss = self.policy_head.action_head.loss(support_pred, flat_actions)
        if self.latent_infer_l2_weight > 0:
            loss = loss + self.latent_infer_l2_weight * z.pow(2).mean()
        return loss

    def _infer_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.enable_grad():
            bsz = batch["current_state"].shape[0]
            init = self.z_init[None, :].expand(bsz, -1).to(batch["current_state"].device)
            if self.training:
                z = init.clone().requires_grad_(True)
            else:
                z = init.detach().clone().requires_grad_(True)
            for _ in range(self.latent_infer_steps):
                loss = self._support_loss(z, batch["past_states"], batch["past_actions"])
                grad = torch.autograd.grad(
                    loss,
                    z,
                    only_inputs=True,
                    create_graph=self.training and self.latent_infer_create_graph,
                )[0]
                z = z - self.latent_infer_lr * grad
                if not self.training:
                    z = z.detach().requires_grad_(True)
        return z if self.training else z.detach()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self._infer_latent(batch)
        pred = self.policy_head(batch["current_state"], z)
        action_loss = self.policy_head.action_head.loss(pred, batch["next_action"])
        return {"loss": action_loss, "pred": pred, "action_loss": action_loss}

    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._infer_latent(batch)

    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._infer_latent(batch)
        out = self.policy_head(batch["current_state"], z)
        return self.policy_head.action_head.predict(out)


# Backward-compatible class alias for imports.
INRTransformer = INRTransformerHistoryConditioned
