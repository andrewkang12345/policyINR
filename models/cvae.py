"""Conditional VAE policy-representation model.

Spec:
  condition:     current_state
  encoder input: past (s, a) history (shuffled bag)
  decoder output: next_action (reconstruction)
  policy repr:   encoder mean mu
  variational:   used for action inference/rollout, not the main object
                 of representation study (we use mu as the representation).
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS
from .base import RepresentationModel, HistoryEncoder, MLP, ActionHead


@MODELS.register("cvae")
class CVAE(RepresentationModel):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        latent_dim: int = 64,
        decoder_hidden: int = 256,
        kl_weight: float = 1e-2,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.action_kind = action_kind

        self.history = HistoryEncoder(
            state_dim=state_dim, action_dim=action_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            history_k=history_k, permutation_invariant=True, dropout=dropout,
            action_kind=action_kind, n_actions=n_actions,
        )
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        self.state_embed = MLP([state_dim, d_model, d_model])
        # Decoder body produces a hidden; ActionHead projects to the right
        # output shape and computes the right loss for the action kind.
        self.decoder_body = MLP(
            [latent_dim + d_model, decoder_hidden, decoder_hidden, decoder_hidden]
        )
        self.action_head = ActionHead(
            hidden_dim=decoder_hidden, action_dim=action_dim,
            action_kind=action_kind, n_actions=n_actions,
        )

    def _encode(self, past_states, past_actions):
        h = self.history(past_states, past_actions)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar

    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past_s = batch["past_states"]
        past_a = batch["past_actions"]
        s_t = batch["current_state"]
        a_t = batch["next_action"]

        mu, logvar = self._encode(past_s, past_a)
        z = self._reparam(mu, logvar) if self.training else mu
        cond = self.state_embed(s_t)
        h = self.decoder_body(torch.cat([z, cond], dim=-1))
        out = self.action_head(h)

        recon = self.action_head.loss(out, a_t)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.kl_weight * kl
        return {"loss": loss, "recon": recon.detach(), "kl": kl.detach(), "pred": out}

    @torch.no_grad()
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, _ = self._encode(batch["past_states"], batch["past_actions"])
        return mu

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, _ = self._encode(batch["past_states"], batch["past_actions"])
        cond = self.state_embed(batch["current_state"])
        h = self.decoder_body(torch.cat([mu, cond], dim=-1))
        return self.action_head.predict(self.action_head(h))
