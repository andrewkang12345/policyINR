"""VQ-VAE policy-representation model.

Same Transformer encoder as CVAE; replace the (mu, logvar) Gaussian
bottleneck with vector-quantization over a learned codebook. Output is
n_latents discrete code slots, each pointing at a 256-entry codebook of
latent_dim/n_latents-d vectors.

Per the choices locked upstream:
  codebook=256, n_latents=4, commitment_beta=0.25, latent_dim=64.

The flat representation returned by `extract_representation` is the
concatenated quantized vectors (latent_dim total), comparable in size to
the CVAE's mu.
"""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS
from .base import RepresentationModel, HistoryEncoder, MLP, ActionHead


class VectorQuantizer(nn.Module):
    """Standard EMA-free VQ with straight-through estimator.

    Splits incoming (B, latent_dim) into n_latents slots of size slot_dim
    each, quantizes every slot independently against a shared codebook
    of size codebook_size, returns the concatenated quantized vector and
    the codebook + commitment losses.
    """
    def __init__(self, latent_dim: int, n_latents: int = 4,
                 codebook_size: int = 256, beta: float = 0.25):
        super().__init__()
        assert latent_dim % n_latents == 0, "latent_dim must be divisible by n_latents"
        self.latent_dim = latent_dim
        self.n_latents = n_latents
        self.slot_dim = latent_dim // n_latents
        self.codebook_size = codebook_size
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, self.slot_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, latent_dim) -> reshape to (B, n_latents, slot_dim)
        B = z_e.size(0)
        z_slots = z_e.view(B, self.n_latents, self.slot_dim)

        # Distances: ||z - e||^2 expanded
        flat = z_slots.reshape(B * self.n_latents, self.slot_dim)
        # (BN, codebook)
        d = (
            (flat ** 2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + (self.codebook.weight ** 2).sum(dim=1)[None]
        )
        idx = d.argmin(dim=1)                                 # (BN,)
        z_q = self.codebook(idx).view(B, self.n_latents, self.slot_dim)

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_slots.detach())
        commit_loss = F.mse_loss(z_slots, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss

        # Straight-through: pass gradients through z_e
        z_q_st = z_slots + (z_q - z_slots).detach()
        return z_q_st.reshape(B, self.latent_dim), loss, idx.view(B, self.n_latents)


@MODELS.register("vqvae")
class VQVAE(RepresentationModel):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        latent_dim: int = 64,
        codebook_size: int = 256,
        n_latents: int = 4,
        commitment_beta: float = 0.25,
        decoder_hidden: int = 256,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_kind = action_kind

        self.history = HistoryEncoder(
            state_dim=state_dim, action_dim=action_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            history_k=history_k, permutation_invariant=True, dropout=dropout,
            action_kind=action_kind, n_actions=n_actions,
        )
        self.to_z = nn.Linear(d_model, latent_dim)
        self.vq = VectorQuantizer(
            latent_dim=latent_dim, n_latents=n_latents,
            codebook_size=codebook_size, beta=commitment_beta,
        )

        self.state_embed = MLP([state_dim, d_model, d_model])
        self.decoder_body = MLP(
            [latent_dim + d_model, decoder_hidden, decoder_hidden, decoder_hidden]
        )
        self.action_head = ActionHead(
            hidden_dim=decoder_hidden, action_dim=action_dim,
            action_kind=action_kind, n_actions=n_actions,
        )

    def _encode(self, past_states, past_actions):
        h = self.history(past_states, past_actions)
        z_e = self.to_z(h)
        z_q, vq_loss, idx = self.vq(z_e)
        return z_q, vq_loss, idx

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past_s = batch["past_states"]
        past_a = batch["past_actions"]
        s_t = batch["current_state"]
        a_t = batch["next_action"]

        z_q, vq_loss, idx = self._encode(past_s, past_a)
        cond = self.state_embed(s_t)
        h = self.decoder_body(torch.cat([z_q, cond], dim=-1))
        out = self.action_head(h)

        recon = self.action_head.loss(out, a_t)
        loss = recon + vq_loss
        return {"loss": loss, "recon": recon.detach(), "vq": vq_loss.detach(), "pred": out}

    @torch.no_grad()
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_q, _, _ = self._encode(batch["past_states"], batch["past_actions"])
        return z_q

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_q, _, _ = self._encode(batch["past_states"], batch["past_actions"])
        cond = self.state_embed(batch["current_state"])
        h = self.decoder_body(torch.cat([z_q, cond], dim=-1))
        return self.action_head.predict(self.action_head(h))
