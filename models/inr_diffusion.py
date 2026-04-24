"""Diffusion-head INR with explicit history-conditioned naming."""

from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS
from .base import RepresentationModel, PairEmbed


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000.0) / max(1, half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class CondDenoiser(nn.Module):
    """Conditional epsilon-predictor eps_hat(a_t, t | s, z)."""

    def __init__(self, state_dim, action_dim, latent_dim, hidden=256, n_layers=3):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.cond = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(action_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, a_noisy, t, s, z):
        h = self.in_proj(a_noisy) + self.t_emb(t) + self.cond(torch.cat([s, z], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


@MODELS.register("inr_diffusion_history_conditioned")
@MODELS.register("inr_diffusion")
class INRDiffusionHistoryConditioned(RepresentationModel):
    """History-conditioned diffusion-head INR."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 64,
        denoise_hidden: int = 256,
        denoise_layers: int = 3,
        n_diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        n_sample_steps: int = 10,
        dropout: float = 0.0,
        action_kind: str = "continuous",
        n_actions: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_k = history_k
        self.n_diffusion_steps = n_diffusion_steps
        self.n_sample_steps = n_sample_steps
        self.action_kind = action_kind
        self.diffusion_dim = n_actions if action_kind == "discrete" else action_dim

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
        self.x0_clip = 10.0

        self.denoiser = CondDenoiser(
            state_dim=state_dim,
            action_dim=self.diffusion_dim,
            latent_dim=latent_dim,
            hidden=denoise_hidden,
            n_layers=denoise_layers,
        )

        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.action_dim = action_dim

    def _encode(self, past_s, past_a):
        pair_tok = self.pair_embed(past_s, past_a) + self.type_pair
        pair_tok = pair_tok + self.pos[: pair_tok.size(1)][None]
        z = self.latent_head(self.encoder(pair_tok).mean(dim=1))
        return self.latent_norm(z)

    def _q_sample(self, a0, t, noise):
        return self.sqrt_alphas_cumprod[t].unsqueeze(-1) * a0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_t = batch["current_state"]
        a_t = batch["next_action"]
        z = self._encode(batch["past_states"], batch["past_actions"])
        if self.action_kind == "discrete":
            a0 = F.one_hot(a_t.long(), num_classes=self.diffusion_dim).float()
        else:
            a0 = a_t
        B = a0.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=a0.device)
        noise = torch.randn_like(a0)
        a_noisy = self._q_sample(a0, t, noise)
        eps_hat = self.denoiser(a_noisy, t, s_t, z)
        loss = F.mse_loss(eps_hat, noise)
        return {"loss": loss, "pred_eps": eps_hat}

    @torch.no_grad()
    def extract_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(batch["past_states"], batch["past_actions"])

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s_t = batch["current_state"]
        z = self._encode(batch["past_states"], batch["past_actions"])
        B = s_t.shape[0]
        device = s_t.device

        step_idx = torch.linspace(0, self.n_diffusion_steps - 1, self.n_sample_steps, device=device).long().flip(0)
        a = torch.randn(B, self.diffusion_dim, device=device)
        for i, ti in enumerate(step_idx):
            t_batch = ti.expand(B)
            eps_hat = self.denoiser(a, t_batch, s_t, z)
            a_cum = self.alphas_cumprod[ti]
            a0 = (a - (1 - a_cum).sqrt() * eps_hat) / a_cum.sqrt().clamp_min(1e-8)
            a0 = a0.clamp(-self.x0_clip, self.x0_clip)
            if i == len(step_idx) - 1:
                a = a0
            else:
                tn = step_idx[i + 1]
                a_cum_next = self.alphas_cumprod[tn]
                a = a_cum_next.sqrt() * a0 + (1 - a_cum_next).sqrt() * eps_hat
        a = a.clamp(-self.x0_clip, self.x0_clip)
        if self.action_kind == "discrete":
            return a.argmax(dim=-1)
        return a


# Backward-compatible class alias for imports.
INRDiffusion = INRDiffusionHistoryConditioned
