"""Fixed state featurizers for high-dimensional / symbolic observations.

Each featurizer is **deterministic and untrained**: applied once at
dataset construction to turn raw observations (images, chess
positions, pixel frames with extras) into a moderate-dim numeric
state vector that the existing `EpisodeStore` / `PolicyDataset`
pipeline can consume unchanged. Keeping this outside the model
deliberately: reproducibility, inspection, and zero GPU cost at
train time.

Provided featurizers:
  * RandomCNNFeaturizer : a seeded, frozen, small CNN → D-dim vector
                           for RGB frames after a resize to 64x64.
  * DMLabFeaturizer     : RandomCNN + concatenation of (last_action_onehot,
                           last_reward). Matches the DMLab observation
                           schema in RL Unplugged.
  * ChessPiecePlanes    : 8x8x12 piece planes flattened + metadata
                           (side to move, castling, ep square, half/full
                           move counters) → fixed-dim vector.

All featurizers are pure functions of their inputs given a fixed seed,
and their output dimension is discoverable via `.feature_dim`.
"""

from __future__ import annotations

from typing import Optional
import hashlib
import io

import numpy as np
import torch
import torch.nn as nn


def _seed_worker(seed: int):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class RandomCNNFeaturizer:
    """Seeded, frozen small CNN: RGB HxW image → D-dim vector.

    Architecture is intentionally tiny so inference is cheap on CPU.
    Weights are initialized with a fixed torch RNG so the same random
    projection applies across runs.
    """

    def __init__(self, in_ch: int = 3, img_size: int = 64, feature_dim: int = 128, seed: int = 0,
                 device: str = "cpu"):
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.device = device
        # Seed torch globally briefly so the CNN is deterministic across runs.
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            net = nn.Sequential(
                nn.Conv2d(in_ch, 16, 5, stride=2), nn.GELU(),
                nn.Conv2d(16, 32, 5, stride=2), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, feature_dim),
            )
        finally:
            torch.random.set_rng_state(state)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device).eval()

    @torch.no_grad()
    def __call__(self, frame_hw_c: np.ndarray) -> np.ndarray:
        """frame_hw_c: (H, W, C) uint8 array or batched (N, H, W, C). Returns (N, D) float32."""
        frame_hw_c = np.ascontiguousarray(np.asarray(frame_hw_c, dtype=np.uint8))
        batched = frame_hw_c.ndim == 4
        if not batched:
            frame_hw_c = frame_hw_c[None]
        x = torch.as_tensor(frame_hw_c, dtype=torch.float32, device=self.device)
        x = x.permute(0, 3, 1, 2) / 255.0  # -> (N, C, H, W) in [0,1]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size),
                                                mode="bilinear", align_corners=False)
        feats = self.net(x)
        feats = feats.cpu().numpy().astype(np.float32)
        return feats if batched else feats[0]


class DMLabFeaturizer:
    """RandomCNN on the frame + (one-hot last_action, last_reward) context.

    DMLab obs dict has fields:
      pixels      : (72, 96, 3) uint8
      last_action : int64  (0..n_actions-1)
      last_reward : float32
    We return a single concatenated vector per step.
    """
    def __init__(self, n_actions: int, cnn_feature_dim: int = 128, img_size: int = 64,
                 seed: int = 0, device: str = "cpu"):
        self.cnn = RandomCNNFeaturizer(in_ch=3, img_size=img_size,
                                        feature_dim=cnn_feature_dim, seed=seed, device=device)
        self.n_actions = int(n_actions)
        self.feature_dim = cnn_feature_dim + n_actions + 1  # +last_action one-hot +reward

    @torch.no_grad()
    def __call__(self, pixels: np.ndarray, last_action: np.ndarray, last_reward: np.ndarray) -> np.ndarray:
        """pixels (T,H,W,C) uint8; last_action (T,) int; last_reward (T,) float. -> (T, D)."""
        feats = self.cnn(pixels)
        T = pixels.shape[0]
        oh = np.zeros((T, self.n_actions), dtype=np.float32)
        la = np.asarray(last_action, dtype=np.int64).reshape(-1)
        la = np.clip(la, 0, self.n_actions - 1)
        oh[np.arange(T), la] = 1.0
        lr = np.asarray(last_reward, dtype=np.float32).reshape(T, 1)
        return np.concatenate([feats, oh, lr], axis=-1).astype(np.float32)


# ---- Chess piece-plane encoder ------------------------------------------

# 12 piece planes (pawn/knight/bishop/rook/queen/king × white/black)
# + metadata: side to move (1), castling (4), ep file (8), halfmove (1), fullmove (1)
CHESS_FEATURE_DIM = 8 * 8 * 12 + 1 + 4 + 8 + 1 + 1  # = 783


def chess_board_to_vector(board) -> np.ndarray:
    """python-chess Board -> fixed-dim float32 feature vector."""
    import chess
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for plane_idx, pt in enumerate(piece_types):
        for sq in board.pieces(pt, chess.WHITE):
            r, f = divmod(sq, 8)
            planes[plane_idx, 7 - r, f] = 1.0
        for sq in board.pieces(pt, chess.BLACK):
            r, f = divmod(sq, 8)
            planes[plane_idx + 6, 7 - r, f] = 1.0
    side = np.array([1.0 if board.turn == chess.WHITE else 0.0], dtype=np.float32)
    castling = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=np.float32)
    ep = np.zeros(8, dtype=np.float32)
    if board.ep_square is not None:
        ep[board.ep_square % 8] = 1.0
    half = np.array([float(board.halfmove_clock) / 50.0], dtype=np.float32)
    full = np.array([float(board.fullmove_number) / 200.0], dtype=np.float32)
    return np.concatenate([planes.reshape(-1), side, castling, ep, half, full]).astype(np.float32)
