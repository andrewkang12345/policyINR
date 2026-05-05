"""Visualize synthetic GRF policy landscapes and architecture extrapolations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.shifts import assign_state_shift
from data.splits import PolicySpec, build_experiment_loaders
from data.synthetic import ACTION_POLICIES, build_synthetic_store
from models import build_model
from models.base import HistoryEncoder, MLP


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite-root",
        type=Path,
        default=ROOT / "outputs" / "suites" / "state_resampled_v1",
    )
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--dataset-name", type=str, default="synthetic_grf")
    ap.add_argument("--grid-size", type=int, default=72)
    ap.add_argument("--n-history-anchors", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    return ap.parse_args()


def _build_base_store(data_cfg):
    return build_synthetic_store(
        n_policies=int(data_cfg.n_policies),
        episodes_per_policy=int(data_cfg.episodes_per_policy),
        episode_length=int(data_cfg.episode_length),
        state_dim=int(data_cfg.state_dim),
        action_dim=int(data_cfg.action_dim),
        state_generator=str(data_cfg.state_generator),
        action_policy=str(data_cfg.action_policy),
        noise_std=float(data_cfg.noise_std),
        seed=0,
        state_gen_kwargs=OmegaConf.to_container(data_cfg.state_gen_kwargs, resolve=True),
        action_gen_kwargs=OmegaConf.to_container(data_cfg.action_gen_kwargs, resolve=True),
    )


def _rebuild_clean_policies(data_cfg):
    rng = np.random.default_rng(0)
    make_policy = ACTION_POLICIES.get(str(data_cfg.action_policy))
    kwargs = OmegaConf.to_container(data_cfg.action_gen_kwargs, resolve=True)
    policies = []
    for _ in range(int(data_cfg.n_policies)):
        policy_seed = int(rng.integers(0, 2**31 - 1))
        policies.append(
            make_policy(
                policy_seed=policy_seed,
                state_dim=int(data_cfg.state_dim),
                action_dim=int(data_cfg.action_dim),
                **kwargs,
            )
        )
    return policies


def _fit_linear_pca(x: np.ndarray, n_components: int):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    comps = vh[:n_components].astype(np.float32)
    return mean.astype(np.float32), comps


def _project(x: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (x - mean) @ comps.T


def _inverse_project(z: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return z @ comps + mean


def _make_grid(states_2d: np.ndarray, grid_size: int):
    mins = states_2d.min(axis=0)
    maxs = states_2d.max(axis=0)
    pad = 0.08 * (maxs - mins + 1e-6)
    xs = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], grid_size, dtype=np.float32)
    ys = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return xx, yy, grid_2d


def _build_norms_from_run(run_dir: Path):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    base_store = _build_base_store(cfg.data)
    policies = [PolicySpec(pid=int(p.pid), train=str(p.train), test=str(p.test)) for p in cfg.experiment.policies]
    shift_kwargs = {
        "ood_fraction": float(cfg.shift.ood_fraction),
        "seed": int(cfg.seed),
        "min_per_partition": int(cfg.shift.min_per_partition),
    }
    loaders = build_experiment_loaders(
        base_store,
        policies=policies,
        history_k=int(cfg.train.history_k),
        shift_kind=str(cfg.shift.kind),
        shift_kwargs=shift_kwargs,
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        num_workers=0,
        shuffle_history_train=bool(cfg.model.shuffle_history_train),
        behavior_unit=str(cfg.model.get("behavior_unit", "episode")),
        unit_window_size=int(cfg.model.get("unit_window_size", cfg.train.history_k)),
        use_unit_latents=bool(cfg.model.get("use_unit_latents", False)),
        seed=int(cfg.seed),
    )
    return cfg, loaders


def _load_model(run_dir: Path, cfg, loaders, device: str):
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state_dict = _remap_legacy_state_dict(state["model"], str(cfg.model.kind))
    if str(cfg.model.kind) == "cvae" and "decoder.net.0.weight" in state_dict:
        model = _build_legacy_cvae(cfg, loaders)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    kind = model_cfg.pop("kind")
    model_cfg.pop("shuffle_history_train", None)
    use_unit_latents = bool(model_cfg.pop("use_unit_latents", False))
    model_cfg.pop("behavior_unit", None)
    model_cfg.pop("unit_window_size", None)
    model_cfg.update(
        state_dim=loaders["state_dim"],
        action_dim=loaders["action_dim"],
        history_k=int(cfg.train.history_k),
        action_kind=loaders.get("action_kind", "continuous"),
        n_actions=loaders.get("n_actions", None),
    )
    if use_unit_latents:
        model_cfg["n_train_units"] = int(loaders.get("n_train_units", 0))
    model = build_model(kind, **model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _remap_legacy_state_dict(state_dict, model_kind: str):
    out = dict(state_dict)
    if model_kind in {"inr_transformer", "inr_transformer_history_conditioned"}:
        encoder_prefixed = {}
        for key, value in list(out.items()):
            if key.startswith(("type_pair", "pos", "pair_embed.", "encoder.layers.", "latent_head.", "latent_norm.")):
                encoder_prefixed[f"encoder.{key}"] = value
                out.pop(key)
        out.update(encoder_prefixed)
    if model_kind in {
        "inr_transformer",
        "inr_transformer_history_conditioned",
        "inr_diffusion",
        "inr_diffusion_history_conditioned",
    } and "pair_embed.weight" in out:
        out["pair_embed.proj.weight"] = out.pop("pair_embed.weight")
        out["pair_embed.proj.bias"] = out.pop("pair_embed.bias")
    if model_kind in {"inr_transformer", "inr_transformer_history_conditioned"} and "encoder.pair_embed.weight" in out:
        out["encoder.pair_embed.proj.weight"] = out.pop("encoder.pair_embed.weight")
        out["encoder.pair_embed.proj.bias"] = out.pop("encoder.pair_embed.bias")
    if model_kind in {"inr_transformer", "inr_transformer_history_conditioned"} and "policy_head.out.weight" in out:
        out["policy_head.action_head.out.weight"] = out.pop("policy_head.out.weight")
        out["policy_head.action_head.out.bias"] = out.pop("policy_head.out.bias")
    return out


class _LegacyCVAE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_k: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        latent_dim: int,
        decoder_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.history = HistoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            history_k=history_k,
            permutation_invariant=True,
            dropout=dropout,
        )
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.state_embed = MLP([state_dim, d_model, d_model])
        self.decoder = MLP([latent_dim + d_model, decoder_hidden, decoder_hidden, action_dim])

    def _encode(self, past_states, past_actions):
        h = self.history(past_states, past_actions)
        return self.to_mu(h), self.to_logvar(h)

    @torch.no_grad()
    def predict_action(self, batch):
        mu, _ = self._encode(batch["past_states"], batch["past_actions"])
        cond = self.state_embed(batch["current_state"])
        return self.decoder(torch.cat([mu, cond], dim=-1))


def _build_legacy_cvae(cfg, loaders):
    return _LegacyCVAE(
        state_dim=loaders["state_dim"],
        action_dim=loaders["action_dim"],
        history_k=int(cfg.train.history_k),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        latent_dim=int(cfg.model.latent_dim),
        decoder_hidden=int(cfg.model.decoder_hidden),
        dropout=float(cfg.model.dropout),
    )


def _select_anchor_histories(shifted_store, policy_id: int, split_name: str, history_k: int, n_anchors: int, shuffle_history: bool):
    idxs = [i for i, m in enumerate(shifted_store.meta) if m.policy_id == policy_id and ((not m.is_ood) if split_name == "ID" else m.is_ood)]
    if not idxs:
        raise RuntimeError(f"No {split_name} episodes found for policy {policy_id}")
    idxs = idxs[: min(n_anchors, len(idxs))]
    anchors = []
    for ei in idxs:
        s = shifted_store.states[ei]
        a = shifted_store.actions[ei]
        mask = shifted_store.meta[ei].extras.get("shared_mask") if split_name == "OOD" else None
        if mask is not None and np.any(mask):
            valid = np.flatnonzero(mask)
        else:
            valid = np.arange(s.shape[0], dtype=np.int64)
        if shuffle_history:
            if valid.size >= history_k:
                pick = np.linspace(0, valid.size - 1, history_k).round().astype(np.int64)
                idx = valid[pick]
            else:
                idx = np.pad(valid, (history_k - valid.size, 0), mode="edge")
        else:
            if valid.size >= history_k:
                idx = valid[-history_k:]
            else:
                idx = np.pad(valid, (history_k - valid.size, 0), mode="edge")
        anchors.append((s[idx].astype(np.float32), a[idx].astype(np.float32), ei))
    return anchors


def _predict_field(model, anchors, grid_states, loaders, device: str, batch_size: int):
    s_mean = loaders["state_mean"].astype(np.float32)
    s_std = loaders["state_std"].astype(np.float32)
    a_mean = loaders["action_mean"].astype(np.float32)
    a_std = loaders["action_std"].astype(np.float32)
    preds = []
    use_unit_latents = bool(loaders.get("use_unit_latents", False))
    with torch.no_grad():
        for past_s_np, past_a_np, _anchor_episode_idx in anchors:
            past_s = ((past_s_np - s_mean) / s_std).astype(np.float32)
            past_a = ((past_a_np - a_mean) / a_std).astype(np.float32)
            per_anchor = []
            for start in range(0, grid_states.shape[0], batch_size):
                cur = grid_states[start:start + batch_size]
                cur_norm = ((cur - s_mean) / s_std).astype(np.float32)
                bsz = cur.shape[0]
                batch = {
                    "past_states": torch.from_numpy(np.repeat(past_s[None], bsz, axis=0)).to(device),
                    "past_actions": torch.from_numpy(np.repeat(past_a[None], bsz, axis=0)).to(device),
                    "current_state": torch.from_numpy(cur_norm).to(device),
                }
                if use_unit_latents:
                    batch["unit_id"] = torch.full((bsz,), -1, dtype=torch.long, device=device)
                    batch["has_unit_latent"] = torch.zeros((bsz,), dtype=torch.long, device=device)
                out = model.predict_action(batch).detach().cpu().numpy().astype(np.float32)
                out = out * a_std + a_mean
                per_anchor.append(out)
            preds.append(np.concatenate(per_anchor, axis=0))
    return np.mean(np.stack(preds, axis=0), axis=0)


def _draw_scalar_field(ax, xx, yy, scalar, vmin, vmax, cmap="coolwarm", alpha=1.0):
    im = ax.imshow(
        scalar.reshape(xx.shape),
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        aspect="auto",
    )
    return im


def _point_colors_from_gt(points_2d: np.ndarray, xx, yy, scalar_grid: np.ndarray, vmin: float, vmax: float, cmap_name="coolwarm"):
    nx = xx.shape[1]
    ny = xx.shape[0]
    x0, x1 = float(xx.min()), float(xx.max())
    y0, y1 = float(yy.min()), float(yy.max())
    x_idx = np.clip(np.round((points_2d[:, 0] - x0) / max(x1 - x0, 1e-6) * (nx - 1)).astype(np.int64), 0, nx - 1)
    y_idx = np.clip(np.round((points_2d[:, 1] - y0) / max(y1 - y0, 1e-6) * (ny - 1)).astype(np.int64), 0, ny - 1)
    vals = scalar_grid.reshape(yy.shape)[y_idx, x_idx]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm(vals))


def _format_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = _parse_args()
    dataset_name = str(args.dataset_name)
    out_root = args.out_root or (args.suite_root / "analysis" / dataset_name)
    out_root.mkdir(parents=True, exist_ok=True)

    synth_run = args.suite_root / f"{dataset_name}__cvae__new_policy__s0"
    if not synth_run.exists():
        synth_run = args.suite_root / "runs" / "synthetic" / f"{dataset_name}__cvae__new_policy__s0"
    cfg0 = OmegaConf.load(synth_run / "config.yaml")
    base_store = _build_base_store(cfg0.data)
    shifted_store, _ = assign_state_shift(
        base_store,
        kind=str(cfg0.shift.kind),
        ood_fraction=float(cfg0.shift.ood_fraction),
        seed=int(cfg0.seed),
        min_per_partition=int(cfg0.shift.min_per_partition),
    )
    clean_policies = _rebuild_clean_policies(cfg0.data)

    all_states = np.concatenate(base_store.states, axis=0).astype(np.float32)
    state_mean, state_comps = _fit_linear_pca(all_states, n_components=2)
    proj_states = _project(all_states, state_mean, state_comps)
    xx, yy, grid_2d = _make_grid(proj_states, args.grid_size)
    grid_states = _inverse_project(grid_2d, state_mean, state_comps).astype(np.float32)

    clean_actions_all = []
    for pid, policy in enumerate(clean_policies):
        for s in base_store.states:
            clean_actions_all.append(policy(s))
    clean_actions_all = np.concatenate(clean_actions_all, axis=0).astype(np.float32)
    action_mean, action_pc = _fit_linear_pca(clean_actions_all, n_components=1)
    action_pc = action_pc[0]

    def action_scalar(actions: np.ndarray) -> np.ndarray:
        return ((actions - action_mean) @ action_pc[:, None]).reshape(-1)

    gt_scalars = {}
    pred_scalars = {}
    run_specs = {
        "CVAE / ID": args.suite_root / f"{dataset_name}__cvae__new_policy__s0",
        "CVAE / OOD": args.suite_root / f"{dataset_name}__cvae__novel_generalization__s0",
        "Transformer / ID": args.suite_root / f"{dataset_name}__inr_transformer_history_conditioned__new_policy__s0",
        "Transformer / OOD": args.suite_root / f"{dataset_name}__inr_transformer_history_conditioned__novel_generalization__s0",
        "Diffusion / ID": args.suite_root / f"{dataset_name}__inr_diffusion_history_conditioned__new_policy__s0",
        "Diffusion / OOD": args.suite_root / f"{dataset_name}__inr_diffusion_history_conditioned__novel_generalization__s0",
        "Transformer FL / ID": args.suite_root / f"{dataset_name}__inr_transformer_fitted_latent__new_policy__s0",
        "Transformer FL / OOD": args.suite_root / f"{dataset_name}__inr_transformer_fitted_latent__novel_generalization__s0",
    }
    run_specs = {
        label: (path if path.exists() else args.suite_root / "runs" / "synthetic" / path.name)
        for label, path in run_specs.items()
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for label, run_dir in run_specs.items():
        cfg, loaders = _build_norms_from_run(run_dir)
        model = _load_model(run_dir, cfg, loaders, device)
        loaded[label] = (cfg, loaders, model)

    scalar_values = []
    for pid, policy in enumerate(clean_policies):
        gt_actions = policy(grid_states).astype(np.float32)
        gt_scalars[pid] = action_scalar(gt_actions)
        scalar_values.append(gt_scalars[pid])
        for label, (cfg, loaders, model) in loaded.items():
            split_name = "ID" if label.endswith("/ ID") else "OOD"
            anchors = _select_anchor_histories(
                shifted_store,
                policy_id=pid,
                split_name=split_name,
                history_k=int(cfg.train.history_k),
                n_anchors=args.n_history_anchors,
                shuffle_history=bool(cfg.model.shuffle_history_train),
            )
            pred_actions = _predict_field(
                model,
                anchors,
                grid_states,
                loaders,
                device=device,
                batch_size=args.batch_size,
            )
            pred_scalars[(pid, label)] = action_scalar(pred_actions)
            scalar_values.append(pred_scalars[(pid, label)])

    scalar_stack = np.concatenate(scalar_values, axis=0)
    vlim = float(np.quantile(np.abs(scalar_stack), 0.99))
    vmin, vmax = -vlim, vlim

    title_order = [
        "CVAE / ID",
        "CVAE / OOD",
        "Transformer / ID",
        "Transformer / OOD",
        "Diffusion / ID",
        "Diffusion / OOD",
    ]
    recent_inr_title_order = [
        "Transformer HC / ID",
        "Transformer HC / OOD",
        "Diffusion HC / ID",
        "Diffusion HC / OOD",
        "Transformer FL / ID",
        "Transformer FL / OOD",
    ]

    for pid, policy in enumerate(clean_policies):
        fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
        policy_states = np.concatenate(
            [s for i, s in enumerate(base_store.states) if base_store.meta[i].policy_id == pid],
            axis=0,
        ).astype(np.float32)
        policy_proj = _project(policy_states, state_mean, state_comps)

        split_points = {}
        for split_name in ("ID", "OOD"):
            pts = []
            for ei, meta in enumerate(shifted_store.meta):
                if meta.policy_id != pid:
                    continue
                if split_name == "ID" and meta.is_ood:
                    continue
                if split_name == "OOD" and not meta.is_ood:
                    continue
                pts.append(_project(shifted_store.states[ei].astype(np.float32), state_mean, state_comps))
            split_points[split_name] = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), dtype=np.float32)

        ax = axes[0, 0]
        im = _draw_scalar_field(ax, xx, yy, gt_scalars[pid], vmin, vmax, cmap="coolwarm")
        ax.set_title(f"Policy {pid}: Ground Truth")
        _format_axis(ax)

        for col, split_name in enumerate(("ID", "OOD"), start=1):
            ax = axes[0, col]
            pts = split_points[split_name]
            marker = "o" if split_name == "ID" else "^"
            pt_colors = _point_colors_from_gt(pts, xx, yy, gt_scalars[pid], vmin, vmax)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c=pt_colors,
                alpha=0.9,
                linewidths=0,
                marker=marker,
            )
            ax.set_title(f"{split_name} Points")
            _format_axis(ax)

        for panel_idx, label in enumerate(title_order):
            row = 1 + panel_idx // 3
            col = panel_idx % 3
            ax = axes[row, col]
            _draw_scalar_field(ax, xx, yy, pred_scalars[(pid, label)], vmin, vmax, cmap="coolwarm")
            split_name = "ID" if label.endswith("/ ID") else "OOD"
            pts = split_points[split_name]
            marker = "o" if split_name == "ID" else "^"
            pt_colors = _point_colors_from_gt(pts, xx, yy, gt_scalars[pid], vmin, vmax)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=6,
                c=pt_colors,
                alpha=0.55,
                linewidths=0,
                marker=marker,
            )
            prefix = label.split(" / ")[0]
            ax.set_title(f"{prefix}: Extrapolated from {split_name}")
            _format_axis(ax)

        cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
        cbar.set_label("Action PC1")
        out_path = out_root / f"policy_{pid}_landscapes.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
        ax = axes[0, 0]
        im = _draw_scalar_field(ax, xx, yy, gt_scalars[pid], vmin, vmax, cmap="coolwarm")
        ax.set_title(f"Policy {pid}: Ground Truth")
        _format_axis(ax)

        for col, split_name in enumerate(("ID", "OOD"), start=1):
            ax = axes[0, col]
            pts = split_points[split_name]
            marker = "o" if split_name == "ID" else "^"
            pt_colors = _point_colors_from_gt(pts, xx, yy, gt_scalars[pid], vmin, vmax)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c=pt_colors,
                alpha=0.9,
                linewidths=0,
                marker=marker,
            )
            ax.set_title(f"{split_name} Points")
            _format_axis(ax)

        recent_label_map = {
            "Transformer HC / ID": ("Transformer / ID", "Transformer HC"),
            "Transformer HC / OOD": ("Transformer / OOD", "Transformer HC"),
            "Diffusion HC / ID": ("Diffusion / ID", "Diffusion HC"),
            "Diffusion HC / OOD": ("Diffusion / OOD", "Diffusion HC"),
            "Transformer FL / ID": ("Transformer FL / ID", "Transformer FL"),
            "Transformer FL / OOD": ("Transformer FL / OOD", "Transformer FL"),
        }
        for panel_idx, label in enumerate(recent_inr_title_order):
            row = 1 + panel_idx // 3
            col = panel_idx % 3
            ax = axes[row, col]
            source_label, prefix = recent_label_map[label]
            _draw_scalar_field(ax, xx, yy, pred_scalars[(pid, source_label)], vmin, vmax, cmap="coolwarm")
            split_name = "ID" if label.endswith("/ ID") else "OOD"
            pts = split_points[split_name]
            marker = "o" if split_name == "ID" else "^"
            pt_colors = _point_colors_from_gt(pts, xx, yy, gt_scalars[pid], vmin, vmax)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=6,
                c=pt_colors,
                alpha=0.55,
                linewidths=0,
                marker=marker,
            )
            ax.set_title(f"{prefix}: Extrapolated from {split_name}")
            _format_axis(ax)

        cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02)
        cbar.set_label("Action PC1")
        out_path = out_root / f"policy_{pid}_recent_inrs.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    summary_path = out_root / "README.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Synthetic GRF landscape plots",
                f"Output root: {out_root}",
                f"Dataset name: {dataset_name}",
                "Ground truth: clean synthetic policy action field on a 2D PCA slice of the 8D state space.",
                "ID/OOD points: actual split-assigned synthetic states for each policy under state_resampled_v1.",
                "Architecture extrapolations:",
                "  ID  -> seed-0 new_policy checkpoint for each architecture.",
                "  OOD -> seed-0 novel_generalization checkpoint for each architecture.",
                "Colors correspond to the first principal component of the 4D clean/predicted action vector.",
                "Additional *_recent_inrs.png files use the recent INR variants:",
                "  Transformer HC, Diffusion HC, Transformer FL.",
            ]
        )
        + "\n"
    )
    print(f"Wrote synthetic GRF plots to {out_root}")


if __name__ == "__main__":
    main()
