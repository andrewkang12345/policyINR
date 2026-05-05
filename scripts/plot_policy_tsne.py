"""Plot t-SNE of learned policy representations for selected datasets,
colored by ground-truth policy_id.

Uses `no_shift` seed-0 checkpoints for each (data, model) cell. Embeddings
are extracted on the train store (bag-of-pairs view) using the existing
`_per_episode_representations` helper.
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.runner import _per_episode_representations
from train.main import _build_base_store, _policies_from_cfg, _device
from data import build_experiment_loaders
from data.shifts import SHIFTS
from scripts.reevaluate_probe_database import (
    _build_model_for_run,
    _load_model_checkpoint,
)

PLOTS_DIR = ROOT / "outputs" / "PLOTS" / "policyClusters"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# (subdir, data_label_for_filename, run_dir_root, run_dir_prefix)
RUN_GROUPS: list[tuple[str, str, Path, str]] = [
    ("hopper", "hopper", ROOT / "outputs" / "full_suite", "minari_hopper"),
    (
        "hopper",
        "hopper10x",
        ROOT / "outputs" / "suites" / "action_resampled_v4_hopper10x" / "runs" / "custom_mujoco_action_resampled_v5",
        "custom_mujoco_action_resampled_v5_hopper",
    ),
    (
        "hopper",
        "hopper20x",
        ROOT / "outputs" / "suites" / "action_resampled_v4_hopper20x" / "runs" / "custom_mujoco_action_resampled_v4_hopper20x",
        "custom_mujoco_action_resampled_v4_hopper20x",
    ),
    (
        "dmlab",
        "dmlab_full",
        ROOT / "outputs" / "full_suite_new_datasets",
        "dmlab_seekavoid_full",
    ),
    (
        "dmlab",
        "dmlab_sa16",
        ROOT / "outputs" / "full_suite_new_datasets",
        "dmlab_seekavoid_sa16",
    ),
    (
        "lichess",
        "lichess_full",
        ROOT / "outputs" / "full_suite_new_datasets",
        "lichess_top3_full",
    ),
    (
        "lichess",
        "lichess_sa16",
        ROOT / "outputs" / "full_suite_new_datasets",
        "lichess_top3_sa16",
    ),
    (
        "lichess",
        "lichess_full_2Xepisode",
        ROOT / "outputs" / "lichess_full_2Xepisode",
        "lichess_full_2Xepisode",
    ),
]
MODELS = [
    "cvae",
    "inr_transformer_history_conditioned",
    "inr_diffusion_history_conditioned",
    "inr_transformer_fitted_latent",
]
POLICY_COLORS = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#ffcc00",  # yellow
}
DEFAULT_POLICY_NAMES = {0: "simple", 1: "medium", 2: "expert"}


def _policy_names_from_store(store, cfg) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for meta in store.meta:
        pid = int(meta.policy_id)
        if pid in names:
            continue
        player = str(meta.extras.get("player", "")).strip()
        policy = str(meta.extras.get("policy", "")).strip()
        if player:
            names[pid] = player
        elif policy:
            names[pid] = policy
    if names:
        return names
    if cfg.data.kind == "lichess":
        players = [str(p) for p in cfg.data.get("players", [])]
        return {pid: name for pid, name in enumerate(players)}
    return DEFAULT_POLICY_NAMES


def _extract(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]] | None:
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "best.pt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        return None
    cfg = OmegaConf.load(cfg_path)
    if str(cfg.data.get("kind", "")) == "lichess":
        pgn_dir = Path(str(cfg.data.pgn_dir)).expanduser()
        if not pgn_dir.exists():
            candidates = []
            if os.environ.get("INR_LICHESS_CACHE"):
                candidates.append(Path(os.environ["INR_LICHESS_CACHE"]).expanduser() / "pgn")
            candidates.append(Path.home() / ".cache" / "INR" / "lichess" / "pgn")
            candidates.append(ROOT / ".cache" / "lichess" / "pgn")
            for candidate in candidates:
                if candidate.exists():
                    cfg.data.pgn_dir = str(candidate)
                    break

    base_store = _build_base_store(cfg.data)
    policies = _policies_from_cfg(cfg.experiment.policies)
    shift_kwargs = dict(ood_fraction=float(cfg.shift.ood_fraction), seed=int(cfg.seed))
    if "min_per_partition" in cfg.shift:
        shift_kwargs["min_per_partition"] = int(cfg.shift.min_per_partition)
    behavior_unit = str(cfg.model.get("behavior_unit", "episode"))
    unit_window_size = int(cfg.model.get("unit_window_size", cfg.train.history_k))
    uul = bool(cfg.model.get("use_unit_latents", False))
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
        seed=int(cfg.seed),
        behavior_unit=behavior_unit,
        unit_window_size=unit_window_size,
        use_unit_latents=uul,
    )

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name", None)
    kind = model_kwargs.pop("kind")
    model_kwargs.pop("shuffle_history_train", None)
    use_unit_latents = bool(model_kwargs.pop("use_unit_latents", False))
    model_kwargs.pop("behavior_unit", None)
    model_kwargs.pop("unit_window_size", None)
    # Stale config fields no longer accepted by current model __init__s.
    for stale in ("z_bridge", "use_z_bridge", "z_bridge_hidden",
                  "bridge_init_scale"):
        model_kwargs.pop(stale, None)
    model_kwargs.update(
        state_dim=loaders["state_dim"],
        action_dim=loaders["action_dim"],
        history_k=int(cfg.train.history_k),
        action_kind=loaders.get("action_kind", "continuous"),
        n_actions=loaders.get("n_actions", None),
    )
    if use_unit_latents:
        model_kwargs["n_train_units"] = int(loaders.get("n_train_units", 0))
        model_kwargs["behavior_unit"] = str(loaders.get("behavior_unit", behavior_unit))
    model = _build_model_for_run(kind, model_kwargs, ckpt_path)
    device = _device(cfg)
    try:
        _load_model_checkpoint(model, ckpt_path, device)
    except RuntimeError as exc:
        # Old checkpoint — retry non-strict and continue with the partial load
        # so we can still visualize the learned representation.
        import torch as _t
        state = _t.load(ckpt_path, map_location=device, weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[tsne]   loaded non-strict; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device)
    model.eval()

    # Extract reps over the *base* store (all policies) rather than the
    # experiment's train/test split, so the t-SNE shows how well the
    # learned representation separates every ground-truth policy — not
    # just the ones included in no_shift's partition.
    #
    # Tag each episode ID / OOD before extraction. Custom-mujoco v4/v5
    # carry a definitive `predefined_split` in extras; plain minari has
    # no built-in tag, so we apply the same shared_region shift the
    # training pipeline uses and take its per-episode is_ood. Either way
    # we stash the result into meta.is_ood so _per_episode_representations
    # can return it parallel to embs/pids.
    policy_names = _policy_names_from_store(base_store, cfg)

    has_pred_split = any(
        str(m.extras.get("predefined_split", "")).upper() in ("ID", "OOD")
        for m in base_store.meta
    )
    if has_pred_split:
        for m in base_store.meta:
            m.is_ood = str(m.extras.get("predefined_split", "ID")).upper() == "OOD"
    else:
        shift_fn = SHIFTS.get(str(cfg.shift.kind))
        is_ood_list = shift_fn(base_store, **shift_kwargs)
        for m, flag in zip(base_store.meta, is_ood_list):
            m.is_ood = bool(flag)

    embs, pids, oods = _per_episode_representations(
        model, base_store,
        history_k=int(loaders.get("history_k", 16)),
        per_episode_samples=int(cfg.eval.per_episode_samples),
        state_mean=loaders["state_mean"], state_std=loaders["state_std"],
        action_mean=loaders["action_mean"], action_std=loaders["action_std"],
        shuffle_history=True, device=device,
        behavior_unit=loaders.get("behavior_unit", "episode"),
        unit_window_size=loaders.get("unit_window_size", 0),
        known_unit_map=loaders.get("known_unit_map"),
    )
    splits = np.where(oods.astype(bool), "OOD", "ID")
    return embs, pids, splits, policy_names


def _plot_one(subdir: str, data_label: str, model: str, split: str,
              embs: np.ndarray, pids: np.ndarray, policy_names: Dict[int, str]):
    if embs.shape[0] < 4:
        print(f"[tsne] {data_label}/{model}/{split}: only {embs.shape[0]} embs, skipping", flush=True)
        return
    perplexity = max(5.0, min(30.0, embs.shape[0] / 4.0))
    ts = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
    xy = ts.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for pid in sorted(set(int(p) for p in pids)):
        mask = pids == pid
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=28, alpha=0.75,
            c=POLICY_COLORS.get(pid, "#888888"),
            label=f"{policy_names.get(pid, f'pid{pid}')}",
            edgecolors="none",
        )
    ax.set_title(f"{data_label} — {model} [{split}]\n(t-SNE of policy repr, n={embs.shape[0]})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_dir = PLOTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tsne_{data_label}_{model}_{split}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[tsne] wrote {out}  (policies={sorted(set(int(p) for p in pids))})", flush=True)


def plot_tsne(subdir: str, data_label: str, model: str, embs: np.ndarray,
              pids: np.ndarray, splits: np.ndarray, policy_names: Dict[int, str]):
    for split in ("ID", "OOD"):
        mask = splits == split
        _plot_one(subdir, data_label, model, split, embs[mask], pids[mask], policy_names)


def main():
    for subdir, data_label, runs_root, data_cfg in RUN_GROUPS:
        for model in MODELS:
            run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
            if not run_dir.exists():
                print(f"[tsne] SKIP {data_label}/{model}: {run_dir} missing", flush=True)
                continue
            try:
                result = _extract(run_dir)
            except Exception as exc:
                print(f"[tsne] FAIL {data_label}/{model}: {exc}", flush=True)
                continue
            if result is None:
                print(f"[tsne] SKIP {data_label}/{model}: no cfg/ckpt", flush=True)
                continue
            embs, pids, splits, policy_names = result
            plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names)


if __name__ == "__main__":
    main()
