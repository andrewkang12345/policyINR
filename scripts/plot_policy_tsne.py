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
import hashlib
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple
import argparse

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
from data.splits import PolicySpec
from data.shifts import SHIFTS
from data.lichess import _game_to_positions_and_moves, _iter_games
from scripts.reevaluate_probe_database import (
    _build_model_for_run,
    _load_model_checkpoint,
    _remap_unavailable_cache_dirs,
)
from scripts.append_fastf1_all_player_metrics import (
    _expanded_policies as _expanded_fastf1_policies,
    _fastf1_cache_path,
    _model_from_run as _fastf1_model_from_run,
    _valid_drivers as _valid_fastf1_drivers,
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
        ROOT / "outputs" / "lichess" / "baseline_top3",
        "lichess_top3_full",
    ),
    (
        "lichess",
        "lichess_sa16",
        ROOT / "outputs" / "lichess" / "baseline_top3",
        "lichess_top3_sa16",
    ),
    (
        "lichess_n200",
        "lichess_full",
        ROOT / "outputs" / "lichess" / "baseline_top3",
        "lichess_top3_full",
    ),
    (
        "lichess_n200",
        "lichess_sa16",
        ROOT / "outputs" / "lichess" / "baseline_top3",
        "lichess_top3_sa16",
    ),
    (
        "lichess",
        "lichess_full_2Xepisode",
        ROOT / "outputs" / "lichess_full_2Xepisode",
        "lichess_full_2Xepisode",
    ),
    (
        "droid",
        "droid_min300_hk300_remove",
        ROOT / "outputs" / "droid" / "balanced_min300_remove_hk300",
        "droid_lowdim_full_balanced_min300_remove",
    ),
]
MODELS = [
    "cvae",
    "inr_transformer_history_conditioned",
    "inr_diffusion_history_conditioned",
    "inr_transformer_fitted_latent",
]
DROID_MODELS = MODELS + ["inr_transformer_infer_latent_maml"]
POLICY_COLORS = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#ffcc00",  # yellow
}
DEFAULT_POLICY_NAMES = {0: "simple", 1: "medium", 2: "expert"}
FASTF1_ALL_PLAYER_GROUP = (
    "f1",
    "f1_all_players",
    ROOT / "outputs" / "fastf1" / "uncapped_full_suite",
    "fastf1_stint_full_uncapped",
)
FASTF1_5PEOPLE_GROUP = (
    "fastf1_5people",
    "fastf1_5people",
    ROOT / "outputs" / "fastf1" / "uncapped_full_suite",
    "fastf1_stint_full_uncapped",
)
FASTF1_3PEOPLE_GROUP = (
    "fastf1_3people",
    "fastf1_3people",
    ROOT / "outputs" / "fastf1" / "uncapped_full_suite",
    "fastf1_stint_full_uncapped",
)
FASTF1_3PEOPLE_DRIVERS = ("VER", "ZHO", "PER")
FASTF1_5PEOPLE_DRIVERS = ("VER", "ZHO", "PER", "BOT", "HUL")
FASTF1_DRIVER_LABELS = {
    "VER": "Max Verstappen",
    "ZHO": "Zhou Guanyu",
    "PER": "Sergio Perez",
    "BOT": "Valtteri Bottas",
    "HUL": "Nico Hulkenberg",
}


def _policy_color(pid: int) -> str:
    if pid in POLICY_COLORS:
        return POLICY_COLORS[pid]
    cmap = plt.get_cmap("tab20")
    r, g, b, _ = cmap(pid % 20)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _policy_names_from_store(store, cfg) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for meta in store.meta:
        pid = int(meta.policy_id)
        if pid in names:
            continue
        player = str(meta.extras.get("player", "")).strip()
        policy = str(meta.extras.get("policy", "")).strip()
        collector = str(meta.extras.get("collector", "")).strip()
        driver = str(meta.extras.get("driver", "")).strip()
        if player:
            names[pid] = player
        elif policy:
            names[pid] = policy
        elif collector:
            names[pid] = collector
        elif driver:
            names[pid] = driver
    if names:
        return names
    if cfg.data.kind == "lichess":
        players = [str(p) for p in cfg.data.get("players", [])]
        return {pid: name for pid, name in enumerate(players)}
    if cfg.data.kind == "droid":
        collectors = [str(c) for c in cfg.data.get("collectors", [])]
        return {pid: name for pid, name in enumerate(collectors)}
    return DEFAULT_POLICY_NAMES


SCENE_TERMS = (
    "drawer", "counter", "table", "bowl", "box", "pot", "pan", "plate",
    "tray", "dish", "shelf", "sink", "stove", "cup", "basket", "cloth",
)
OBJECT_TERMS = (
    "marker", "plushie", "plushy", "plush", "toy", "object", "bottle",
    "cup", "block", "bowl", "cloth", "towel", "sponge", "can", "apple",
    "banana", "carrot", "beetroot", "tulip", "dish", "plate", "pot",
)
TASK_PATTERNS = (
    ("remove", ("remove",)),
    ("put/place", ("put", "place", "set")),
    ("move/transfer", ("move", "transfer", "bring", "take", "pick")),
    ("open", ("open",)),
    ("close", ("close",)),
    ("wipe", ("wipe",)),
    ("pour", ("pour",)),
    ("fold/unfold", ("fold", "unfold")),
    ("hang/unhang", ("hang", "unhang")),
    ("turn/switch", ("turn", "switch")),
    ("press", ("press",)),
    ("stir", ("stir",)),
    ("slide", ("slide",)),
    ("lock", ("lock",)),
    ("cover/uncover", ("cover", "uncover")),
    ("dismantle", ("dismantle", "unassemble")),
)


def _first_matching_term(text: str, terms: Sequence[str], default: str = "other") -> str:
    words = set(str(text).lower().replace("-", " ").split())
    for term in terms:
        if term in words:
            return term
    return default


def _context_label(meta, label_kind: str) -> str:
    extras = dict(getattr(meta, "extras", {}) or {})
    instruction = str(extras.get("instruction", "")).lower()
    if label_kind == "fastf1_gp":
        return str(extras.get("gp", "unknown"))
    if label_kind == "droid_task":
        words = set(instruction.replace("-", " ").split())
        for label, terms in TASK_PATTERNS:
            if any(term in words for term in terms):
                return label
        return "other"
    if label_kind == "droid_scene":
        return _first_matching_term(instruction, SCENE_TERMS)
    if label_kind == "droid_object":
        return _first_matching_term(instruction, OBJECT_TERMS)
    raise ValueError(f"unknown context label kind: {label_kind}")


def _lichess_tracked_player_outcome(game, player: str) -> str:
    result = str(game.headers.get("Result", "")).strip()
    white = str(game.headers.get("White", ""))
    black = str(game.headers.get("Black", ""))
    player_l = player.lower()
    is_white = white.lower() == player_l
    is_black = black.lower() == player_l
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "won" if is_white else "lost" if is_black else "unknown"
    if result == "0-1":
        return "won" if is_black else "lost" if is_white else "unknown"
    return "unknown"


def _lichess_winner_name(game) -> str:
    result = str(game.headers.get("Result", "")).strip()
    if result == "1-0":
        return str(game.headers.get("White", "White"))
    if result == "0-1":
        return str(game.headers.get("Black", "Black"))
    if result == "1/2-1/2":
        return "Draw"
    return "Unknown"


def _lichess_winner_labels(cfg) -> Dict[int, str]:
    pgn_dir = Path(str(cfg.data.pgn_dir)).expanduser()
    if not pgn_dir.exists():
        candidates = []
        if os.environ.get("INR_LICHESS_CACHE"):
            candidates.append(Path(os.environ["INR_LICHESS_CACHE"]).expanduser() / "pgn")
        candidates.append(Path.home() / ".cache" / "INR" / "lichess" / "pgn")
        candidates.append(ROOT / ".cache" / "lichess" / "pgn")
        for candidate in candidates:
            if candidate.exists():
                pgn_dir = candidate
                break
    players = [str(p) for p in cfg.data.get("players", [])]
    max_games = int(cfg.data.get("max_games_per_player", 200))
    min_plies = int(cfg.data.get("min_plies", 20))
    max_plies = int(cfg.data.get("max_plies", 120))
    tracked = bool(cfg.data.get("tracked_player_only", True))
    games_per_episode = int(cfg.data.get("games_per_episode", 1))

    raw_labels: list[str] = []
    raw_pids: list[int] = []
    for pid, player in enumerate(players):
        pgn_file = pgn_dir / f"{player}.pgn"
        games_taken = 0
        for game in _iter_games(pgn_file):
            if games_taken >= max_games:
                break
            _, moves = _game_to_positions_and_moves(
                game,
                min_plies=min_plies,
                max_plies=max_plies,
                tracked_player=player if tracked else None,
            )
            if not moves:
                continue
            raw_labels.append(_lichess_tracked_player_outcome(game, player))
            raw_pids.append(pid)
            games_taken += 1

    labels: list[str] = []
    cursor = 0
    if games_per_episode <= 1:
        labels = raw_labels
    else:
        while cursor < len(raw_pids):
            pid = int(raw_pids[cursor])
            end = cursor
            while end < len(raw_pids) and int(raw_pids[end]) == pid:
                end += 1
            usable = ((end - cursor) // games_per_episode) * games_per_episode
            for start in range(cursor, cursor + usable, games_per_episode):
                chunk = raw_labels[start:start + games_per_episode]
                labels.append(chunk[0] if len(set(chunk)) == 1 else "Mixed")
            cursor = end
    return {episode_id: label for episode_id, label in enumerate(labels)}


def _sample_store_for_plot_extraction(store, max_points_per_split: int | None, seed_text: str):
    if max_points_per_split is None:
        return store
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for is_ood in (False, True):
        idx = [i for i, m in enumerate(store.meta) if bool(getattr(m, "is_ood", False)) == is_ood]
        if len(idx) <= max_points_per_split:
            selected.extend(idx)
            continue
        pids = np.array([int(store.meta[i].policy_id) for i in idx], dtype=np.int64)
        idx_arr = np.array(idx, dtype=np.int64)
        picked_idx, _ = _subsample_for_plot(idx_arr[:, None], pids, max_points_per_split, seed=seed + int(is_ood))
        selected.extend(int(x) for x in picked_idx[:, 0])
    return store.subset(sorted(selected))


def _extract(
    run_dir: Path,
    max_points_per_split: int | None = None,
    label_fn: Callable[[object], str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]] | tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], np.ndarray] | None:
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

    base_store = _sample_store_for_plot_extraction(
        base_store,
        max_points_per_split,
        seed_text=f"{run_dir.name}:{cfg.seed}",
    )
    meta_labels = np.array([label_fn(m) for m in base_store.meta], dtype=object) if label_fn is not None else None

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
    if meta_labels is not None:
        return embs, pids, splits, policy_names, meta_labels
    return embs, pids, splits, policy_names


def _extract_fastf1_all_players(
    run_dir: Path,
    label_fn: Callable[[object], str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]] | tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], np.ndarray] | None:
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        return None

    cfg = OmegaConf.load(cfg_path)
    _remap_unavailable_cache_dirs(cfg)
    if not _fastf1_cache_path(cfg.data).exists() and (ROOT / ".cache" / "fastf1").exists():
        cfg.data.cache_dir = str(ROOT / ".cache" / "fastf1")

    drivers = _valid_fastf1_drivers(cfg.data)
    cfg.data.n_drivers = len(drivers)
    cfg.data.preferred_drivers = drivers
    cfg.data.name = f"{cfg.data.name}_all_players"

    base_store = _build_base_store(cfg.data)
    template = [PolicySpec(pid=int(p.pid), train=str(p.train), test=str(p.test)) for p in cfg.experiment.policies]
    policies = _expanded_fastf1_policies(template, len(drivers))
    shift_kwargs = dict(ood_fraction=float(cfg.shift.ood_fraction), seed=int(cfg.seed))
    if "min_per_partition" in cfg.shift:
        shift_kwargs["min_per_partition"] = int(cfg.shift.min_per_partition)
    behavior_unit = str(cfg.model.get("behavior_unit", "episode"))
    unit_window_size = int(cfg.model.get("unit_window_size", cfg.train.history_k))
    use_unit_latents = bool(cfg.model.get("use_unit_latents", False))
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
        use_unit_latents=use_unit_latents,
    )

    model = _fastf1_model_from_run(cfg, loaders, ckpt_path)
    device = _device(cfg)
    _load_model_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    policy_names = _policy_names_from_store(base_store, cfg)
    for meta in base_store.meta:
        meta.is_ood = str(meta.extras.get("predefined_split", "ID")).upper() == "OOD"
    meta_labels = np.array([label_fn(m) for m in base_store.meta], dtype=object) if label_fn is not None else None

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
    if meta_labels is not None:
        return embs, pids, splits, policy_names, meta_labels
    return embs, pids, splits, policy_names


def _filter_fastf1_drivers(
    embs: np.ndarray,
    pids: np.ndarray,
    splits: np.ndarray,
    policy_names: Dict[int, str],
    driver_codes: tuple[str, ...],
    labels: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]] | tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], np.ndarray]:
    code_to_pid = {name: pid for pid, name in policy_names.items()}
    missing = [code for code in driver_codes if code not in code_to_pid]
    if missing:
        raise ValueError(f"FastF1 driver codes missing from extracted policies: {missing}")
    keep_pids = {code_to_pid[code] for code in driver_codes}
    keep = np.array([int(pid) in keep_pids for pid in pids], dtype=bool)
    filtered_names = {
        code_to_pid[code]: f"{FASTF1_DRIVER_LABELS.get(code, code)} ({code})"
        for code in driver_codes
    }
    if labels is not None:
        return embs[keep], pids[keep], splits[keep], filtered_names, labels[keep]
    return embs[keep], pids[keep], splits[keep], filtered_names


def _plot_one(subdir: str, data_label: str, model: str, split: str,
              embs: np.ndarray, pids: np.ndarray, policy_names: Dict[int, str]):
    if embs.shape[0] < 4:
        print(f"[tsne] {data_label}/{model}/{split}: only {embs.shape[0]} embs, skipping", flush=True)
        return
    perplexity = max(2.0, min(30.0, embs.shape[0] / 4.0, embs.shape[0] - 1.0))
    ts = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
    xy = ts.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for pid in sorted(set(int(p) for p in pids)):
        mask = pids == pid
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=28, alpha=0.75,
            c=_policy_color(pid),
            label=f"{policy_names.get(pid, f'pid{pid}')}",
            edgecolors="none",
        )
    ax.set_title(f"{data_label} — {model} [{split}]\n(t-SNE of policy repr, n={embs.shape[0]})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    n_policies = len(set(int(p) for p in pids))
    ax.legend(loc="best", fontsize=6 if n_policies > 8 else 8, framealpha=0.6, ncol=2 if n_policies > 12 else 1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_dir = PLOTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tsne_{data_label}_{model}_{split}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[tsne] wrote {out}  (policies={sorted(set(int(p) for p in pids))})", flush=True)


def _category_color(i: int) -> str:
    cmap = plt.get_cmap("tab20")
    r, g, b, _ = cmap(i % 20)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _plot_one_by_labels(
    subdir: str,
    data_label: str,
    model: str,
    split: str,
    embs: np.ndarray,
    labels: np.ndarray,
    label_title: str,
):
    if embs.shape[0] < 4:
        print(f"[tsne] {data_label}/{model}/{split}/{label_title}: only {embs.shape[0]} embs, skipping", flush=True)
        return
    perplexity = max(2.0, min(30.0, embs.shape[0] / 4.0, embs.shape[0] - 1.0))
    xy = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca").fit_transform(embs)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    categories = sorted({str(x) for x in labels})
    for i, category in enumerate(categories):
        mask = labels.astype(str) == category
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=28, alpha=0.75,
            c=_category_color(i),
            label=category,
            edgecolors="none",
        )
    ax.set_title(f"{data_label} — {model} [{split}]\n(t-SNE colored by {label_title}, n={embs.shape[0]})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=6 if len(categories) > 8 else 8, framealpha=0.6, ncol=2 if len(categories) > 12 else 1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_dir = PLOTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label_title.lower().replace(" ", "_")
    out = out_dir / f"tsne_{data_label}_{model}_{split}_by_{safe_label}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[tsne] wrote {out}  ({label_title}={categories})", flush=True)


def _subsample_for_plot(
    embs: np.ndarray,
    pids: np.ndarray,
    max_points: int | None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points is None or embs.shape[0] <= max_points:
        return embs, pids
    rng = np.random.default_rng(seed)
    unique = sorted(set(pids.tolist() if hasattr(pids, "tolist") else pids), key=lambda x: str(x))
    base = max_points // len(unique)
    remainder = max_points % len(unique)
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for rank, pid in enumerate(unique):
        idx = np.flatnonzero(pids == pid)
        rng.shuffle(idx)
        quota = base + (1 if rank < remainder else 0)
        selected.append(idx[:min(quota, len(idx))])
        leftovers.append(idx[min(quota, len(idx)):])
    chosen = np.concatenate([x for x in selected if len(x)]) if selected else np.array([], dtype=int)
    if chosen.size < max_points:
        rest = np.concatenate([x for x in leftovers if len(x)]) if leftovers else np.array([], dtype=int)
        rng.shuffle(rest)
        chosen = np.concatenate([chosen, rest[:max_points - chosen.size]])
    rng.shuffle(chosen)
    chosen = chosen[:max_points]
    return embs[chosen], pids[chosen]


def plot_tsne(subdir: str, data_label: str, model: str, embs: np.ndarray,
              pids: np.ndarray, splits: np.ndarray, policy_names: Dict[int, str],
              max_points_per_plot: int | None = None):
    for split in ("ID", "OOD"):
        mask = splits == split
        split_embs, split_pids = _subsample_for_plot(
            embs[mask],
            pids[mask],
            max_points_per_plot,
            seed=int(hashlib.sha256(f"{data_label}:{model}:{split}".encode("utf-8")).hexdigest()[:8], 16),
        )
        _plot_one(subdir, data_label, model, split, split_embs, split_pids, policy_names)


def plot_tsne_by_labels(
    subdir: str,
    data_label: str,
    model: str,
    embs: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    label_title: str,
    max_points_per_plot: int | None = None,
):
    for split in ("ID", "OOD"):
        mask = splits == split
        split_embs, split_labels = _subsample_for_plot(
            embs[mask],
            labels[mask],
            max_points_per_plot,
            seed=int(hashlib.sha256(f"{data_label}:{model}:{split}:{label_title}".encode("utf-8")).hexdigest()[:8], 16),
        )
        _plot_one_by_labels(subdir, data_label, model, split, split_embs, split_labels, label_title)


def _filter_to_top_label_types(
    embs: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    n_types: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    counts = Counter(str(x) for x in labels)
    keep_labels = [label for label, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n_types]]
    keep = np.array([str(x) in set(keep_labels) for x in labels], dtype=bool)
    return embs[keep], splits[keep], labels[keep], keep_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default=None, help="Only render this policyClusters subdir, e.g. droid")
    ap.add_argument("--data-label", default=None, help="Only render this data label")
    ap.add_argument("--max-points-per-plot", type=int, default=None)
    args = ap.parse_args()

    for subdir, data_label, runs_root, data_cfg in RUN_GROUPS:
        if args.subdir is not None and subdir != args.subdir:
            continue
        if args.data_label is not None and data_label != args.data_label:
            continue
        models = DROID_MODELS if subdir == "droid" else MODELS
        for model in models:
            run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
            if not run_dir.exists():
                print(f"[tsne] SKIP {data_label}/{model}: {run_dir} missing", flush=True)
                continue
            try:
                result = _extract(run_dir, max_points_per_split=args.max_points_per_plot)
            except Exception as exc:
                print(f"[tsne] FAIL {data_label}/{model}: {exc}", flush=True)
                continue
            if result is None:
                print(f"[tsne] SKIP {data_label}/{model}: no cfg/ckpt", flush=True)
                continue
            embs, pids, splits, policy_names = result
            plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names, args.max_points_per_plot)

    subdir, data_label, runs_root, data_cfg = FASTF1_ALL_PLAYER_GROUP
    if (args.subdir is None or subdir == args.subdir) and (args.data_label is None or data_label == args.data_label):
        for model in MODELS:
            run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
            if not run_dir.exists():
                print(f"[tsne] SKIP {data_label}/{model}: {run_dir} missing", flush=True)
                continue
            try:
                result = _extract_fastf1_all_players(run_dir)
            except Exception as exc:
                print(f"[tsne] FAIL {data_label}/{model}: {exc}", flush=True)
                continue
            if result is None:
                print(f"[tsne] SKIP {data_label}/{model}: no cfg/ckpt", flush=True)
                continue
            embs, pids, splits, policy_names = result
            plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names, args.max_points_per_plot)

    subdir, data_label, runs_root, data_cfg = FASTF1_3PEOPLE_GROUP
    if (args.subdir is None or subdir == args.subdir) and (args.data_label is None or data_label == args.data_label):
        for model in MODELS:
            run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
            if not run_dir.exists():
                print(f"[tsne] SKIP {data_label}/{model}: {run_dir} missing", flush=True)
                continue
            try:
                result = _extract_fastf1_all_players(run_dir)
                if result is not None:
                    result = _filter_fastf1_drivers(*result, FASTF1_3PEOPLE_DRIVERS)
            except Exception as exc:
                print(f"[tsne] FAIL {data_label}/{model}: {exc}", flush=True)
                continue
            if result is None:
                print(f"[tsne] SKIP {data_label}/{model}: no cfg/ckpt", flush=True)
                continue
            embs, pids, splits, policy_names = result
            plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names, args.max_points_per_plot)

    subdir, data_label, runs_root, data_cfg = FASTF1_5PEOPLE_GROUP
    if (args.subdir is None or subdir == args.subdir) and (args.data_label is None or data_label == args.data_label):
        for model in MODELS:
            run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
            if not run_dir.exists():
                print(f"[tsne] SKIP {data_label}/{model}: {run_dir} missing", flush=True)
                continue
            try:
                result = _extract_fastf1_all_players(run_dir)
                if result is not None:
                    result = _filter_fastf1_drivers(*result, FASTF1_5PEOPLE_DRIVERS)
            except Exception as exc:
                print(f"[tsne] FAIL {data_label}/{model}: {exc}", flush=True)
                continue
            if result is None:
                print(f"[tsne] SKIP {data_label}/{model}: no cfg/ckpt", flush=True)
                continue
            embs, pids, splits, policy_names = result
            plot_tsne(subdir, data_label, model, embs, pids, splits, policy_names, args.max_points_per_plot)

    context_jobs = [
        ("fastf1_3people_track", "fastf1_3people", "track", "fastf1_gp"),
        ("droid_task_type", "droid_min300_hk300_remove", "task type", "droid_task"),
        ("droid_scene_type", "droid_min300_hk300_remove", "scene type", "droid_scene"),
        ("droid_object_type", "droid_min300_hk300_remove", "object type", "droid_object"),
        ("lichess_n200_winner", "lichess_full", "winner", "lichess_winner_full"),
        ("lichess_n200_winner", "lichess_sa16", "winner", "lichess_winner_sa16"),
    ]
    for context_subdir, context_data_label, label_title, label_kind in context_jobs:
        if args.subdir is not None and context_subdir != args.subdir:
            continue
        if args.data_label is not None and context_data_label != args.data_label:
            continue

        if label_kind == "fastf1_gp":
            _, _, runs_root, data_cfg = FASTF1_3PEOPLE_GROUP
            for model in MODELS:
                run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
                if not run_dir.exists():
                    print(f"[tsne] SKIP {context_data_label}/{model}: {run_dir} missing", flush=True)
                    continue
                try:
                    result = _extract_fastf1_all_players(run_dir, label_fn=lambda m: _context_label(m, label_kind))
                    if result is not None:
                        embs, pids, splits, policy_names, labels = result
                        result = _filter_fastf1_drivers(embs, pids, splits, policy_names, FASTF1_3PEOPLE_DRIVERS, labels)
                except Exception as exc:
                    print(f"[tsne] FAIL {context_data_label}/{model}/by_{label_title}: {exc}", flush=True)
                    continue
                if result is None:
                    print(f"[tsne] SKIP {context_data_label}/{model}: no cfg/ckpt", flush=True)
                    continue
                embs, _, splits, _, labels = result
                keep = np.array([str(x) in {"won", "lost"} for x in labels], dtype=bool)
                embs, splits, labels = embs[keep], splits[keep], labels[keep]
                plot_tsne_by_labels(context_subdir, context_data_label, model, embs, splits, labels, label_title, args.max_points_per_plot)
            continue

        if label_kind.startswith("droid_"):
            runs_root = ROOT / "outputs" / "droid" / "balanced_min300_remove_hk300"
            data_cfg = "droid_lowdim_full_balanced_min300_remove"
            for model in DROID_MODELS:
                run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
                if not run_dir.exists():
                    print(f"[tsne] SKIP {context_data_label}/{model}: {run_dir} missing", flush=True)
                    continue
                try:
                    result = _extract(run_dir, max_points_per_split=args.max_points_per_plot, label_fn=lambda m, lk=label_kind: _context_label(m, lk))
                except Exception as exc:
                    print(f"[tsne] FAIL {context_data_label}/{model}/by_{label_title}: {exc}", flush=True)
                    continue
                if result is None:
                    print(f"[tsne] SKIP {context_data_label}/{model}: no cfg/ckpt", flush=True)
                    continue
                embs, _, splits, _, labels = result
                embs, splits, labels, keep_labels = _filter_to_top_label_types(embs, splits, labels, n_types=3)
                print(f"[tsne] {context_data_label}/{model}/by_{label_title}: keeping top 3 {label_title} labels {keep_labels}", flush=True)
                plot_tsne_by_labels(context_subdir, context_data_label, model, embs, splits, labels, label_title, args.max_points_per_plot)
            continue

        if label_kind.startswith("lichess_winner"):
            runs_root = ROOT / "outputs" / "lichess" / "baseline_top3"
            data_cfg = "lichess_top3_full" if context_data_label == "lichess_full" else "lichess_top3_sa16"
            for model in MODELS:
                run_dir = runs_root / f"{data_cfg}__{model}__no_shift__s0"
                if not run_dir.exists():
                    print(f"[tsne] SKIP {context_data_label}/{model}: {run_dir} missing", flush=True)
                    continue
                try:
                    cfg = OmegaConf.load(run_dir / "config.yaml")
                    winners = _lichess_winner_labels(cfg)
                    result = _extract(
                        run_dir,
                        max_points_per_split=args.max_points_per_plot,
                        label_fn=lambda m, wm=winners: wm.get(int(m.episode_id), "Unknown"),
                    )
                except Exception as exc:
                    print(f"[tsne] FAIL {context_data_label}/{model}/by_{label_title}: {exc}", flush=True)
                    continue
                if result is None:
                    print(f"[tsne] SKIP {context_data_label}/{model}: no cfg/ckpt", flush=True)
                    continue
                embs, _, splits, _, labels = result
                keep = np.array([str(x) in {"won", "lost"} for x in labels], dtype=bool)
                embs, splits, labels = embs[keep], splits[keep], labels[keep]
                plot_tsne_by_labels(context_subdir, context_data_label, model, embs, splits, labels, label_title, args.max_points_per_plot)


if __name__ == "__main__":
    main()
