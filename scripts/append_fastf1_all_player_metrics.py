#!/usr/bin/env python3
"""Append all-valid-driver FastF1 probe/kNN rows for existing checkpoints.

This is intentionally probe-only: it reuses trained representation models,
expands the FastF1 store to every driver with both ID and OOD stints, computes
linear probe and cosine kNN metrics, and writes separate summary rows under
`all_player_metrics/` so existing aggregate rows are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path = [str(ROOT)] + [p for p in sys.path if p != str(ROOT)]

from data import build_experiment_loaders
from data.splits import PolicySpec
from eval.linear_probe import cosine_knn_probe, linear_probe
from eval.runner import _per_episode_representations
from eval.summary import aggregate_runs
from scripts.reevaluate_probe_database import (
    _build_base_store,
    _build_model_for_run,
    _device,
    _load_model_checkpoint,
    _remap_unavailable_cache_dirs,
)


def _fastf1_cache_path(data_cfg) -> Path:
    cache_dir = Path(str(data_cfg.get("cache_dir", os.environ.get("INR_FASTF1_CACHE", str(Path.home() / ".cache/INR/fastf1"))))).expanduser()
    seasons = tuple(data_cfg.get("seasons", (2023, 2024)))
    gps = tuple(data_cfg.get("gps", ("Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco", "British", "Italian")))
    tag = "_".join(str(x) for x in seasons) + "_" + "_".join(str(g).replace(" ", "") for g in gps)
    max_points = data_cfg.get("max_points", 800)
    return cache_dir / f"fastf1_stints_{tag}_P{max_points or 'full'}.npz"


def _valid_drivers(data_cfg) -> List[str]:
    path = _fastf1_cache_path(data_cfg)
    records = list(np.load(path, allow_pickle=True)["records"])
    ood_gp = str(data_cfg.get("ood_gp", "Monaco"))
    by_driver_gp = defaultdict(Counter)
    for r in records:
        by_driver_gp[str(r["driver"])][str(r["gp"])] += 1
    eligible = [
        d for d, gps in by_driver_gp.items()
        if gps.get(ood_gp, 0) > 0 and sum(v for k, v in gps.items() if k != ood_gp) > 0
    ]
    eligible.sort(key=lambda d: sum(by_driver_gp[d].values()), reverse=True)
    preferred = [str(d) for d in data_cfg.get("preferred_drivers", ()) if str(d) in eligible]
    out: List[str] = []
    for d in preferred + eligible:
        if d not in out:
            out.append(d)
    return out


def _expanded_policies(template, n_policies: int) -> List[PolicySpec]:
    if not template:
        raise ValueError("experiment has no policy template")
    out = []
    for pid in range(n_policies):
        src = template[pid % len(template)]
        out.append(PolicySpec(pid=pid, train=str(src.train), test=str(src.test)))
    return out


def _model_from_run(cfg, loaders, ckpt_path: Path):
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name", None)
    kind = model_kwargs.pop("kind")
    model_kwargs.pop("shuffle_history_train", None)
    use_unit_latents = bool(model_kwargs.pop("use_unit_latents", False))
    behavior_unit = str(model_kwargs.pop("behavior_unit", "episode"))
    model_kwargs.pop("unit_window_size", None)
    model_kwargs.update(
        state_dim=loaders["state_dim"],
        action_dim=loaders["action_dim"],
        history_k=int(cfg.train.history_k),
        action_kind=loaders.get("action_kind", "continuous"),
        n_actions=loaders.get("n_actions", None),
    )
    if use_unit_latents:
        ckpt_units = _checkpoint_unit_latent_count(ckpt_path)
        # All-player evaluation is probe-only and fitted-latent
        # extract_representation infers z from support history. The checkpoint's
        # original embedding table is still needed for initialization, so its
        # size must match the saved state, not the expanded all-player loader.
        model_kwargs["n_train_units"] = int(ckpt_units or loaders.get("n_train_units", 0))
        model_kwargs["behavior_unit"] = str(loaders.get("behavior_unit", behavior_unit))
    return _build_model_for_run(kind, model_kwargs, ckpt_path)


def _checkpoint_unit_latent_count(ckpt_path: Path) -> int | None:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    except Exception:
        return None
    weight = state.get("unit_latents.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def _compute_run(run_dir: Path, root: Path, overwrite: bool = False) -> None:
    cfg_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    if not (cfg_path.exists() and summary_path.exists() and ckpt_path.exists()):
        raise FileNotFoundError(f"missing config/summary/checkpoint in {run_dir}")

    cfg = OmegaConf.load(cfg_path)
    _remap_unavailable_cache_dirs(cfg)
    if str(cfg.data.get("kind", "")) != "fastf1":
        return

    out_dir = root / "all_player_metrics" / run_dir.name
    out_summary = out_dir / "summary.json"
    if out_summary.exists() and not overwrite:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    drivers = _valid_drivers(cfg.data)
    cfg.data.n_drivers = len(drivers)
    cfg.data.preferred_drivers = drivers
    cfg.data.name = f"{cfg.data.name}_all_players"

    base_store = _build_base_store(cfg.data)
    template = [PolicySpec(pid=int(p.pid), train=str(p.train), test=str(p.test)) for p in cfg.experiment.policies]
    policies = _expanded_policies(template, len(drivers))
    shift_kwargs = dict(ood_fraction=float(cfg.shift.ood_fraction), seed=int(cfg.seed))
    if "min_per_partition" in cfg.shift:
        shift_kwargs["min_per_partition"] = int(cfg.shift.min_per_partition)
    loaders = build_experiment_loaders(
        base_store,
        policies=policies,
        history_k=int(cfg.train.history_k),
        shift_kind=str(cfg.shift.kind),
        shift_kwargs=shift_kwargs,
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle_history_train=bool(cfg.model.shuffle_history_train),
        behavior_unit=str(cfg.model.get("behavior_unit", "episode")),
        unit_window_size=int(cfg.model.get("unit_window_size", cfg.train.history_k)),
        use_unit_latents=bool(cfg.model.get("use_unit_latents", False)),
        materialize_datasets=bool(cfg.train.get("materialize_dataset", False)),
        seed=int(cfg.seed),
    )

    device = _device(cfg)
    model = _model_from_run(cfg, loaders, ckpt_path)
    _load_model_checkpoint(model, ckpt_path, device)
    model.to(device).eval()

    train_embs, train_labels, _ = _per_episode_representations(
        model, loaders["train_store"],
        history_k=int(loaders.get("history_k", 16)),
        per_episode_samples=int(cfg.eval.per_episode_samples),
        state_mean=loaders["state_mean"], state_std=loaders["state_std"],
        action_mean=loaders["action_mean"], action_std=loaders["action_std"],
        shuffle_history=True, device=device,
        min_len=int(cfg.eval.min_episode_length_for_eval),
        behavior_unit=str(loaders.get("behavior_unit", "episode")),
        unit_window_size=int(loaders.get("unit_window_size", 0)),
        known_unit_map=loaders.get("train_unit_map") if loaders.get("use_unit_latents", False) else None,
    )
    test_embs, test_labels, oods = _per_episode_representations(
        model, loaders["test_store"],
        history_k=int(loaders.get("history_k", 16)),
        per_episode_samples=int(cfg.eval.per_episode_samples),
        state_mean=loaders["state_mean"], state_std=loaders["state_std"],
        action_mean=loaders["action_mean"], action_std=loaders["action_std"],
        shuffle_history=True, device=device,
        min_len=int(cfg.eval.min_episode_length_for_eval),
        behavior_unit=str(loaders.get("behavior_unit", "episode")),
        unit_window_size=int(loaders.get("unit_window_size", 0)),
        known_unit_map=loaders.get("train_unit_map") if loaders.get("use_unit_latents", False) else None,
    )
    train_pids = sorted({int(m.policy_id) for m in loaders["train_store"].meta})
    eval_out = linear_probe(
        train_embs, train_labels,
        test_embs, test_labels,
        train_seen_pids=train_pids,
        C=float(cfg.eval.probe_C),
        max_iter=int(cfg.eval.probe_max_iter),
    )
    eval_out.update(cosine_knn_probe(train_embs, train_labels, test_embs, test_labels, k=5))
    eval_out.update({
        "train_pids": [int(p) for p in train_pids],
        "test_pids": [int(p) for p in sorted(set(test_labels.tolist()))],
        "n_train_episodes_used": int(train_embs.shape[0]),
        "n_test_episodes_used": int(test_embs.shape[0]),
        "n_test_ood": int(oods.sum()),
        "probe_protocol": "all_valid_fastf1_drivers_train_probe_on_train_embeddings_eval_on_test_embeddings",
        "probe_only": True,
        "fastf1_drivers": drivers,
        "n_fastf1_drivers": len(drivers),
    })

    src_summary = json.loads(summary_path.read_text())
    out_summary_obj = {
        "data": str(cfg.data.name),
        "model": src_summary.get("model", cfg.model.name),
        "experiment": src_summary.get("experiment", cfg.experiment.name),
        "seed": src_summary.get("seed", int(cfg.seed)),
        "n_train_episodes": len(loaders["train_store"]),
        "n_val_episodes": len(loaders["val_store"]) if loaders["val_store"] else 0,
        "n_test_episodes": len(loaders["test_store"]) if loaders["test_store"] else 0,
        "source_run_dir": str(run_dir),
        "eval": eval_out,
    }
    with out_summary.open("w") as f:
        json.dump(out_summary_obj, f, indent=2, default=float)


def _source_run_dirs(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("summary.json"):
        if "all_player_metrics" in p.parts:
            continue
        rd = p.parent
        if (rd / "config.yaml").exists() and ((rd / "best.pt").exists() or (rd / "last.pt").exists()):
            out.append(rd)
    return sorted(out)


def _worker(gpu_id: int, q: "Queue[Path | None]", root: Path, overwrite: bool, log_path: Path) -> None:
    with log_path.open("a") as log_fp:
        def log(msg: str) -> None:
            print(msg, flush=True)
            log_fp.write(msg + "\n")
            log_fp.flush()

        while True:
            run_dir = q.get()
            if run_dir is None:
                q.task_done()
                return
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--run-dir", str(run_dir), "--root", str(root),
            ]
            if overwrite:
                cmd.append("--overwrite")
            t0 = time.time()
            log(f"[gpu{gpu_id}] START {run_dir}")
            res = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                tail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown error"
                log(f"[gpu{gpu_id}] FAIL {run_dir} :: {tail}")
            else:
                log(f"[gpu{gpu_id}] OK {run_dir} ({time.time() - t0:.1f}s)")
            q.task_done()


def _run_root(root: Path, n_gpus: int, overwrite: bool) -> None:
    runs = _source_run_dirs(root)
    log_path = root / "all_player_metrics.log"
    if log_path.exists() and overwrite:
        log_path.unlink()
    print(f"[launcher] {len(runs)} source runs across {n_gpus} GPUs")
    q: "Queue[Path | None]" = Queue()
    for rd in runs:
        q.put(rd)
    for _ in range(n_gpus):
        q.put(None)
    workers = [Thread(target=_worker, args=(gpu, q, root, overwrite, log_path), daemon=True) for gpu in range(n_gpus)]
    for w in workers:
        w.start()
    q.join()
    for w in workers:
        w.join()
    aggregate_runs(root, out_csv=root / "aggregate.csv", out_md=root / "aggregate.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("INR_ROOT", ".")) / "outputs/fastf1/uncapped_full_suite")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.run_dir is not None:
        _compute_run(args.run_dir.resolve(), root=root, overwrite=bool(args.overwrite))
        return
    _run_root(root, n_gpus=int(args.n_gpus), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
