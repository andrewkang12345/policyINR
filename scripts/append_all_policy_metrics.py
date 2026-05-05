#!/usr/bin/env python3
"""Append all-valid-policy probe/kNN rows for existing checkpoints.

The added rows are probe-only and live under `all_policy_metrics/`, so source
run summaries and existing aggregate rows are preserved.  "Valid" means a
policy has at least one ID and one OOD episode under the run's configured split.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterable, List

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path = [str(ROOT)] + [p for p in sys.path if p != str(ROOT)]

from data import build_experiment_loaders
from data.shifts import assign_state_shift
from data.splits import PolicySpec
from eval.linear_probe import cosine_knn_probe, linear_probe
from eval.runner import _per_episode_representations
from eval.summary import aggregate_runs
from scripts.reevaluate_probe_database import (
    _build_model_for_run,
    _load_model_checkpoint,
    _remap_runtime_cache_roots,
    _remap_unavailable_cache_dirs,
)
from train.main import _build_base_store, _device


def _source_run_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("summary.json"):
        if "all_policy_metrics" in p.parts or "all_player_metrics" in p.parts:
            continue
        rd = p.parent
        if (rd / "config.yaml").exists() and ((rd / "best.pt").exists() or (rd / "last.pt").exists()):
            out.append(rd)
    return sorted(out)


def _checkpoint_unit_latent_count(ckpt_path: Path) -> int | None:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    except Exception:
        return None
    weight = state.get("unit_latents.weight")
    return int(weight.shape[0]) if weight is not None else None


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
        # Probe extraction for fitted-latent INR infers z from support history;
        # the saved table is only needed to restore the checkpoint/init mean.
        model_kwargs["n_train_units"] = int(ckpt_units or loaders.get("n_train_units", 0))
        model_kwargs["behavior_unit"] = str(loaders.get("behavior_unit", behavior_unit))
    return _build_model_for_run(kind, model_kwargs, ckpt_path)


def _valid_policy_ids(base_store, cfg) -> List[int]:
    shift_kwargs = dict(ood_fraction=float(cfg.shift.ood_fraction), seed=int(cfg.seed))
    if "min_per_partition" in cfg.shift:
        shift_kwargs["min_per_partition"] = int(cfg.shift.min_per_partition)
    shifted, _ = assign_state_shift(base_store, kind=str(cfg.shift.kind), **shift_kwargs)
    counts = Counter((int(m.policy_id), bool(m.is_ood)) for m in shifted.meta)
    pids = sorted({int(m.policy_id) for m in shifted.meta})
    return [pid for pid in pids if counts[(pid, False)] > 0 and counts[(pid, True)] > 0]


def _expanded_policies(template: Iterable, valid_pids: List[int]) -> List[PolicySpec]:
    template = list(template)
    if not template:
        raise ValueError("experiment has no policy template")
    out: List[PolicySpec] = []
    for i, pid in enumerate(valid_pids):
        src = template[i % len(template)]
        out.append(PolicySpec(pid=int(pid), train=str(src.train), test=str(src.test)))
    return out


def _compute_run(run_dir: Path, root: Path, overwrite: bool = False) -> None:
    cfg_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    if not (cfg_path.exists() and summary_path.exists() and ckpt_path.exists()):
        raise FileNotFoundError(f"missing config/summary/checkpoint in {run_dir}")

    out_dir = root / "all_policy_metrics" / run_dir.name
    out_summary = out_dir / "summary.json"
    if out_summary.exists() and not overwrite:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(cfg_path)
    _remap_unavailable_cache_dirs(cfg)
    _remap_runtime_cache_roots(cfg)
    base_store = _build_base_store(cfg.data)
    valid_pids = _valid_policy_ids(base_store, cfg)
    if not valid_pids:
        raise RuntimeError(f"no valid all-policy ids found for {run_dir}")

    template = [PolicySpec(pid=int(p.pid), train=str(p.train), test=str(p.test)) for p in cfg.experiment.policies]
    policies = _expanded_policies(template, valid_pids)
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
        "all_policy_pids": [int(p) for p in valid_pids],
        "n_all_policies": int(len(valid_pids)),
        "n_train_episodes_used": int(train_embs.shape[0]),
        "n_test_episodes_used": int(test_embs.shape[0]),
        "n_test_ood": int(oods.sum()),
        "probe_protocol": "all_valid_policies_train_probe_on_train_embeddings_eval_on_test_embeddings",
        "probe_only": True,
    })

    src_summary = json.loads(summary_path.read_text())
    data_name = str(src_summary.get("data", cfg.data.name))
    out_summary_obj = {
        "data": f"{data_name}_all_policies",
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
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir), "--root", str(root)]
            if overwrite:
                cmd.append("--overwrite")
            t0 = time.time()
            log(f"[gpu{gpu_id}] START {run_dir}")
            res = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                tail = "\n".join((res.stderr.strip() or res.stdout.strip()).splitlines()[-8:])
                log(f"[gpu{gpu_id}] FAIL {run_dir} :: {tail}")
            else:
                log(f"[gpu{gpu_id}] OK {run_dir} ({time.time() - t0:.1f}s)")
            q.task_done()


def _run_root(root: Path, n_gpus: int, overwrite: bool) -> None:
    runs = _source_run_dirs(root)
    log_path = root / "all_policy_metrics.log"
    if log_path.exists() and overwrite:
        log_path.unlink()
    print(f"[launcher] {root}: {len(runs)} source runs across {n_gpus} GPUs", flush=True)
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
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.run_dir is not None:
        _compute_run(args.run_dir.resolve(), root=root, overwrite=bool(args.overwrite))
        return
    _run_root(root, n_gpus=max(1, int(args.n_gpus)), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
