#!/usr/bin/env python3
"""Recompute strict train->test probe metrics for saved runs.

Single-run mode:
  python scripts/reevaluate_probe_database.py --run-dir <run_dir>

Database mode:
  python scripts/reevaluate_probe_database.py --root outputs --n-gpus 4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path = [str(ROOT)] + [p for p in sys.path if p != str(ROOT)]
for mod_name in ("eval", "eval.runner", "eval.summary"):
    sys.modules.pop(mod_name, None)

from eval.summary import aggregate_runs
from eval.runner import _per_episode_representations
from eval.linear_probe import cosine_knn_probe, linear_probe
from models import build_model
from train.main import _build_base_store, _device, _policies_from_cfg
from data import build_experiment_loaders
from models.base import HistoryEncoder, MLP


class LegacyCVAE(nn.Module):
    """Checkpoint-compatible CVAE used only for reevaluating old runs."""

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
        self.n_actions = n_actions
        self.history = HistoryEncoder(
            state_dim=state_dim, action_dim=action_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            history_k=history_k, permutation_invariant=True, dropout=dropout,
            action_kind=action_kind, n_actions=n_actions,
        )
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.state_embed = MLP([state_dim, d_model, d_model])
        out_dim = n_actions if action_kind == "discrete" else action_dim
        self.decoder = MLP([latent_dim + d_model, decoder_hidden, decoder_hidden, out_dim])

    def _encode(self, past_states, past_actions):
        h = self.history(past_states, past_actions)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar

    def forward(self, batch):
        mu, logvar = self._encode(batch["past_states"], batch["past_actions"])
        z = mu
        cond = self.state_embed(batch["current_state"])
        out = self.decoder(torch.cat([z, cond], dim=-1))
        target = batch["next_action"]
        if self.action_kind == "discrete":
            recon = F.cross_entropy(out, target.long())
        else:
            recon = F.mse_loss(out, target)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {"loss": recon + self.kl_weight * kl, "pred": out, "recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def extract_representation(self, batch):
        mu, _ = self._encode(batch["past_states"], batch["past_actions"])
        return mu

    @torch.no_grad()
    def predict_action(self, batch):
        mu, _ = self._encode(batch["past_states"], batch["past_actions"])
        cond = self.state_embed(batch["current_state"])
        out = self.decoder(torch.cat([mu, cond], dim=-1))
        if self.action_kind == "discrete":
            return out.argmax(dim=-1)
        return out


def _remap_legacy_state_dict_keys(state_dict: dict) -> dict:
    remapped = dict(state_dict)
    key_map = {
        "pair_embed.weight": "pair_embed.proj.weight",
        "pair_embed.bias": "pair_embed.proj.bias",
        "policy_head.out.weight": "policy_head.action_head.out.weight",
        "policy_head.out.bias": "policy_head.action_head.out.bias",
        "decoder.net.0.weight": "decoder_body.net.0.weight",
        "decoder.net.0.bias": "decoder_body.net.0.bias",
        "decoder.net.2.weight": "decoder_body.net.2.weight",
        "decoder.net.2.bias": "decoder_body.net.2.bias",
        "decoder.net.4.weight": "action_head.out.weight",
        "decoder.net.4.bias": "action_head.out.bias",
    }
    for old_key, new_key in key_map.items():
        if old_key in remapped and new_key not in remapped:
            remapped[new_key] = remapped.pop(old_key)
    # Older history-conditioned INR checkpoints stored the Transformer
    # history encoder at module root. Current models wrap it under
    # `encoder`, so only the policy_head keys are already aligned.
    for key in list(remapped.keys()):
        if key.startswith("policy_head."):
            continue
        if key in {"type_pair", "pos"} or key.startswith(("pair_embed.", "latent_head.", "latent_norm.")):
            new_key = f"encoder.{key}"
        elif key.startswith("encoder.layers."):
            new_key = f"encoder.{key}"
        else:
            continue
        if new_key not in remapped:
            remapped[new_key] = remapped.pop(key)
    return remapped


def _build_model_for_run(kind: str, model_kwargs: dict, ckpt_path: Path):
    if kind == "cvae":
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        if any(k.startswith("decoder.net.") for k in state):
            return LegacyCVAE(**model_kwargs)
    return build_model(kind, **model_kwargs)


def _load_model_checkpoint(model, ckpt_path: Path, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state["model"]
    try:
        model.load_state_dict(model_state)
        return
    except RuntimeError:
        model_state = _remap_legacy_state_dict_keys(model_state)
        model.load_state_dict(model_state)


def _recompute_run(run_dir: Path) -> None:
    cfg_path = run_dir / "config.yaml"
    eval_path = run_dir / "eval.json"
    summary_path = run_dir / "summary.json"
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
    if not (cfg_path.exists() and summary_path.exists() and ckpt_path.exists()):
        raise FileNotFoundError(f"missing config/summary/checkpoint in {run_dir}")

    cfg = OmegaConf.load(cfg_path)
    _remap_unavailable_cache_dirs(cfg)
    _remap_runtime_cache_roots(cfg)

    base_store = _build_base_store(cfg.data)
    policies = _policies_from_cfg(cfg.experiment.policies)
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
        model_kwargs["n_train_units"] = int(loaders.get("n_train_units", 0))
        model_kwargs["behavior_unit"] = str(loaders.get("behavior_unit", behavior_unit))
    model = _build_model_for_run(kind, model_kwargs, ckpt_path)
    device = _device(cfg)
    _load_model_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

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
    probe_out = linear_probe(
        train_embs, train_labels,
        test_embs, test_labels,
        train_seen_pids=train_pids,
        C=float(cfg.eval.probe_C),
        max_iter=int(cfg.eval.probe_max_iter),
    )
    probe_out.update(cosine_knn_probe(train_embs, train_labels, test_embs, test_labels, k=5))
    probe_out["train_pids"] = [int(p) for p in train_pids]
    probe_out["test_pids"] = [int(p) for p in sorted(set(test_labels.tolist()))]
    probe_out["n_train_episodes_used"] = int(train_embs.shape[0])
    probe_out["n_test_episodes_used"] = int(test_embs.shape[0])
    probe_out["n_test_ood"] = int(oods.sum())
    probe_out["probe_protocol"] = "train_probe_on_train_embeddings_eval_on_test_embeddings"

    eval_out = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    eval_out.update(probe_out)
    with eval_path.open("w") as f:
        json.dump(eval_out, f, indent=2, default=float)

    with summary_path.open() as f:
        summary = json.load(f)
    summary["eval"] = eval_out
    summary["n_train_episodes"] = len(loaders["train_store"])
    summary["n_val_episodes"] = len(loaders["val_store"]) if loaders["val_store"] else 0
    summary["n_test_episodes"] = len(loaders["test_store"]) if loaders["test_store"] else 0
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)


def _remap_unavailable_cache_dirs(cfg) -> None:
    """Use local cache mirrors when old run configs point at unavailable HPC paths."""
    if "data" not in cfg or "cache_dir" not in cfg.data:
        return
    cache_dir = Path(str(cfg.data.cache_dir)).expanduser()
    try:
        available = cache_dir.exists()
    except PermissionError:
        available = False
    if available:
        return

    path_s = str(cache_dir)
    if "INR_cache/fastf1" in path_s:
        cfg.data.cache_dir = os.environ.get("INR_FASTF1_CACHE",   str(pathlib.Path.home() / ".cache/INR/fastf1"))
    elif "INR_cache/droid" in path_s:
        cfg.data.cache_dir = os.environ.get("INR_DROID_CACHE",    str(pathlib.Path.home() / ".cache/INR/droid"))
    elif "INR_cache/lichess" in path_s:
        cfg.data.cache_dir = os.environ.get("INR_LICHESS_CACHE",  str(pathlib.Path.home() / ".cache/INR/lichess"))


def _remap_runtime_cache_roots(cfg) -> None:
    """Point modules with import-time default cache roots at local mirrors."""
    if "data" not in cfg:
        return
    if str(cfg.data.get("kind", "")) == "dmlab":
        import data.dmlab as dmlab
        dmlab.DEFAULT_CACHE_ROOT = Path(os.environ.get("INR_DMLAB_CACHE",     str(pathlib.Path.home() / ".cache/INR/dmlab")))
        dmlab.DEFAULT_SHARD_CACHE = Path(os.environ.get("INR_RLU_DMLAB_CACHE", str(pathlib.Path.home() / ".cache/INR/rlu_dmlab")))


def _has_knn_metrics(summary_path: Path) -> bool:
    try:
        with summary_path.open() as f:
            eval_out = json.load(f).get("eval", {})
    except Exception:
        return False
    return "knn_acc1" in eval_out and "knn_acc5" in eval_out


def _collect_run_dirs(root: Path, *, missing_knn_only: bool = False) -> List[Path]:
    run_dirs = []
    for summary_path in root.rglob("summary.json"):
        if missing_knn_only and _has_knn_metrics(summary_path):
            continue
        run_dir = summary_path.parent
        if (run_dir / "config.yaml").exists() and ((run_dir / "best.pt").exists() or (run_dir / "last.pt").exists()):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def _collect_aggregate_targets(root: Path) -> List[Path]:
    parents = set()
    for p in root.rglob("aggregate.csv"):
        parents.add(p.parent)
    for p in root.rglob("aggregate.md"):
        parents.add(p.parent)
    return sorted(parents)


def _refresh_aggregates(root: Path) -> None:
    for agg_root in _collect_aggregate_targets(root):
        data_root = agg_root / "runs" if (agg_root / "runs").exists() else agg_root
        aggregate_runs(data_root, out_csv=agg_root / "aggregate.csv", out_md=agg_root / "aggregate.md")


def _worker(gpu_id: int, q: "Queue[Path | None]", root: Path, log_path: Path) -> None:
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
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir)]
            t0 = time.time()
            log(f"[gpu{gpu_id}] START {run_dir}")
            try:
                res = subprocess.run(cmd, cwd=str(root), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                ok = res.returncode == 0
                if not ok and res.stderr:
                    tail = res.stderr.strip().splitlines()[-1]
                    log(f"[gpu{gpu_id}] ERR {run_dir} :: {tail}")
            except Exception as e:
                ok = False
                log(f"[gpu{gpu_id}] EXC {run_dir} :: {e}")
            dt = time.time() - t0
            log(f"[gpu{gpu_id}] {'OK' if ok else 'FAIL'} {run_dir} ({dt:.1f}s)")
            q.task_done()


def _run_database(root: Path, n_gpus: int, *, missing_knn_only: bool = False) -> None:
    run_dirs = _collect_run_dirs(root, missing_knn_only=missing_knn_only)
    log_path = root / "reevaluate_probe_database.log"
    if log_path.exists():
        log_path.unlink()
    print(f"[launcher] {len(run_dirs)} runs across {n_gpus} GPUs")
    q: "Queue[Path | None]" = Queue()
    for run_dir in run_dirs:
        q.put(run_dir)
    for _ in range(n_gpus):
        q.put(None)
    workers = [Thread(target=_worker, args=(gpu, q, ROOT, log_path), daemon=True) for gpu in range(n_gpus)]
    for w in workers:
        w.start()
    q.join()
    for w in workers:
        w.join()
    _refresh_aggregates(root)
    print(f"[launcher] done. log at {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("INR_OUTPUTS_ROOT", str(pathlib.Path(__file__).resolve().parent.parent / "outputs"))))
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--missing-knn-only", action="store_true")
    args = ap.parse_args()

    if args.run_dir is not None:
        _recompute_run(args.run_dir.resolve())
        return
    _run_database(args.root.resolve(), args.n_gpus, missing_knn_only=bool(args.missing_knn_only))


if __name__ == "__main__":
    main()
