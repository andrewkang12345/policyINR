"""Hydra entrypoint: single-run training + evaluation.

Composition:
  configs/data/<name>.yaml + configs/model/<name>.yaml +
  configs/experiment/<name>.yaml + configs/eval/default.yaml

Run:
  python -m train.main data=synthetic_grf_small model=cvae experiment=no_shift seed=0

Outputs land in `<output_dir>/`:
  config.yaml          resolved config
  metrics.jsonl        per-epoch train/val metrics
  best.pt / last.pt    checkpoints
  eval.json            linear probe + generative metrics
  summary.json         top-level run summary (+ eval metrics + wallclock)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

# allow running as `python -m train.main ...` OR `python train/main.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from utils import set_seed, get_logger
from utils.registry import MODELS
from data import (
    build_synthetic_store,
    build_minari_store,
    build_custom_mujoco_store,
    build_droid_store,
    build_fastf1_store,
    build_experiment_loaders,
)
from data.dmlab import build_dmlab_store
from data.lichess import build_lichess_store
from data.splits import PolicySpec
from models import build_model
from train.trainer import Trainer, TrainConfig
from eval import run_full_eval


def _build_base_store(data_cfg: DictConfig):
    if data_cfg.kind == "synthetic":
        return build_synthetic_store(
            n_policies=data_cfg.n_policies,
            episodes_per_policy=data_cfg.episodes_per_policy,
            episode_length=data_cfg.episode_length,
            state_dim=data_cfg.state_dim,
            action_dim=data_cfg.action_dim,
            state_generator=data_cfg.state_generator,
            action_policy=data_cfg.action_policy,
            noise_std=data_cfg.noise_std,
            seed=0,   # dataset-level seed; distinct from training seed
            state_gen_kwargs=OmegaConf.to_container(data_cfg.state_gen_kwargs, resolve=True) if "state_gen_kwargs" in data_cfg else {},
            action_gen_kwargs=OmegaConf.to_container(data_cfg.action_gen_kwargs, resolve=True) if "action_gen_kwargs" in data_cfg else {},
        )
    if data_cfg.kind == "minari":
        return build_minari_store(
            env_key=data_cfg.env,
            max_episodes_per_policy=data_cfg.get("max_episodes_per_policy", None),
            min_length=data_cfg.get("min_length", 32),
            max_length=data_cfg.get("max_length", None),
        )
    if data_cfg.kind == "custom_mujoco":
        generation_cfg = OmegaConf.to_container(data_cfg.get("generation", {}), resolve=True)
        return build_custom_mujoco_store(
            env_key=data_cfg.env,
            dataset_id=data_cfg.get("dataset_id", None),
            max_episodes_per_policy=data_cfg.get("max_episodes_per_policy", None),
            min_length=data_cfg.get("min_length", 32),
            max_length=data_cfg.get("max_length", None),
            generation_cfg=generation_cfg,
        )
    if data_cfg.kind == "dmlab":
        return build_dmlab_store(
            max_episodes_per_policy=data_cfg.get("max_episodes_per_policy", 60),
            cnn_feature_dim=data_cfg.get("cnn_feature_dim", 128),
            seed=data_cfg.get("featurizer_seed", 0),
        )
    if data_cfg.kind == "lichess":
        return build_lichess_store(
            pgn_dir=data_cfg.pgn_dir,
            players=tuple(data_cfg.get("players", ("DrNykterstein", "Hikaru", "FabianoCaruana"))),
            max_games_per_player=data_cfg.get("max_games_per_player", 200),
            min_plies=data_cfg.get("min_plies", 20),
            max_plies=data_cfg.get("max_plies", 120),
            tracked_player_only=bool(data_cfg.get("tracked_player_only", True)),
            games_per_episode=int(data_cfg.get("games_per_episode", 1)),
        )
    if data_cfg.kind == "droid":
        return build_droid_store(
            source=data_cfg.get("source", "droid_100"),
            data_dir=data_cfg.get("data_dir", os.environ.get("INR_DROID_CACHE",
                                              str(Path.home() / ".cache/INR/droid"))),
            max_shards=data_cfg.get("max_shards", None),
            max_episodes=int(data_cfg.get("max_episodes", 300)),
            n_collectors=int(data_cfg.get("n_collectors", 3)),
            min_episodes_per_collector=int(data_cfg.get("min_episodes_per_collector", 8)),
            min_length=int(data_cfg.get("min_length", 8)),
            max_length=data_cfg.get("max_length", None),
            ood_task_family=data_cfg.get("ood_task_family", None),
            collectors=tuple(data_cfg.get("collectors", ())) or None,
        )
    if data_cfg.kind == "fastf1":
        return build_fastf1_store(
            seasons=tuple(data_cfg.get("seasons", (2023, 2024))),
            gps=tuple(data_cfg.get("gps", ("Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco", "British", "Italian"))),
            ood_gp=str(data_cfg.get("ood_gp", "Monaco")),
            n_drivers=int(data_cfg.get("n_drivers", 3)),
            preferred_drivers=tuple(data_cfg.get("preferred_drivers", ())),
            min_points=int(data_cfg.get("min_points", 32)),
            max_points=data_cfg.get("max_points", 800),
            cache_dir=data_cfg.get("cache_dir", os.environ.get("INR_FASTF1_CACHE",
                                                str(Path.home() / ".cache/INR/fastf1"))),
        )
    raise ValueError(f"Unknown data kind: {data_cfg.kind}")


def _policies_from_cfg(policy_list) -> List[PolicySpec]:
    return [PolicySpec(pid=int(p.pid), train=str(p.train), test=str(p.test)) for p in policy_list]


def _device(cfg):
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return cfg.train.device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = get_logger("inr.main")
    set_seed(int(cfg.seed))

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # 1) data
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
    logger.info(
        f"data={cfg.data.name} exp={cfg.experiment.name} model={cfg.model.name} "
        f"#train_ep={len(loaders['train_store'])} "
        f"#val_ep={len(loaders['val_store']) if loaders['val_store'] else 0} "
        f"#test_ep={len(loaders['test_store']) if loaders['test_store'] else 0} "
        f"state_dim={loaders['state_dim']} action_dim={loaders['action_dim']} "
        f"shift_strength={loaders['shift_strength']} "
        f"shift_overlap={loaders['shift_overlap']:.4f} "
        f"shift_overlap_ratio={loaders['shift_overlap_ratio']:.4f} "
        f"effective_shared={loaders['effective_shared']:.3f} "
        f"fallback={loaders['shift_fallback']}"
    )

    # 2) model
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop("name", None)
    kind = model_kwargs.pop("kind")
    model_kwargs.pop("shuffle_history_train", None)
    use_unit_latents = bool(model_kwargs.pop("use_unit_latents", False))
    behavior_unit = str(model_kwargs.pop("behavior_unit", "episode"))
    unit_window_size = int(model_kwargs.pop("unit_window_size", cfg.train.history_k))
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
    model = build_model(kind, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model={kind} params={n_params/1e6:.2f}M latent={model.latent_dim}")

    # 3) train
    tcfg = TrainConfig(
        epochs=int(cfg.train.epochs),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        grad_clip=float(cfg.train.grad_clip),
        log_every=int(cfg.train.log_every),
        val_every_epoch=int(cfg.train.val_every_epoch),
        device=_device(cfg),
        amp=bool(cfg.train.amp),
        early_stop_patience=int(cfg.train.early_stop_patience),
    )
    trainer = Trainer(
        model, loaders["train_loader"], loaders["val_loader"],
        cfg=tcfg, output_dir=out, run_name=cfg.run_name,
    )
    train_summary = trainer.fit()

    # load best (or last if no val) for eval
    best_ckpt = out / "best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=tcfg.device, weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()

    # 4) eval
    eval_out = run_full_eval(
        model=model,
        loaders=loaders,
        eval_cfg=cfg.eval,
        device=tcfg.device,
    )
    with (out / "eval.json").open("w") as f:
        json.dump(eval_out, f, indent=2, default=float)

    # 5) write a top-level summary
    summary = {
        "run_name": str(cfg.run_name),
        "shift_strength": {int(k): float(v) for k, v in loaders["shift_strength"].items()},
        "shift_overlap": float(loaders["shift_overlap"]),
        "shift_overlap_ratio": float(loaders["shift_overlap_ratio"]),
        "effective_shared": float(loaders["effective_shared"]),
        "shift_fallback": str(loaders["shift_fallback"]),
        "data": str(cfg.data.name),
        "model": str(cfg.model.name),
        "experiment": str(cfg.experiment.name),
        "seed": int(cfg.seed),
        "state_dim": loaders["state_dim"],
        "action_dim": loaders["action_dim"],
        "n_train_episodes": len(loaders["train_store"]),
        "n_val_episodes": len(loaders["val_store"]) if loaders["val_store"] else 0,
        "n_test_episodes": len(loaders["test_store"]) if loaders["test_store"] else 0,
        "train": train_summary,
        "eval": eval_out,
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"done: {summary}")


if __name__ == "__main__":
    main()
