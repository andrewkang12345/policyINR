#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path = [str(ROOT)] + [p for p in sys.path if p != str(ROOT)]

from data import build_experiment_loaders
from eval import run_full_eval
from models import build_model
from scripts.reevaluate_probe_database import _build_model_for_run, _load_model_checkpoint
from train.main import _build_base_store, _device, _policies_from_cfg


REPRESENTATIVE_SOURCE = {
    "no_shift": "no_shift",
    "single_shift": "no_shift",
    "specialization": "no_shift",
    "new_policy": "no_shift",
    "conflation": "conflation",
    "generalization": "generalization",
    "novel_generalization": "generalization",
}


def _load_experiment_cfg(name: str):
    return OmegaConf.load(ROOT / "configs" / "experiment" / f"{name}.yaml")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        try:
            if src.resolve() == dst.resolve():
                return
        except FileNotFoundError:
            pass
        shutil.copy2(src, dst)


def materialize_run(root: Path, data_name: str, model: str, experiment: str, seed: int) -> Path:
    src_exp = REPRESENTATIVE_SOURCE[experiment]
    src_run = root / f"{data_name}__{model}__{src_exp}__s{seed}"
    dst_run = root / f"{data_name}__{model}__{experiment}__s{seed}"
    if not src_run.exists():
        raise FileNotFoundError(f"source run missing: {src_run}")
    best_ckpt = src_run / "best.pt"
    last_ckpt = src_run / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing in {src_run}")

    src_cfg = OmegaConf.load(src_run / "config.yaml")
    dst_cfg = OmegaConf.create(OmegaConf.to_container(src_cfg, resolve=True))
    dst_cfg.experiment = _load_experiment_cfg(experiment)
    dst_cfg.run_name = f"{data_name}__{model}__{experiment}__s{seed}"
    dst_cfg.output_dir = str(dst_run)

    dst_run.mkdir(parents=True, exist_ok=True)
    with (dst_run / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(dst_cfg, resolve=True))

    _copy_if_exists(src_run / "best.pt", dst_run / "best.pt")
    _copy_if_exists(src_run / "last.pt", dst_run / "last.pt")
    _copy_if_exists(src_run / "metrics.jsonl", dst_run / "metrics.jsonl")
    _copy_if_exists(src_run / "stdout.log", dst_run / "stdout.log")
    _copy_if_exists(src_run / "main.log", dst_run / "main.log")

    base_store = _build_base_store(dst_cfg.data)
    policies = _policies_from_cfg(dst_cfg.experiment.policies)
    shift_kwargs = dict(ood_fraction=float(dst_cfg.shift.ood_fraction), seed=int(dst_cfg.seed))
    if "min_per_partition" in dst_cfg.shift:
        shift_kwargs["min_per_partition"] = int(dst_cfg.shift.min_per_partition)
    loaders = build_experiment_loaders(
        base_store,
        policies=policies,
        history_k=int(dst_cfg.train.history_k),
        shift_kind=str(dst_cfg.shift.kind),
        shift_kwargs=shift_kwargs,
        batch_size=int(dst_cfg.train.batch_size),
        eval_batch_size=int(dst_cfg.train.eval_batch_size),
        num_workers=int(dst_cfg.train.num_workers),
        shuffle_history_train=bool(dst_cfg.model.shuffle_history_train),
        behavior_unit=str(dst_cfg.model.get("behavior_unit", "episode")),
        unit_window_size=int(dst_cfg.model.get("unit_window_size", dst_cfg.train.history_k)),
        use_unit_latents=bool(dst_cfg.model.get("use_unit_latents", False)),
        seed=int(dst_cfg.seed),
    )

    model_kwargs = OmegaConf.to_container(dst_cfg.model, resolve=True)
    model_kwargs.pop("name", None)
    kind = model_kwargs.pop("kind")
    model_kwargs.pop("shuffle_history_train", None)
    use_unit_latents = bool(model_kwargs.pop("use_unit_latents", False))
    behavior_unit = str(model_kwargs.pop("behavior_unit", "episode"))
    model_kwargs.pop("unit_window_size", None)
    model_kwargs.update(
        state_dim=loaders["state_dim"],
        action_dim=loaders["action_dim"],
        history_k=int(dst_cfg.train.history_k),
        action_kind=loaders.get("action_kind", "continuous"),
        n_actions=loaders.get("n_actions", None),
    )
    if use_unit_latents:
        model_kwargs["n_train_units"] = int(loaders.get("n_train_units", 0))
        model_kwargs["behavior_unit"] = str(loaders.get("behavior_unit", behavior_unit))
    model = _build_model_for_run(kind, model_kwargs, ckpt_path)
    device = _device(dst_cfg)
    _load_model_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    eval_out = run_full_eval(model=model, loaders=loaders, eval_cfg=dst_cfg.eval, device=device)
    with (dst_run / "eval.json").open("w") as f:
        json.dump(eval_out, f, indent=2, default=float)

    src_summary = json.loads((src_run / "summary.json").read_text())
    src_summary["run_name"] = dst_cfg.run_name
    src_summary["data"] = str(dst_cfg.data.name)
    src_summary["model"] = str(dst_cfg.model.name)
    src_summary["experiment"] = str(dst_cfg.experiment.name)
    src_summary["seed"] = int(dst_cfg.seed)
    src_summary["n_train_episodes"] = len(loaders["train_store"])
    src_summary["n_val_episodes"] = len(loaders["val_store"]) if loaders["val_store"] else 0
    src_summary["n_test_episodes"] = len(loaders["test_store"])
    src_summary["eval"] = eval_out
    src_summary["shared_checkpoint_from"] = src_run.name
    with (dst_run / "summary.json").open("w") as f:
        json.dump(src_summary, f, indent=2, default=float)
    return dst_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-name", type=str, default="lichess_full_2Xepisode")
    ap.add_argument("--models", type=str, default="cvae,inr_transformer_history_conditioned,inr_diffusion_history_conditioned,inr_transformer_fitted_latent")
    ap.add_argument("--experiments", type=str, default="no_shift,new_policy,single_shift,conflation,generalization,specialization,novel_generalization")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    for model in models:
        for experiment in experiments:
            run_dir = materialize_run(args.root, args.data_name, model, experiment, args.seed)
            print(run_dir, flush=True)


if __name__ == "__main__":
    main()
