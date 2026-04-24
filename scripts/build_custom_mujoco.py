"""Generate custom MuJoCo Minari datasets across multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.custom_mujoco import MUJOCO_ENVS, _default_dataset_id, generate_custom_mujoco_dataset


@dataclass
class Job:
    env_key: str
    log_dir: Path


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_single_env_args(args, env_key: str) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-env", env_key,
        "--generation-mode", args.generation_mode,
        "--id-episodes", str(args.id_episodes),
        "--ood-episodes", str(args.ood_episodes),
        "--episode-horizon", str(args.episode_horizon),
        "--min-episode-length", str(args.min_episode_length),
        "--calibration-episodes", str(args.calibration_episodes),
        "--calibration-horizon", str(args.calibration_horizon),
        "--calibration-stride", str(args.calibration_stride),
        "--n-components", str(args.n_components),
        "--qpos-noise-scale", str(args.qpos_noise_scale),
        "--qvel-noise-scale", str(args.qvel_noise_scale),
        "--std-floor", str(args.std_floor),
        "--clip-margin-scale", str(args.clip_margin_scale),
        "--knn-k", str(args.knn_k),
        "--tail-fraction", str(args.tail_fraction),
        "--partition-train-frac", str(args.partition_train_frac),
        "--partition-val-frac", str(args.partition_val_frac),
        "--candidate-multiplier", str(args.candidate_multiplier),
        "--seed", str(args.seed),
    ] + (["--force-rebuild"] if args.force_rebuild else [])


def worker(gpu_id: int, q: "Queue[Job | None]", args, log):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        job.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job.log_dir / "build.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = _build_single_env_args(args, job.env_key)
        t0 = time.time()
        log(f"[gpu{gpu_id}] START {job.env_key}")
        with stdout_path.open("w") as f:
            res = subprocess.run(cmd, env=env, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        log(f"[gpu{gpu_id}] {'OK' if res.returncode == 0 else 'FAIL'} {job.env_key} ({dt:.1f}s)")
        q.task_done()


def generate_one(args):
    if args.generation_mode in {"rollout_episode", "resampled_steps"}:
        dist_cfg = {
            "kind": "clustered_reference",
            "n_components": args.n_components,
            "qpos_noise_scale": args.qpos_noise_scale,
            "qvel_noise_scale": args.qvel_noise_scale,
            "std_floor": args.std_floor,
            "clip_margin_scale": args.clip_margin_scale,
        }
        extra = {"state_distribution": dist_cfg}
    elif args.generation_mode == "action_resampled_steps":
        dist_cfg = {
            "kind": "policy_action_clustered_reference",
            "n_components": args.n_components,
            "qpos_noise_scale": args.qpos_noise_scale,
            "qvel_noise_scale": args.qvel_noise_scale,
            "std_floor": args.std_floor,
            "clip_margin_scale": args.clip_margin_scale,
        }
        extra = {"action_distribution": dist_cfg}
    elif args.generation_mode in {"state_resampled_v2", "state_resampled_v3"}:
        extra = {
            "state_distribution": {
                "kind": "shared_state_density_v2",
                "knn_k": args.knn_k,
                "tail_fraction": args.tail_fraction,
            }
        }
    elif args.generation_mode == "action_resampled_v2":
        extra = {
            "action_distribution": {
                "kind": "shared_state_action_disagreement_v2",
                "tail_fraction": args.tail_fraction,
            }
        }
    elif args.generation_mode == "action_resampled_v3":
        extra = {
            "action_distribution": {
                "kind": "shared_state_action_mean_matched_v3",
                "tail_fraction": args.tail_fraction,
                "train_frac": args.partition_train_frac,
                "val_frac": args.partition_val_frac,
                "candidate_multiplier": args.candidate_multiplier,
            }
        }
    elif args.generation_mode in {"action_resampled_v4", "action_resampled_v5"}:
        extra = {
            "action_distribution": {
                "kind": "minari_id_action_matched_ood_v4",
                "tail_fraction": args.tail_fraction,
                "train_frac": args.partition_train_frac,
                "val_frac": args.partition_val_frac,
                "acceptance_threshold": args.acceptance_threshold,
                "candidate_batch_size": args.candidate_batch_size,
                "max_candidate_batches": args.max_candidate_batches,
                "probe_like_samples": args.probe_like_samples,
                "probe_like_repeats": args.probe_like_repeats,
            }
        }
    else:
        raise KeyError(f"Unsupported generation_mode '{args.generation_mode}'")
    generate_custom_mujoco_dataset(
        args.single_env,
        dataset_id=args.dataset_id,
        device=args.device,
        deterministic=True,
        episode_horizon=args.episode_horizon,
        min_episode_length=args.min_episode_length,
        episodes_per_policy={"ID": args.id_episodes, "OOD": args.ood_episodes},
        calibration={
            "episodes_per_policy": args.calibration_episodes,
            "horizon": args.calibration_horizon,
            "sample_stride": args.calibration_stride,
        },
        generation_mode=args.generation_mode,
        force_rebuild=args.force_rebuild,
        seed=args.seed,
        **extra,
    )


def launch_many(args):
    envs = _parse_csv(args.envs)
    for env_key in envs:
        if env_key not in MUJOCO_ENVS:
            raise KeyError(f"Unknown env '{env_key}'. Choices: {sorted(MUJOCO_ENVS)}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_fp = (out_root / "launcher.log").open("w")

    def log(msg: str):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    q: "Queue[Job | None]" = Queue()
    for env_key in envs:
        q.put(Job(env_key=env_key, log_dir=out_root / env_key))
    for _ in range(args.n_gpus):
        q.put(None)

    log(f"[launcher] {len(envs)} custom MuJoCo builds across {args.n_gpus} GPUs")
    workers = [Thread(target=worker, args=(gpu, q, args, log), daemon=True) for gpu in range(args.n_gpus)]
    for w in workers:
        w.start()
    q.join()
    for w in workers:
        w.join()
    log_fp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=str, default="ant,halfcheetah,hopper,humanoid,walker2d")
    ap.add_argument("--single-env", type=str, default="")
    ap.add_argument("--generation-mode", type=str, default="rollout_episode")
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--out-root", type=str, default="outputs/custom_mujoco_build")
    ap.add_argument("--dataset-id", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--id-episodes", type=int, default=256)
    ap.add_argument("--ood-episodes", type=int, default=128)
    ap.add_argument("--episode-horizon", type=int, default=128)
    ap.add_argument("--min-episode-length", type=int, default=32)
    ap.add_argument("--calibration-episodes", type=int, default=4)
    ap.add_argument("--calibration-horizon", type=int, default=256)
    ap.add_argument("--calibration-stride", type=int, default=4)
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--qpos-noise-scale", type=float, default=0.15)
    ap.add_argument("--qvel-noise-scale", type=float, default=0.25)
    ap.add_argument("--std-floor", type=float, default=0.05)
    ap.add_argument("--clip-margin-scale", type=float, default=0.25)
    ap.add_argument("--knn-k", type=int, default=32)
    ap.add_argument("--tail-fraction", type=float, default=0.4)
    ap.add_argument("--partition-train-frac", type=float, default=0.7)
    ap.add_argument("--partition-val-frac", type=float, default=0.15)
    ap.add_argument("--candidate-multiplier", type=int, default=12)
    ap.add_argument("--acceptance-threshold", type=float, default=2.5)
    ap.add_argument("--candidate-batch-size", type=int, default=2048)
    ap.add_argument("--max-candidate-batches", type=int, default=128)
    ap.add_argument("--probe-like-samples", type=int, default=8)
    ap.add_argument("--probe-like-repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    if args.single_env:
        if not args.dataset_id:
            args.dataset_id = _default_dataset_id(args.single_env, args.generation_mode)
        generate_one(args)
        return

    launch_many(args)


if __name__ == "__main__":
    main()
