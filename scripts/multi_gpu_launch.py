"""Fan out (dataset, model, experiment, seed) runs across N GPUs.

Each job is a subprocess of `train.main` with CUDA_VISIBLE_DEVICES
pinned to a single GPU. We keep N GPUs busy with a simple per-GPU
worker queue. Results go to <out_root>/<data>_<model>_<exp>_s<seed>/.
"""

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


@dataclass
class Job:
    data: str
    model: str
    experiment: str
    seed: int
    out_dir: Path


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_visible_devices(n_gpus: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        devices = [d.strip() for d in raw.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(n_gpus)]
    if len(devices) < n_gpus:
        raise ValueError(
            f"Requested {n_gpus} GPUs but only {len(devices)} visible via "
            f"CUDA_VISIBLE_DEVICES={raw!r}"
        )
    return devices[:n_gpus]


def worker(gpu_id: int, physical_gpu: str, q: "Queue[Job | None]", root: Path, extra_overrides: List[str], log):
    while True:
        job = q.get()
        if job is None:
            q.task_done()
            return
        job.out_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job.out_dir / "stdout.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        # Keep one worker pinned to one physical GPU (or one entry from the
        # inherited visible-device list). This preserves outer shell pinning.
        cmd = [
            sys.executable, "-m", "train.main",
            f"data={job.data}",
            f"model={job.model}",
            f"experiment={job.experiment}",
            f"seed={job.seed}",
            f"run_name={job.out_dir.name}",
            f"output_dir={job.out_dir}",
        ] + extra_overrides
        t0 = time.time()
        log(f"[gpu{gpu_id}:{physical_gpu}] START {job.out_dir.name}")
        with stdout_path.open("w") as f:
            try:
                res = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(root))
                ok = res.returncode == 0
            except Exception as e:
                f.write(f"\n[launcher] exception: {e}\n")
                ok = False
        dt = time.time() - t0
        log(f"[gpu{gpu_id}:{physical_gpu}] {'OK' if ok else 'FAIL'} {job.out_dir.name} ({dt:.1f}s)")
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpus", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--out-root", type=str, default="outputs/full_suite")
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--models", type=str, required=True)
    ap.add_argument("--experiments", type=str, required=True)
    ap.add_argument("--skip-completed", action="store_true",
                    help="Do not enqueue jobs whose output directory already has summary.json.")
    ap.add_argument("--overrides", nargs=argparse.REMAINDER, default=[],
                    help="Extra Hydra-style overrides appended to every run.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    visible_devices = _resolve_visible_devices(args.n_gpus)

    datasets = _parse_csv(args.datasets)
    models = _parse_csv(args.models)
    experiments = _parse_csv(args.experiments)
    seeds = [int(s) for s in _parse_csv(args.seeds)]

    jobs: List[Job] = []
    skipped = 0
    for d in datasets:
        for m in models:
            for e in experiments:
                for s in seeds:
                    name = f"{d}__{m}__{e}__s{s}"
                    out_dir = out_root / name
                    if args.skip_completed and (out_dir / "summary.json").exists():
                        skipped += 1
                        continue
                    jobs.append(Job(d, m, e, s, out_dir))

    print(f"[launcher] {len(jobs)} jobs across {args.n_gpus} GPUs "
          f"({len(datasets)} datasets x {len(models)} models x "
          f"{len(experiments)} experiments x {len(seeds)} seeds)")
    print(f"[launcher] visible devices: {visible_devices}")
    if skipped:
        print(f"[launcher] skipped {skipped} completed jobs")

    # extra overrides are passed through
    extras = args.overrides
    # strip leading `--`
    if extras and extras[0] == "--":
        extras = extras[1:]

    q: "Queue[Job | None]" = Queue()
    for j in jobs:
        q.put(j)
    for _ in range(args.n_gpus):
        q.put(None)

    log_path = out_root / "launcher.log"
    log_fp = log_path.open("w")

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    workers = [
        Thread(target=worker, args=(g, visible_devices[g], q, root, extras, log), daemon=True)
        for g in range(args.n_gpus)
    ]
    for w in workers:
        w.start()
    q.join()
    for w in workers:
        w.join()
    log_fp.close()
    print(f"[launcher] done. logs at {log_path}")


if __name__ == "__main__":
    main()
