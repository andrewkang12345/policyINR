#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPRESENTATIVE_EXPERIMENTS = ("no_shift", "conflation", "generalization")
DERIVED_SOURCE = {
    "new_policy": "no_shift",
    "single_shift": "no_shift",
    "specialization": "no_shift",
    "novel_generalization": "generalization",
}
DERIVED_GPU = {
    "new_policy": 0,
    "single_shift": 1,
    "specialization": 2,
    "novel_generalization": 3,
}


def _run_name(data_name: str, model: str, experiment: str, seed: int) -> str:
    return f"{data_name}__{model}__{experiment}__s{seed}"


def _summary_exists(root: Path, data_name: str, model: str, experiment: str, seed: int) -> bool:
    return (root / _run_name(data_name, model, experiment, seed) / "summary.json").exists()


def _summary_path(root: Path, data_name: str, model: str, experiment: str, seed: int) -> Path:
    return root / _run_name(data_name, model, experiment, seed) / "summary.json"


def _report_summary(root: Path, data_name: str, model: str, experiment: str, seed: int) -> None:
    path = _summary_path(root, data_name, model, experiment, seed)
    summary = json.loads(path.read_text())
    ev = summary.get("eval", {})
    print(
        "[result] "
        f"{experiment}: "
        f"probe_acc={ev.get('probe_acc', float('nan')):.3f} "
        f"probe_acc_seen={ev.get('probe_acc_seen', float('nan')):.3f} "
        f"gen_acc={ev.get('gen_acc', float('nan')):.3f} "
        f"gen_nll={ev.get('gen_nll', float('nan')):.3f}",
        flush=True,
    )


def _launch_materialize(
    repo: Path,
    root: Path,
    data_name: str,
    model: str,
    experiment: str,
    seed: int,
    gpu: int,
) -> subprocess.Popen:
    log_dir = root / "parallel_materialize_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "scripts/materialize_shared_lichess_2x_runs.py",
        "--root", str(root),
        "--data-name", data_name,
        "--models", model,
        "--experiments", experiment,
        "--seed", str(seed),
    ]
    log_path = log_dir / f"{experiment}.log"
    print(f"[watcher] launch {experiment} on gpu {gpu}: {log_path}", flush=True)
    log = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._watcher_log_file = log  # type: ignore[attr-defined]
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-name", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-seconds", type=int, default=300)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = args.root
    print(f"[watcher] event monitor for {root}", flush=True)
    reported: set[str] = set()
    launched: dict[str, subprocess.Popen] = {}
    completed_derived: set[str] = set()
    all_experiments = set(REPRESENTATIVE_EXPERIMENTS) | set(DERIVED_SOURCE)
    while True:
        for exp in sorted(all_experiments):
            if exp not in reported and _summary_exists(root, args.data_name, args.model, exp, args.seed):
                _report_summary(root, args.data_name, args.model, exp, args.seed)
                reported.add(exp)

        for exp, src in DERIVED_SOURCE.items():
            if exp in launched or _summary_exists(root, args.data_name, args.model, exp, args.seed):
                continue
            if _summary_exists(root, args.data_name, args.model, src, args.seed):
                launched[exp] = _launch_materialize(
                    repo,
                    root,
                    args.data_name,
                    args.model,
                    exp,
                    args.seed,
                    DERIVED_GPU[exp],
                )

        for exp, proc in list(launched.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log = getattr(proc, "_watcher_log_file", None)
            if log is not None:
                log.close()
            print(f"[watcher] {exp} rc={rc}", flush=True)
            if rc != 0:
                raise RuntimeError(f"materialization failed for {exp}: rc={rc}")
            completed_derived.add(exp)
            launched.pop(exp)

        if all(_summary_exists(root, args.data_name, args.model, exp, args.seed) for exp in all_experiments):
            break

        if len(launched) == 0:
            done = sorted(exp for exp in all_experiments if _summary_exists(root, args.data_name, args.model, exp, args.seed))
            print(f"[watcher] done={done} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)

        time.sleep(args.poll_seconds)

    for exp in sorted(all_experiments - reported):
        _report_summary(root, args.data_name, args.model, exp, args.seed)
    subprocess.check_call([
        sys.executable,
        "-m",
        "eval.summary",
        str(root),
        "--out",
        str(root / "aggregate.csv"),
        "--md",
        str(root / "aggregate.md"),
    ], cwd=repo)
    print("[watcher] complete", flush=True)


if __name__ == "__main__":
    main()
