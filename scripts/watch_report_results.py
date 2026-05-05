#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def _run_name(data_name: str, model: str, experiment: str, seed: int) -> str:
    return f"{data_name}__{model}__{experiment}__s{seed}"


def _summary_path(root: Path, data_name: str, model: str, experiment: str, seed: int) -> Path:
    return root / _run_name(data_name, model, experiment, seed) / "summary.json"


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "nan"


def _report(path: Path) -> None:
    summary = json.loads(path.read_text())
    ev = summary.get("eval", {})
    train = summary.get("train", {})
    print(
        "[result] "
        f"{summary.get('experiment')}: "
        f"probe_acc={_fmt(ev.get('probe_acc'))} "
        f"probe_acc_seen={_fmt(ev.get('probe_acc_seen'))} "
        f"gen_acc={_fmt(ev.get('gen_acc'))} "
        f"gen_nll={_fmt(ev.get('gen_nll'))} "
        f"best_epoch={train.get('best_epoch')} "
        f"epochs={train.get('total_epochs')}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-name", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--experiments", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    reported: set[str] = set()
    print(f"[watcher] reporting {len(experiments)} runs under {args.root}", flush=True)
    while len(reported) < len(experiments):
        for exp in experiments:
            if exp in reported:
                continue
            path = _summary_path(args.root, args.data_name, args.model, exp, args.seed)
            if path.exists():
                _report(path)
                reported.add(exp)
        done = sorted(reported)
        print(f"[watcher] done={done} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        time.sleep(args.poll_seconds)

    subprocess.check_call([
        sys.executable,
        "-m",
        "eval.summary",
        str(args.root),
        "--out",
        str(args.root / "aggregate.csv"),
        "--md",
        str(args.root / "aggregate.md"),
    ], cwd=repo)
    print("[watcher] aggregate written", flush=True)


if __name__ == "__main__":
    main()
