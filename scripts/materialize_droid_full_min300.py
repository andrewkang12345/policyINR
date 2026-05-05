#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from data.droid import GCS_BUCKET, GCS_PREFIX_BY_SOURCE, _episode_to_arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="droid")
    ap.add_argument("--data-dir", default=os.environ.get("INR_DROID_CACHE", str(Path.home() / ".cache/INR/droid")))
    ap.add_argument("--collectors", default="RAIL,IRIS,CLVR")
    ap.add_argument("--ood-task-family", default="marker")
    ap.add_argument("--min-length", type=int, default=300)
    ap.add_argument("--max-episodes", type=int, default=1_000_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds
    from google.cloud import storage

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    data_dir = Path(args.data_dir).expanduser()
    ds_dir = data_dir / args.source / "1.0.0"
    if not (ds_dir / "dataset_info.json").exists():
        raise SystemExit(f"Missing TFDS metadata at {ds_dir}; run a capped DROID download first.")

    collectors = tuple(x.strip() for x in args.collectors.split(",") if x.strip())
    collector_tag = "-".join(collectors)
    ood_tag = args.ood_task_family
    out = Path(args.out) if args.out else data_dir / (
        f"{args.source}_v4_Sall_E{args.max_episodes}_C{len(collectors)}_"
        f"P{collector_tag}_O{ood_tag}_min{args.min_length}_Lfull.npz"
    )
    progress_path = out.with_suffix(".progress.json")

    states, actions, infos = [], [], []
    done = set()
    if args.resume and progress_path.exists():
        progress = json.load(open(progress_path))
        done = set(progress.get("done", []))
        partial = out.with_suffix(".partial.npz")
        if partial.exists():
            z = np.load(partial, allow_pickle=True)
            states = list(z["states"])
            actions = list(z["actions"])
            infos = list(z["infos"])

    builder = tfds.builder_from_directory(str(ds_dir))
    client = storage.Client.create_anonymous_client()
    prefix = GCS_PREFIX_BY_SOURCE[args.source]
    blobs = sorted(
        [b for b in client.list_blobs(GCS_BUCKET, prefix=prefix) if ".tfrecord-" in b.name],
        key=lambda b: b.name,
    )
    existing = {p.name: p for p in ds_dir.glob("*.tfrecord-*")}

    print(f"[materialize] shards={len(blobs)} existing={len(existing)} out={out}", flush=True)
    for si, blob in enumerate(blobs):
        name = Path(blob.name).name
        if name in done:
            continue
        local_path = existing.get(name)
        delete_after = False
        if local_path is None:
            fd, tmp_name = tempfile.mkstemp(prefix=f"{name}.", suffix=".tmp", dir=str(data_dir))
            os.close(fd)
            local_path = Path(tmp_name)
            blob.download_to_filename(str(local_path))
            delete_after = True

        kept_this = 0
        try:
            for raw in tf.data.TFRecordDataset([str(local_path)]):
                ex = builder.info.features.deserialize_example_np(raw.numpy())
                s, a, info = _episode_to_arrays(ex)
                if info.get("collector") not in collectors:
                    continue
                if len(s) < args.min_length:
                    continue
                states.append(s)
                actions.append(a)
                infos.append(info)
                kept_this += 1
                if len(states) >= args.max_episodes:
                    break
        finally:
            if delete_after:
                local_path.unlink(missing_ok=True)

        done.add(name)
        if si % 10 == 0 or kept_this or len(states) >= args.max_episodes:
            partial = out.with_suffix(".partial.npz")
            np.savez_compressed(
                partial,
                states=np.array(states, dtype=object),
                actions=np.array(actions, dtype=object),
                infos=np.array(infos, dtype=object),
            )
            json.dump(
                {"done": sorted(done), "n": len(states), "last_shard": name},
                open(progress_path, "w"),
                indent=2,
            )
            print(f"[materialize] {si+1}/{len(blobs)} kept_total={len(states)} kept_shard={kept_this}", flush=True)
        if len(states) >= args.max_episodes:
            break

    np.savez_compressed(
        out,
        states=np.array(states, dtype=object),
        actions=np.array(actions, dtype=object),
        infos=np.array(infos, dtype=object),
    )
    print(f"[materialize] complete episodes={len(states)} out={out}", flush=True)


if __name__ == "__main__":
    main()
