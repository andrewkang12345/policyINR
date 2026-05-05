"""DROID low-dimensional RLDS loader.

This loader intentionally ignores image/video observations. It consumes the
public RLDS release (the default capped source is ``droid_100``), extracts
only proprioceptive state and continuous robot actions, and labels policies
by collector/source id inferred from the RLDS file path.

ID/OOD is a predefined split over task/object families derived from language
instructions. The builder chooses the top collectors that share enough task
families, then reserves one shared family as OOD for every selected collector.
Use ``shift.kind=predefined_split`` in experiment configs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List
import os
import re

import numpy as np

from .base import EpisodeMeta, EpisodeStore


DEFAULT_CACHE_ROOT = Path(os.environ.get("INR_DROID_CACHE",
                                          Path.home() / ".cache/INR/droid")).expanduser()
GCS_BUCKET = "gresearch"
GCS_PREFIX_BY_SOURCE = {
    "droid_100": "robotics/droid_100/1.0.0",
    "droid": "robotics/droid/1.0.0",
}


def _decode_bytes(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _download_gcs_prefix(source: str, data_dir: Path, max_shards: int | None = None) -> Path:
    """Download TFDS metadata and a capped number of TFRecord shards."""
    from google.cloud import storage

    if source not in GCS_PREFIX_BY_SOURCE:
        raise ValueError(f"Unknown DROID source '{source}'. Expected one of {sorted(GCS_PREFIX_BY_SOURCE)}")
    prefix = GCS_PREFIX_BY_SOURCE[source]
    target = data_dir / source / "1.0.0"
    target.mkdir(parents=True, exist_ok=True)

    client = storage.Client.create_anonymous_client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=prefix))
    metadata = [b for b in blobs if b.name.endswith(("dataset_info.json", "features.json", "CC-BY-4.0"))]
    shards = sorted([b for b in blobs if ".tfrecord-" in b.name], key=lambda b: b.name)
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    for blob in metadata + shards:
        dest = target / Path(blob.name).name
        if dest.exists() and dest.stat().st_size == blob.size:
            continue
        blob.download_to_filename(dest)
    return target


def _collector_from_path(path: str) -> str:
    parts = [p for p in Path(path).parts if p and p != "/"]
    for marker in ("r2d2-data-full", "r2d2-data"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    # Fallback to a stable high-level source folder.
    return parts[-5] if len(parts) >= 5 else "unknown"


_STOPWORDS = {
    "the", "a", "an", "in", "into", "on", "onto", "to", "from", "with", "and",
    "put", "place", "move", "pick", "up", "open", "close", "take", "set",
    "bring", "push", "pull", "drawer", "table", "counter",
    "one", "two", "three", "four", "five", "first", "second", "third",
    "green", "red", "blue", "white", "black", "grey", "gray", "orange",
    "yellow", "small", "large", "big", "middle", "left", "right",
}


def _task_family(instruction: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", instruction.lower())
    toks = [t for t in text.split() if t and t not in _STOPWORDS]
    if not toks:
        return text.strip() or "unknown"
    # First content token usually captures the object family: marker, cup, bowl, etc.
    return toks[0]


def _load_examples(data_dir: Path, source: str, max_shards: int | None):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    ds_dir = data_dir / source / "1.0.0"
    if not (ds_dir / "dataset_info.json").exists():
        ds_dir = _download_gcs_prefix(source, data_dir, max_shards=max_shards)
    elif max_shards is not None:
        # Ensure the requested capped shard set exists even if metadata was
        # downloaded by an earlier failed/partial run.
        existing = sorted(ds_dir.glob("*.tfrecord-*"))
        if len(existing) < int(max_shards):
            ds_dir = _download_gcs_prefix(source, data_dir, max_shards=max_shards)
    builder = tfds.builder_from_directory(str(ds_dir))
    shard_paths = sorted(ds_dir.glob("*.tfrecord-*"))
    if max_shards is not None:
        shard_paths = shard_paths[: int(max_shards)]

    # Read only the local capped shards. ``builder.as_dataset`` expects all
    # shards advertised in dataset_info.json, which is not true for capped
    # full-release slices.
    def iter_local_records():
        for shard_path in shard_paths:
            for raw in tf.data.TFRecordDataset([str(shard_path)]):
                yield builder.info.features.deserialize_example_np(raw.numpy())

    if max_shards is not None:
        return iter_local_records()
    return tfds.as_numpy(builder.as_dataset(split="train"))


def _episode_to_arrays(ex) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = ex["episode_metadata"]
    file_path = _decode_bytes(meta.get("file_path", b""))
    recording_path = _decode_bytes(meta.get("recording_folderpath", b""))

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    instruction = ""
    for step in ex["steps"]:
        obs = step["observation"]
        state = np.concatenate([
            np.asarray(obs["cartesian_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["joint_position"], dtype=np.float32).reshape(-1),
            np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1),
        ])
        action = np.asarray(step["action"], dtype=np.float32).reshape(-1)
        states.append(state)
        actions.append(action)
        if not instruction:
            instruction = _decode_bytes(step.get("language_instruction", b""))
    info = {
        "file_path": file_path,
        "recording_folderpath": recording_path,
        "collector": _collector_from_path(file_path or recording_path),
        "instruction": instruction,
        "task_family": _task_family(instruction),
    }
    return np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32), info


def _select_collectors(records: list[tuple[np.ndarray, np.ndarray, dict]],
                       n_collectors: int,
                       min_episodes_per_collector: int) -> list[str]:
    counts = Counter(info["collector"] for _, _, info in records)
    candidates = [c for c, n in counts.most_common() if n >= min_episodes_per_collector]
    if len(candidates) < n_collectors:
        candidates = [c for c, _ in counts.most_common()]

    by_collector = defaultdict(Counter)
    for _, _, info in records:
        by_collector[info["collector"]][info["task_family"]] += 1

    import itertools
    best = None
    best_score = None
    for combo in itertools.combinations(candidates, n_collectors):
        shared = set.intersection(*[
            {fam for fam in by_collector[c] if fam and fam != "unknown"}
            for c in combo
        ])
        if not shared:
            continue
        family_counts = [
            [by_collector[c][fam] for c in combo]
            for fam in shared
        ]
        # Prefer a balanced OOD family across collectors over simply choosing
        # the biggest collectors, otherwise capped DROID slices can yield
        # unusable OOD splits with only 1 episode for one collector.
        max_min_ood = max(min(cs) for cs in family_counts)
        best_sum_ood = max(sum(cs) for cs in family_counts if min(cs) == max_min_ood)
        score = (max_min_ood, best_sum_ood, sum(counts[c] for c in combo), len(shared))
        if best_score is None or score > best_score:
            best = combo
            best_score = score
    if best is not None:
        return list(best)
    return candidates[:n_collectors]


def _choose_ood_family(records: list[tuple[np.ndarray, np.ndarray, dict]], collectors: list[str]) -> str:
    by_collector = defaultdict(Counter)
    for _, _, info in records:
        if info["collector"] in collectors:
            by_collector[info["collector"]][info["task_family"]] += 1
    shared = set.intersection(*[set(by_collector[c]) for c in collectors])
    shared = {fam for fam in shared if fam and fam != "unknown"}
    if not shared:
        all_counts = Counter()
        for c in collectors:
            all_counts.update({fam: n for fam, n in by_collector[c].items() if fam and fam != "unknown"})
        if not all_counts:
            for c in collectors:
                all_counts.update(by_collector[c])
        return all_counts.most_common(1)[0][0]
    return max(shared, key=lambda fam: (min(by_collector[c][fam] for c in collectors),
                                        sum(by_collector[c][fam] for c in collectors)))


def build_droid_store(
    source: str = "droid_100",
    data_dir: str | Path = DEFAULT_CACHE_ROOT,
    max_shards: int | None = None,
    max_episodes: int = 300,
    n_collectors: int = 3,
    min_episodes_per_collector: int = 8,
    min_length: int = 8,
    max_length: int | None = None,
    ood_task_family: str | None = None,
    collectors: list[str] | tuple[str, ...] | None = None,
) -> EpisodeStore:
    data_dir = Path(data_dir).expanduser()
    collector_tag = "auto" if collectors is None else "-".join(str(c) for c in collectors)
    ood_tag = ood_task_family or "auto"
    cache_tag = (
        f"{source}_v4_S{max_shards or 'all'}_E{max_episodes}_C{n_collectors}_"
        f"P{collector_tag}_O{ood_tag}_min{min_length}_L{max_length or 'full'}"
    )
    cache_path = data_dir / f"{cache_tag}.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        states = list(z["states"])
        actions = list(z["actions"])
        infos = list(z["infos"])
    else:
        records = []
        for ex in _load_examples(data_dir, source, max_shards=max_shards):
            s, a, info = _episode_to_arrays(ex)
            if len(s) < min_length:
                continue
            if max_length is not None:
                s = s[: int(max_length)]
                a = a[: int(max_length)]
            records.append((s, a, info))
            if len(records) >= max_episodes:
                break
        selected_collectors = (
            [str(c) for c in collectors]
            if collectors is not None
            else _select_collectors(records, n_collectors, min_episodes_per_collector)
        )
        records = [r for r in records if r[2]["collector"] in selected_collectors]
        states = [r[0] for r in records]
        actions = [r[1] for r in records]
        infos = [r[2] for r in records]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            states=np.array(states, dtype=object),
            actions=np.array(actions, dtype=object),
            infos=np.array(infos, dtype=object),
        )

    selected_collectors = (
        [str(c) for c in collectors]
        if collectors is not None
        else _select_collectors(list(zip(states, actions, infos)), n_collectors, min_episodes_per_collector)
    )
    ood_family = ood_task_family or _choose_ood_family(list(zip(states, actions, infos)), selected_collectors)
    collector_to_pid = {c: i for i, c in enumerate(selected_collectors)}

    out_states: List[np.ndarray] = []
    out_actions: List[np.ndarray] = []
    out_meta: List[EpisodeMeta] = []
    for s, a, info in zip(states, actions, infos):
        collector = info["collector"]
        if collector not in collector_to_pid:
            continue
        is_ood = info["task_family"] == ood_family
        out_states.append(np.asarray(s, dtype=np.float32))
        out_actions.append(np.asarray(a, dtype=np.float32))
        out_meta.append(EpisodeMeta(
            episode_id=len(out_meta),
            policy_id=collector_to_pid[collector],
            is_ood=is_ood,
            source=f"droid/{collector}",
            extras={
                **dict(info),
                "predefined_split": "OOD" if is_ood else "ID",
                "ood_task_family": ood_family,
            },
        ))

    if not out_states:
        raise RuntimeError("DROID builder produced no episodes after collector/task filtering")
    return EpisodeStore(
        states=out_states,
        actions=out_actions,
        meta=out_meta,
        state_dim=int(out_states[0].shape[-1]),
        action_dim=int(out_actions[0].shape[-1]),
        source=f"droid/{source}",
        action_kind="continuous",
    )
