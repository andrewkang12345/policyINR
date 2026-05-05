"""RL Unplugged DMLab snapshot loader.

Policy / shift layout (per user spec):

  policy_id=0  ->  snapshot_0   (eps=0.0 ID, eps=0.25 OOD)
  policy_id=1  ->  snapshot_1   (eps=0.0 ID, eps=0.25 OOD)

For each policy we load both the eps=0.0 and the eps=0.25 shards. The
eps=0.0 episodes carry predefined_split='ID' and is_ood=False; the
eps=0.25 episodes carry predefined_split='OOD' and is_ood=True. Use
`shift.kind=predefined_split` to consume those tags directly. The
snapshot_0_eps_0.25 config that used to be a standalone third policy
is no longer exposed as a separate pid; it now serves as snapshot_0's
OOD partition.

Rather than going through `tfds.load()` (which requires downloading
and rebuilding the full dataset locally via apache_beam), we read a
small number of TFRecord shards directly from the public GCS bucket
`gs://rl_unplugged/dmlab` via HTTPS. Per-shard size is ~200 MB and each
shard contains many 301-step episodes; a handful of shards per config
is plenty for the representation experiments.

Schema (from tfds RLU DMLab builder):
  episode_length = 301 for all snapshot configs
  observations_pixels : (T,) string (PNG bytes, 72x96x3)
  observations_reward : (T,) float32  (previous reward)
  observations_action : (T, 15) float32 (one-hot previous-action)
  actions             : (T,) int64    (current action)

Per-step state for the pipeline is built by the `DMLabFeaturizer`
(frozen CNN on pixels + one-hot last_action + last_reward).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import os

import numpy as np

from .base import EpisodeMeta, EpisodeStore


DMLAB_POLICY_CONFIGS: List[Tuple[str, List[Tuple[str, str]]]] = [
    # (policy_name, [(config_name, split_tag), ...]). Each policy pulls
    # eps=0.0 episodes (ID) and eps=0.25 episodes (OOD) from the same
    # snapshot, producing a per-policy ID/OOD partition consumed by
    # shift.kind=predefined_split.
    ("snapshot_0", [("snapshot_0_eps_0.0", "ID"), ("snapshot_0_eps_0.25", "OOD")]),
    ("snapshot_1", [("snapshot_1_eps_0.0", "ID"), ("snapshot_1_eps_0.25", "OOD")]),
]
DMLAB_N_ACTIONS = 15
DMLAB_EPISODE_LENGTH = 301
DMLAB_TASK = "seekavoid_arena_01"
DMLAB_SHARDS_BUCKET = "https://storage.googleapis.com/rl_unplugged/dmlab"
DEFAULT_CACHE_ROOT = Path(os.environ.get("INR_DMLAB_CACHE",
                                          Path.home() / ".cache/INR/dmlab")).expanduser()
DEFAULT_SHARD_CACHE = Path(os.environ.get("INR_DMLAB_SHARD_CACHE",
                                           os.environ.get("INR_RLU_DMLAB_CACHE",
                                           Path.home() / ".cache/INR/rlu_dmlab"))).expanduser()


def _shard_url(config_name: str, shard_id: int) -> str:
    return f"{DMLAB_SHARDS_BUCKET}/{DMLAB_TASK}/{config_name}/tfrecord-{shard_id:05d}-of-00500"


def _download_shard(config_name: str, shard_id: int) -> Path:
    dest = DEFAULT_SHARD_CACHE / config_name / f"tfrecord-{shard_id:05d}-of-00500"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import urllib.request
    url = _shard_url(config_name, shard_id)
    urllib.request.urlretrieve(url, dest)
    return dest


def _parse_tfrecord_shard(shard_path: Path,
                          max_episodes: int,
                          ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Parse one RLU DMLab tfrecord shard into lists of numpy arrays.

    Episodes are always full length (T = DMLAB_EPISODE_LENGTH = 301).
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    feature_desc = {
        "episode_id": tf.io.FixedLenFeature([], tf.int64),
        "episode_return": tf.io.FixedLenFeature([], tf.float32),
        "observations_pixels": tf.io.FixedLenFeature([DMLAB_EPISODE_LENGTH], tf.string),
        "observations_reward": tf.io.FixedLenFeature([DMLAB_EPISODE_LENGTH], tf.float32),
        "observations_action": tf.io.FixedLenFeature([DMLAB_EPISODE_LENGTH, DMLAB_N_ACTIONS], tf.float32),
        "actions": tf.io.FixedLenFeature([DMLAB_EPISODE_LENGTH], tf.int64),
    }
    ds = tf.data.TFRecordDataset([str(shard_path)], compression_type="GZIP")
    pixels_out, la_out, lr_out, acts_out = [], [], [], []
    taken = 0
    T = DMLAB_EPISODE_LENGTH
    for raw in ds:
        if taken >= max_episodes:
            break
        ex = tf.io.parse_single_example(raw, feature_desc)
        pngs = ex["observations_pixels"].numpy()
        imgs = np.zeros((T, 72, 96, 3), dtype=np.uint8)
        for i in range(T):
            imgs[i] = tf.io.decode_png(pngs[i], channels=3).numpy()
        la_oh = ex["observations_action"].numpy()  # (T, 15)
        la = la_oh.argmax(axis=-1).astype(np.int64)
        lr = ex["observations_reward"].numpy().astype(np.float32)
        acts = ex["actions"].numpy().astype(np.int64)
        pixels_out.append(imgs)
        la_out.append(la)
        lr_out.append(lr)
        acts_out.append(acts)
        taken += 1
    return pixels_out, la_out, lr_out, acts_out


def _cache_path(config_name: str, max_episodes: int) -> Path:
    """Cache key reflects: config + episode count + full-length flag (L301)."""
    tag = f"{config_name}_N{max_episodes}_L{DMLAB_EPISODE_LENGTH}.npz"
    return DEFAULT_CACHE_ROOT / tag


def _feature_cache_path(config_name: str, max_episodes: int, cnn_feature_dim: int, seed: int) -> Path:
    tag = (
        f"{config_name}_N{max_episodes}_L{DMLAB_EPISODE_LENGTH}"
        f"_features_D{cnn_feature_dim}_seed{seed}.npz"
    )
    return DEFAULT_CACHE_ROOT / tag


def build_dmlab_store(max_episodes_per_policy: int = 60,
                      cnn_feature_dim: int = 128,
                      seed: int = 0,
                      device: str = "cpu",
                      **_legacy: dict) -> EpisodeStore:
    """Build an EpisodeStore for the three configured DMLab policies.

    `max_episodes_per_policy` is the per-policy episode cap, so the store
    has `3 * max_episodes_per_policy` total episodes. Episodes are kept at
    the full RLU length (301 steps) with no truncation.
    """
    DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    feat = None

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_meta: List[EpisodeMeta] = []
    ep_counter = 0

    for pid, (policy_name, eps_configs) in enumerate(DMLAB_POLICY_CONFIGS):
        for config_name, split_tag in eps_configs:
            feature_cache = _feature_cache_path(config_name, max_episodes_per_policy, cnn_feature_dim, seed)
            if feature_cache.exists():
                z = np.load(feature_cache, allow_pickle=True)
                feat_list = list(z["states"])
                act_list = list(z["action"])
            else:
                from utils.featurizers import DMLabFeaturizer

                if feat is None:
                    feat = DMLabFeaturizer(
                        n_actions=DMLAB_N_ACTIONS,
                        cnn_feature_dim=cnn_feature_dim,
                        seed=seed,
                        device=device,
                    )
                feat_list = []
                act_list = []
            cache = _cache_path(config_name, max_episodes_per_policy)
            if not feature_cache.exists():
                if cache.exists():
                    z = np.load(cache, allow_pickle=True)
                    pixels_list = list(z["pixels"])
                    la_list = list(z["last_action"])
                    lr_list = list(z["last_reward"])
                    act_list = list(z["action"])
                else:
                    pixels_list, la_list, lr_list, act_list = [], [], [], []
                    shard_id = 0
                    while len(pixels_list) < max_episodes_per_policy and shard_id < 500:
                        shard_path = _download_shard(config_name, shard_id)
                        p, la, lr, ac = _parse_tfrecord_shard(
                            shard_path,
                            max_episodes=max_episodes_per_policy - len(pixels_list),
                        )
                        pixels_list.extend(p)
                        la_list.extend(la)
                        lr_list.extend(lr)
                        act_list.extend(ac)
                        shard_id += 1
                    np.savez_compressed(
                        cache,
                        pixels=np.array(pixels_list, dtype=object),
                        last_action=np.array(la_list, dtype=object),
                        last_reward=np.array(lr_list, dtype=object),
                        action=np.array(act_list, dtype=object),
                    )
                for pixels, la, lr in zip(pixels_list, la_list, lr_list):
                    feat_list.append(feat(pixels, la, lr))
                tmp_cache = feature_cache.with_name(f"{feature_cache.stem}.{os.getpid()}.tmp.npz")
                np.savez_compressed(
                    tmp_cache,
                    states=np.array(feat_list, dtype=object),
                    action=np.array(act_list, dtype=object),
                )
                os.replace(tmp_cache, feature_cache)

            for feats, a in zip(feat_list, act_list):
                all_states.append(np.asarray(feats, dtype=np.float32))
                all_actions.append(np.asarray(a).reshape(-1, 1))  # store as (T, 1) int for uniformity
                all_meta.append(EpisodeMeta(
                    episode_id=ep_counter, policy_id=pid,
                    is_ood=(split_tag == "OOD"),
                    source=f"dmlab/{config_name}",
                    extras={"config": config_name,
                            "policy": policy_name,
                            "predefined_split": split_tag},
                ))
                ep_counter += 1

    state_dim = all_states[0].shape[-1] if all_states else int(cnn_feature_dim + DMLAB_N_ACTIONS + 1)
    return EpisodeStore(
        states=all_states, actions=all_actions, meta=all_meta,
        state_dim=int(state_dim), action_dim=1,
        source="dmlab/seekavoid_arena_01",
        action_kind="discrete", n_actions=DMLAB_N_ACTIONS,
    )
