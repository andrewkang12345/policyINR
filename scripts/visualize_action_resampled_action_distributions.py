"""Visualize action-resampled MuJoCo action distributions and write a short note.

Outputs land under:
  outputs/suites/action_resampled_v1/analysis/action_distributions/
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minari
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ENV_IDS = {
    "ant": "inr_mujoco_action_resampled_v1/ant/controlled-v0",
    "halfcheetah": "inr_mujoco_action_resampled_v1/halfcheetah/controlled-v0",
    "hopper": "inr_mujoco_action_resampled_v1/hopper/controlled-v0",
    "humanoid": "inr_mujoco_action_resampled_v1/humanoid/controlled-v0",
    "walker2d": "inr_mujoco_action_resampled_v1/walker2d/controlled-v0",
}

POLICY_ORDER = ["simple", "medium", "expert"]
SPLIT_ORDER = ["ID", "OOD"]
POLICY_COLORS = {
    "simple": "#1f77b4",
    "medium": "#d62728",
    "expert": "#f2c300",
}
SPLIT_COLORS = {
    "ID": "#1f77b4",
    "OOD": "#d62728",
}


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _sample_rows(rng: np.random.Generator, x: np.ndarray, limit: int) -> np.ndarray:
    if len(x) <= limit:
        return x
    idx = rng.choice(len(x), size=limit, replace=False)
    return x[idx]


def _load_env_data(dataset_id: str):
    ds = minari.load_dataset(dataset_id)
    actions_by_policy_split = defaultdict(list)
    episode_mean_actions_by_policy_split = defaultdict(list)
    episode_sample_mean_actions_by_policy_split = defaultdict(list)
    rng = np.random.default_rng(0)
    for ep in ds.iterate_episodes():
        info = ep.infos if isinstance(ep.infos, dict) else {}
        policy = str(_decode_scalar(info.get("policy_name", "unknown")))
        split = str(_decode_scalar(info.get("state_split", "ID"))).upper()
        actions = np.asarray(ep.actions, dtype=np.float32)
        actions_by_policy_split[(policy, split)].append(actions)
        episode_mean_actions_by_policy_split[(policy, split)].append(actions.mean(axis=0))
        k = min(8, len(actions))
        picks = rng.choice(len(actions), size=k, replace=False)
        episode_sample_mean_actions_by_policy_split[(policy, split)].append(actions[picks].mean(axis=0))

    raw_out = {}
    episode_mean_out = {}
    episode_sample_mean_out = {}
    for key, seqs in actions_by_policy_split.items():
        raw_out[key] = np.concatenate(seqs, axis=0)
    for key, seqs in episode_mean_actions_by_policy_split.items():
        episode_mean_out[key] = np.stack(seqs, axis=0)
    for key, seqs in episode_sample_mean_actions_by_policy_split.items():
        episode_sample_mean_out[key] = np.stack(seqs, axis=0)
    return raw_out, episode_mean_out, episode_sample_mean_out


def _balanced_samples(actions_by_policy_split, rng: np.random.Generator, per_policy_per_split: int):
    xs = []
    ys = []
    ss = []
    for policy in POLICY_ORDER:
        for split in SPLIT_ORDER:
            arr = actions_by_policy_split[(policy, split)]
            arr = _sample_rows(rng, arr, per_policy_per_split)
            xs.append(arr)
            ys.extend([policy] * len(arr))
            ss.extend([split] * len(arr))
    return np.concatenate(xs, axis=0), np.asarray(ys), np.asarray(ss)


def _classification_accuracy(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return float(accuracy_score(y_test, pred))


def _policy_radius_and_centroids(actions_by_policy_split):
    policy_arrays = {
        policy: np.concatenate([actions_by_policy_split[(policy, s)] for s in SPLIT_ORDER], axis=0)
        for policy in POLICY_ORDER
    }
    centroids = {policy: arr.mean(axis=0) for policy, arr in policy_arrays.items()}
    radii = {
        policy: float(np.sqrt(np.mean(np.sum((arr - centroids[policy]) ** 2, axis=1))))
        for policy, arr in policy_arrays.items()
    }
    pairwise = {}
    for i, p0 in enumerate(POLICY_ORDER):
        for p1 in POLICY_ORDER[i + 1 :]:
            pairwise[(p0, p1)] = float(np.linalg.norm(centroids[p0] - centroids[p1]))
    return radii, pairwise


def _make_overview(env_name: str, actions_by_policy_split, out_path: Path, seed: int):
    rng = np.random.default_rng(seed)
    x, y, s = _balanced_samples(actions_by_policy_split, rng, per_policy_per_split=1500)
    xy = PCA(n_components=2, random_state=seed).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for split, ax in zip(SPLIT_ORDER, axes):
        mask_split = s == split
        for policy in POLICY_ORDER:
            mask = mask_split & (y == policy)
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=5,
                alpha=0.65,
                c=POLICY_COLORS[policy],
                label=policy,
                linewidths=0,
            )
        ax.set_title(f"{env_name} {split}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle(f"{env_name}: action distribution by policy", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _make_episode_overview(env_name: str, episode_actions_by_policy_split, out_path: Path, seed: int, *, title: str):
    rng = np.random.default_rng(seed)
    x, y, s = _balanced_samples(episode_actions_by_policy_split, rng, per_policy_per_split=1500)
    xy = PCA(n_components=2, random_state=seed).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for split, ax in zip(SPLIT_ORDER, axes):
        mask_split = s == split
        for policy in POLICY_ORDER:
            mask = mask_split & (y == policy)
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=18,
                alpha=0.82,
                c=POLICY_COLORS[policy],
                label=policy,
                linewidths=0,
            )
        ax.set_title(f"{env_name} {split}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _make_policy_plot(env_name: str, policy: str, actions_by_policy_split, out_path: Path, seed: int):
    rng = np.random.default_rng(seed)
    arr_id = _sample_rows(rng, actions_by_policy_split[(policy, "ID")], 2500)
    arr_ood = _sample_rows(rng, actions_by_policy_split[(policy, "OOD")], 2500)
    x = np.concatenate([arr_id, arr_ood], axis=0)
    split = np.asarray(["ID"] * len(arr_id) + ["OOD"] * len(arr_ood))
    xy = PCA(n_components=2, random_state=seed).fit_transform(x)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for split_name in SPLIT_ORDER:
        mask = split == split_name
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=6,
            alpha=0.70,
            c=SPLIT_COLORS[split_name],
            label=split_name,
            linewidths=0,
        )
    ax.set_title(f"{env_name} {policy}: ID vs OOD actions")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(loc="best", frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_note(results: dict, out_path: Path):
    lines = []
    lines.append("Action-resampled MuJoCo action-distribution analysis")
    lines.append("")
    lines.append("What the plots show")
    lines.append("- Each environment overview projects sampled actions into a shared 2D PCA space.")
    lines.append("- Each policy plot shows the action cloud split by ID and OOD action-cluster splits for that policy.")
    lines.append("")
    lines.append("Main conclusion")
    lines.append("- The action-resampled datasets remain strongly policy-separable in action space, which is expected because the split is defined directly from policy action structure.")
    lines.append("")
    lines.append("Quantitative evidence")
    for env_name, stats in results.items():
        lines.append(
            f"{env_name}: action-only linear accuracy={stats['action_acc']:.3f}, "
            f"mean pairwise centroid distance={stats['mean_pairwise_centroid_distance']:.3f}, "
            f"mean within-policy radius={stats['mean_policy_radius']:.3f}"
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append("- Unlike the state-resampled construction, this dataset family intentionally partitions the common state bag by each policy's action clusters.")
    lines.append("- That means policy identity is preserved more directly in the action marginals.")
    lines.append("- High probe accuracy here is therefore structurally expected, not a surprise from residual state leakage.")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs/_archive/mujoco_suites/action_resampled_v1",
    )
    ap.add_argument(
        "--dataset-prefix",
        type=str,
        default="inr_mujoco_action_resampled_v1",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = args.suite_root / "analysis" / "action_distributions"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    results = {}

    env_ids = {
        env_name: f"{args.dataset_prefix}/{env_name}/controlled-v0"
        for env_name in ENV_IDS
    }

    for env_name, dataset_id in env_ids.items():
        env_root = out_root / env_name
        env_root.mkdir(parents=True, exist_ok=True)

        actions_by_policy_split, episode_mean_actions_by_policy_split, episode_sample_mean_actions_by_policy_split = _load_env_data(dataset_id)
        _make_overview(env_name, actions_by_policy_split, env_root / "overview.png", args.seed)
        _make_episode_overview(
            env_name,
            episode_mean_actions_by_policy_split,
            env_root / "overview_episode_mean.png",
            args.seed,
            title=f"{env_name}: episode-mean action distribution by policy",
        )
        _make_episode_overview(
            env_name,
            episode_sample_mean_actions_by_policy_split,
            env_root / "overview_probe_like_8sample_mean.png",
            args.seed,
            title=f"{env_name}: 8-sample episode-mean action distribution by policy",
        )
        for policy in POLICY_ORDER:
            _make_policy_plot(env_name, policy, actions_by_policy_split, env_root / f"{policy}.png", args.seed)

        x, y, _ = _balanced_samples(actions_by_policy_split, rng, per_policy_per_split=1200)
        radii, pairwise = _policy_radius_and_centroids(actions_by_policy_split)
        results[env_name] = {
            "action_acc": _classification_accuracy(x, y, args.seed),
            "mean_policy_radius": float(np.mean(list(radii.values()))),
            "mean_pairwise_centroid_distance": float(np.mean(list(pairwise.values()))),
        }

    _write_note(results, out_root / "why_actions_still_separate_policies.txt")
    print(f"Wrote plots and analysis to {out_root}")


if __name__ == "__main__":
    main()
