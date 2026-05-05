"""Visualize state-resampled MuJoCo state distributions and write a short analysis.

Outputs land under:
  outputs/suites/state_resampled_*/analysis/state_distributions/

For each environment, the script writes:
  - overview.png: all policies together in a shared PCA projection
  - <policy>.png: ID/OOD projection for one policy

It also writes:
  - why_probe_accuracy_is_high.txt
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
    "ant": "inr_mujoco_state_resampled_v1/ant/controlled-v0",
    "halfcheetah": "inr_mujoco_state_resampled_v1/halfcheetah/controlled-v0",
    "hopper": "inr_mujoco_state_resampled_v1/hopper/controlled-v0",
    "humanoid": "inr_mujoco_state_resampled_v1/humanoid/controlled-v0",
    "walker2d": "inr_mujoco_state_resampled_v1/walker2d/controlled-v0",
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
    obs_by_policy_split = defaultdict(list)
    actions_by_policy = defaultdict(list)
    for ep in ds.iterate_episodes():
        info = ep.infos if isinstance(ep.infos, dict) else {}
        policy = str(_decode_scalar(info.get("policy_name", "unknown")))
        split = str(_decode_scalar(info.get("state_split", "ID"))).upper()
        obs = np.asarray(ep.observations[:-1], dtype=np.float32)
        acts = np.asarray(ep.actions, dtype=np.float32)
        obs_by_policy_split[(policy, split)].append(obs)
        actions_by_policy[policy].append(acts)

    out = {}
    for key, seqs in obs_by_policy_split.items():
        out[key] = np.concatenate(seqs, axis=0)
    act_out = {}
    for key, seqs in actions_by_policy.items():
        act_out[key] = np.concatenate(seqs, axis=0)
    return out, act_out


def _balanced_state_samples(obs_by_policy_split, rng: np.random.Generator, per_policy_per_split: int):
    xs = []
    ys = []
    ss = []
    for policy in POLICY_ORDER:
        for split in SPLIT_ORDER:
            arr = obs_by_policy_split[(policy, split)]
            arr = _sample_rows(rng, arr, per_policy_per_split)
            xs.append(arr)
            ys.extend([policy] * len(arr))
            ss.extend([split] * len(arr))
    return np.concatenate(xs, axis=0), np.asarray(ys), np.asarray(ss)


def _balanced_action_samples(actions_by_policy, rng: np.random.Generator, per_policy: int):
    xs = []
    ys = []
    for policy in POLICY_ORDER:
        arr = _sample_rows(rng, actions_by_policy[policy], per_policy)
        xs.append(arr)
        ys.extend([policy] * len(arr))
    return np.concatenate(xs, axis=0), np.asarray(ys)


def _classification_accuracy(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return float(accuracy_score(y_test, pred))


def _policy_radius_and_centroids(obs_by_policy_split):
    policy_arrays = {
        policy: np.concatenate([obs_by_policy_split[(policy, s)] for s in SPLIT_ORDER], axis=0)
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
    return centroids, radii, pairwise


def _make_overview(env_name: str, obs_by_policy_split, out_path: Path, seed: int):
    rng = np.random.default_rng(seed)
    x, y, s = _balanced_state_samples(obs_by_policy_split, rng, per_policy_per_split=1500)
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(x)

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
    fig.suptitle(f"{env_name}: observation distribution by policy", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _make_policy_plot(env_name: str, policy: str, obs_by_policy_split, out_path: Path, seed: int):
    rng = np.random.default_rng(seed)
    arr_id = _sample_rows(rng, obs_by_policy_split[(policy, "ID")], 2500)
    arr_ood = _sample_rows(rng, obs_by_policy_split[(policy, "OOD")], 2500)
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
    ax.set_title(f"{env_name} {policy}: ID vs OOD")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(loc="best", frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_analysis(results: dict, out_path: Path):
    lines = []
    lines.append("Step-resampled MuJoCo probe-accuracy analysis")
    lines.append("")
    lines.append("What the plots show")
    lines.append("- Each environment overview projects the sampled observation states into a shared 2D PCA space.")
    lines.append("- Each policy plot projects one policy's ID and OOD observation clouds.")
    lines.append("- These are observation distributions, not rollout trajectories.")
    lines.append("")
    lines.append("Main conclusion")
    lines.append(
        "- The step-resampled construction mostly removed policy-identifying information from the observation distribution itself, "
        "but the downstream representation is still trained on state-action histories, so actions remain strongly policy-identifying."
    )
    lines.append("")
    lines.append("Quantitative evidence")
    for env_name, stats in results.items():
        lines.append(f"{env_name}:")
        lines.append(
            f"  state-only linear accuracy={stats['state_acc']:.3f}, "
            f"ID-only={stats['state_acc_id']:.3f}, OOD-only={stats['state_acc_ood']:.3f}, "
            f"action-only={stats['action_acc']:.3f}"
        )
        lines.append(
            f"  mean pairwise centroid distance={stats['mean_pairwise_centroid_distance']:.3f}, "
            f"mean within-policy radius={stats['mean_policy_radius']:.3f}"
        )
    lines.append("")
    lines.append("Why the probe accuracy is still high")
    lines.append(
        "- In the step-resampled dataset, each policy is queried on mostly matched state distributions, so state-only classification is near chance in several environments."
    )
    lines.append(
        "- But each checkpoint still produces policy-specific actions on those states. The training pipeline consumes state-action histories, not states alone."
    )
    lines.append(
        "- A representation that can predict or reconstruct actions will naturally preserve policy identity even when the state marginals are aligned."
    )
    lines.append(
        "- Finite-sample effects still leave some residual state mismatch, but that is not the dominant reason for the near-1.0 probe results."
    )
    lines.append("")
    lines.append("How to fix it")
    lines.append(
        "- If the goal is to remove state leakage specifically, force every policy to use the exact same state bag for train/val/test and for both ID and OOD."
    )
    lines.append(
        "- If the goal is to make policy identification genuinely hard, you need to break the direct action signal too. Two practical options:"
    )
    lines.append(
        "  1. Evaluate representations built from states only, or from masked/noised actions, so the probe cannot read policy identity directly from control outputs."
    )
    lines.append(
        "  2. Replace the policy-ID probe with a task-relevant probe, such as return prediction, dynamics prediction, transfer, or OOD action prediction."
    )
    lines.append(
        "- A stronger dataset construction would use a single canonical pool of sampled states per environment, then replay that exact pool for every checkpoint."
    )
    lines.append(
        "- If you still want action-conditioned sequences, compare policies only on matched state-action residuals, for example action deltas relative to a shared reference controller."
    )
    lines.append("")
    lines.append("Practical next step")
    lines.append(
        "- The next clean experiment is a matched-state dataset where each policy receives the exact same sampled state tensor in the same episode order, plus a state-only ablation of the encoder/probe."
    )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs/_archive/mujoco_suites/state_resampled_v1",
    )
    ap.add_argument(
        "--dataset-prefix",
        type=str,
        default="inr_mujoco_state_resampled_v1",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = args.suite_root / "analysis" / "state_distributions"
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

        obs_by_policy_split, actions_by_policy = _load_env_data(dataset_id)

        _make_overview(env_name, obs_by_policy_split, env_root / "overview.png", args.seed)
        for policy in POLICY_ORDER:
            _make_policy_plot(env_name, policy, obs_by_policy_split, env_root / f"{policy}.png", args.seed)

        x_state, y_state, split_state = _balanced_state_samples(
            obs_by_policy_split, rng, per_policy_per_split=1200
        )
        x_action, y_action = _balanced_action_samples(actions_by_policy, rng, per_policy=2400)
        _, radii, pairwise = _policy_radius_and_centroids(obs_by_policy_split)

        results[env_name] = {
            "state_acc": _classification_accuracy(x_state, y_state, args.seed),
            "state_acc_id": _classification_accuracy(
                x_state[split_state == "ID"], y_state[split_state == "ID"], args.seed
            ),
            "state_acc_ood": _classification_accuracy(
                x_state[split_state == "OOD"], y_state[split_state == "OOD"], args.seed
            ),
            "action_acc": _classification_accuracy(x_action, y_action, args.seed),
            "mean_policy_radius": float(np.mean(list(radii.values()))),
            "mean_pairwise_centroid_distance": float(np.mean(list(pairwise.values()))),
        }

    _write_analysis(results, out_root / "why_probe_accuracy_is_high.txt")
    print(f"Wrote plots and analysis to {out_root}")


if __name__ == "__main__":
    main()
