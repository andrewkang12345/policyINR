"""Move per-run subdirs from outputs/_combined/<sweep>/ into the
domain-tree (outputs/<domain>/<suite>/<run>/).

Some launchers — `run_full_suite.sh`, `run_full_suite_new_datasets.sh`,
`run_lichess_dmlab_v2.sh`, `run_droid_fastf1_full_suite.sh` — sweep
multiple datasets in one go. Each per-run dir is named
`<dataset>__<model>__<experiment>__s<seed>`. We dispatch to a domain
based on the dataset prefix.

Usage:
    python scripts/split_combined_outputs.py outputs/_combined/full_suite
    python scripts/split_combined_outputs.py outputs/_combined/full_suite --dry-run

The aggregate.csv / aggregate.md / launcher.log / *_metrics dirs are
copied (not moved) into each touched domain folder so each lands as a
complete unit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# dataset prefix -> (domain, suite_subdir).
# When a launcher mixes domains, every per-run dir starts with one of
# these prefixes (matched longest-first).
DATASET_PREFIX_MAP = [
    ("synthetic_grf_10x__",    ("synthetic", "baseline_10x")),
    ("synthetic_grf__",        ("synthetic", "baseline_full_suite")),
    ("minari_",                ("mujoco",       "baseline_minari_full_suite")),
    ("custom_mujoco_",         ("mujoco",       "baseline_custom_mujoco")),
    ("dmlab_seekavoid_",       ("dmlabseekavoid", "{sweep_name}")),
    ("lichess_top3_",          ("lichess",      "{sweep_name}")),
    ("droid_lowdim_",          ("droid",        "{sweep_name}")),
    ("fastf1_stint_",          ("fastf1",       "{sweep_name}")),
]


def _classify(run_name: str, sweep_name: str) -> tuple[str, str] | None:
    for prefix, (domain, suite_tpl) in DATASET_PREFIX_MAP:
        if run_name.startswith(prefix):
            return domain, suite_tpl.format(sweep_name=sweep_name)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("combined_dir", type=Path,
                    help="Path to outputs/_combined/<sweep_name>/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    src = args.combined_dir.resolve()
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr); return 2
    sweep_name = src.name
    out_root = args.repo_root / "outputs"

    moved: dict[tuple[str, str], int] = {}
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        cls = _classify(name, sweep_name=sweep_name)
        if cls is None:
            print(f"  ?? skip (no matching prefix): {name}")
            continue
        domain, suite = cls
        dst_parent = out_root / domain / suite
        dst = dst_parent / name
        action = "DRY" if args.dry_run else "MV "
        print(f"  {action} {child.relative_to(args.repo_root)} -> {dst.relative_to(args.repo_root)}")
        if not args.dry_run:
            dst_parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                print(f"    !! destination exists, skipping: {dst}")
                continue
            shutil.move(str(child), str(dst))
        moved[(domain, suite)] = moved.get((domain, suite), 0) + 1

    # Copy sidecar files (aggregate.{csv,md}, launcher.log, *_metrics dirs)
    sidecars = ["aggregate.csv", "aggregate.md", "launcher.log"]
    sidecar_dirs = ["all_policy_metrics", "all_player_metrics"]
    for (domain, suite), n in moved.items():
        dst_parent = out_root / domain / suite
        for sc in sidecars:
            f = src / sc
            if f.exists():
                target = dst_parent / f"{f.stem}_combined_{sweep_name}{f.suffix}"
                if args.dry_run:
                    print(f"  DRY copy {f.relative_to(args.repo_root)} -> {target.relative_to(args.repo_root)}")
                else:
                    shutil.copy2(f, target)
        for sd in sidecar_dirs:
            d = src / sd
            if d.exists():
                target = dst_parent / f"{sd}_combined_{sweep_name}"
                if args.dry_run:
                    print(f"  DRY copytree {d.relative_to(args.repo_root)} -> {target.relative_to(args.repo_root)}")
                elif not target.exists():
                    shutil.copytree(d, target)

    print(f"\n[summary] {sum(moved.values())} runs distributed across "
          f"{len(moved)} (domain, suite) buckets")
    for (domain, suite), n in sorted(moved.items()):
        print(f"  {domain}/{suite}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
