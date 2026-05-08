"""Regenerate outputs/aggregate_all.md from every aggregate.csv under
outputs/, with these filters:

  - drop any row whose `data` starts with `custom_mujoco_` (mujoco
    micro-environments are noisy individually and clutter the table).
  - drop the *stale* all-policies datasets we explicitly retired
    (`lichess_top3_all_policies`, `lichess_top3_full_all_policies`,
    `synthetic_grf_10x_all_policies`). Keep `fastf1_*_all_*` and the
    new `*_5p_all_policies` ones.
  - drop any base `<data>` row when a corresponding `<data>_all_policies`
    row exists for the same (model, experiment) — the base row
    duplicates the all-policies probe in datasets where all pids are
    used at training time (e.g. droid_lowdim_full_balanced_min300_remove_5col).
  - excludes paths under Trash/ or TRASH_*, plus the HF upload .cache/.

Each unique (data, model, experiment) is rendered as a single row with
mean ± std across seeds (when more than one seed is present).
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
OUT_MD = ROOT / "aggregate_all.md"

# Stale all-policies datasets explicitly retired. These got re-run under
# different names (e.g. *_5p_all_policies) so the old rows are dead weight.
STALE_ALL_POLICIES = {
    "lichess_top3_all_policies",
    "lichess_top3_full_all_policies",
    "lichess_top3_sa16_all_policies",
    "synthetic_grf_10x_all_policies",
    "synthetic_grf_all_policies",
}


def is_trash(p: Path) -> bool:
    parts = p.parts
    if any(part.lower() == "trash" or part.upper().startswith("TRASH") for part in parts):
        return True
    if ".cache" in parts:
        return True
    return False


def populated(r: dict) -> int:
    n = 0
    for k, v in r.items():
        if k in {"run_dir", "data", "model", "experiment", "seed", "__source__"}:
            continue
        if v not in (None, "", "nan", "NaN"):
            try:
                float(v); n += 1
            except (TypeError, ValueError):
                if v.strip(): n += 1
    return n


def fmt_one(x: float) -> str:
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3e}"
    return f"{x:.4g}"


def fmt_seed_agg(values: list[str]) -> str:
    nums = []
    for v in values:
        if v in (None, "", "nan", "NaN"): continue
        try:
            x = float(v)
            if x != x: continue
            nums.append(x)
        except (TypeError, ValueError):
            pass
    if not nums:
        # If no numeric values, fall back to first non-empty string
        for v in values:
            if v not in (None, "", "nan", "NaN"):
                return str(v)
        return "-"
    if len(nums) == 1:
        return fmt_one(nums[0])
    mean = statistics.mean(nums)
    std = statistics.stdev(nums) if len(nums) > 1 else 0.0
    return f"{fmt_one(mean)}±{fmt_one(std)}"


def main() -> None:
    all_files = sorted(ROOT.rglob("aggregate.csv"))
    files = [f for f in all_files if not is_trash(f.relative_to(ROOT))]
    rows: list[dict] = []
    for f in files:
        with f.open() as fp:
            for r in csv.DictReader(fp):
                r["__source__"] = str(f.relative_to(ROOT))
                rows.append(r)

    # --- filter rules ---
    n0 = len(rows)
    # 1) drop custom_mujoco_* rows
    rows = [r for r in rows if not r.get("data", "").startswith("custom_mujoco_")]
    n_after_mujoco = len(rows)
    # 2) drop stale all-policies datasets
    rows = [r for r in rows if r.get("data", "") not in STALE_ALL_POLICIES]
    n_after_stale = len(rows)
    # 3) drop base <data> rows when <data>_all_policies counterpart exists
    #    on same (model, experiment) — they duplicate the all-policies probe.
    have_all_policies: set[tuple[str, str, str]] = {
        (r["data"][: -len("_all_policies")], r["model"], r["experiment"])
        for r in rows
        if r.get("data", "").endswith("_all_policies")
    }
    rows = [
        r for r in rows
        if r.get("data", "").endswith("_all_policies")
        or (r.get("data", ""), r.get("model", ""), r.get("experiment", "")) not in have_all_policies
    ]
    n_after_dedup = len(rows)

    # --- dedupe per-seed first (in case the same (data,model,exp,seed)
    # appears twice in different aggregate sources, prefer the more-
    # populated one)
    by_key = {}
    for r in rows:
        key = (r["data"], r["model"], r["experiment"], r.get("seed", ""))
        if key not in by_key or populated(r) > populated(by_key[key]):
            by_key[key] = r
    rows = list(by_key.values())

    # --- group by (data, model, experiment) and aggregate over seeds
    metric_cols = ["probe_acc", "probe_acc_seen", "knn_acc1", "knn_acc5",
                   "novel_mean_embed_dist", "gen_nmse", "gen_median_se",
                   "gen_acc", "gen_nll", "n_train_episodes",
                   "n_test_episodes_used"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["data"], r["model"], r["experiment"])].append(r)

    rendered = []
    for k, group in grouped.items():
        data, model, exp = k
        seeds = sorted({r.get("seed", "") for r in group})
        n_seeds = len(seeds)
        cell = {"data": data, "model": model, "experiment": exp,
                "n_seeds": str(n_seeds)}
        for c in metric_cols:
            cell[c] = fmt_seed_agg([r.get(c, "") for r in group])
        rendered.append(cell)

    rendered.sort(key=lambda r: (r["data"], r["model"], r["experiment"]))

    cols = ["data", "model", "experiment", "n_seeds"] + metric_cols

    src_counts: dict[str, int] = defaultdict(int)
    for r in by_key.values():
        src_counts[r["__source__"]] += 1

    lines = [
        "# `outputs/aggregate_all.md` — unified results table\n",
        f"_{len(rendered)} unique (data, model, experiment) groups merged from "
        f"{len(files)} `aggregate.csv` files (excludes `Trash/`, `TRASH_*`, `.cache/`, "
        f"`custom_mujoco_*` rows, and stale all-policies datasets {sorted(STALE_ALL_POLICIES)}). "
        f"Base `<data>` rows are dropped when a `<data>_all_policies` counterpart exists "
        f"for the same (model, experiment)._\n",
        f"_Rendered as mean±std across seeds when n_seeds > 1._\n",
        "\nFilter pipeline (rows in -> rows out):\n",
        f"- raw rows from CSVs: {n0}",
        f"- after dropping custom_mujoco_*: {n_after_mujoco}",
        f"- after dropping stale all_policies datasets: {n_after_stale}",
        f"- after dropping base rows with all_policies counterpart: {n_after_dedup}",
        f"- after per-seed dedup across sources: {len(by_key)}",
        f"- after group-by-(data,model,exp) seed aggregation: {len(rendered)}",
        "\nPer-source row counts (after exclusions, before seed aggregation):\n",
    ]
    for src, n in sorted(src_counts.items()):
        lines.append(f"- `{src}` — {n}")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rendered:
        lines.append("| " + " | ".join(r.get(c, "-") for c in cols) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD} ({OUT_MD.stat().st_size/1024:.1f} KB, {len(rendered)} rows)")


if __name__ == "__main__":
    main()
