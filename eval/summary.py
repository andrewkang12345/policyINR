"""Aggregate `summary.json` files from many runs into a single table.

- CSV dump of every run (raw fields).
- Markdown summary: mean ± std across seeds, plus median, clipped
  mean (winsorize at the 95th percentile of gen_mse across all runs),
  and a count of degenerate runs (non-finite gen_mse OR
  finite_fraction < 1.0).

Usage (CLI):
    python -m eval.summary outputs/full_suite --out outputs/full_suite/aggregate.csv --md ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

FIELDS = [
    "data", "model", "experiment", "seed",
    "probe_acc", "probe_acc_seen", "knn_acc1", "knn_acc5", "novel_mean_embed_dist",
    "gen_mse", "gen_nmse", "gen_median_se", "gen_rmse",
    "gen_acc", "gen_nll",
    "finite_fraction", "target_var",
    "n_train_episodes", "n_test_episodes", "n_test_episodes_used", "n_test_ood",
]


def _collect(root: Path) -> List[Dict]:
    rows = []
    for p in root.rglob("summary.json"):
        try:
            with p.open() as f:
                s = json.load(f)
        except Exception:
            continue
        row = {"run_dir": str(p.parent)}
        for k in ("data", "model", "experiment", "seed"):
            row[k] = s.get(k)
        for k in ("n_train_episodes", "n_test_episodes"):
            row[k] = s.get(k)
        e = s.get("eval", {})
        for k in ("probe_acc", "probe_acc_seen", "novel_mean_embed_dist",
                  "knn_acc1", "knn_acc5",
                  "gen_mse", "gen_nmse", "gen_median_se", "gen_rmse",
                  "gen_acc", "gen_nll",
                  "finite_fraction", "target_var",
                  "n_test_episodes_used", "n_test_ood"):
            row[k] = e.get(k)
        row["probe_only"] = bool(e.get("probe_only", False))
        rows.append(row)
    return rows


def _isnan(x):
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _finite_values(rs, key):
    vals = []
    for r in rs:
        v = r.get(key)
        if v is None or _isnan(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            vals.append(fv)
    return np.array(vals, dtype=float)


def _mstd(v):
    if v.size == 0:
        return "-"
    if v.size == 1:
        return f"{v.mean():.3f}"
    return f"{v.mean():.3f}±{v.std():.3f}"


def aggregate_runs(root: Path, out_csv: Path | None = None, out_md: Path | None = None):
    rows = _collect(Path(root))
    if out_csv is not None:
        import csv
        keys = ["run_dir"] + FIELDS
        with Path(out_csv).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    grouped: Dict[tuple, List[Dict]] = {}
    for r in rows:
        key = (r.get("data"), r.get("model"), r.get("experiment"))
        grouped.setdefault(key, []).append(r)

    # Degenerate = the run produced no usable headline metric:
    # * continuous: non-finite gen_mse OR finite_fraction < 1.0
    # * discrete  : non-finite gen_acc (nan accuracy means no samples)
    # A run with finite gen_acc but NaN gen_mse (discrete by design) is fine.
    def degenerate_count(rs):
        n = 0
        for r in rs:
            if r.get("probe_only"):
                continue
            ff = r.get("finite_fraction")
            mse = r.get("gen_mse")
            acc = r.get("gen_acc")
            if ff is not None and ff < 1.0 - 1e-9:
                n += 1
                continue
            def finite(x):
                try:
                    return x is not None and math.isfinite(float(x))
                except Exception:
                    return False
            if finite(mse) or finite(acc):
                continue
            n += 1
        return n

    lines = [
        f"# Aggregate ({len(rows)} runs)",
        "",
        "Metrics: `probe_acc` = strict train-split probe accuracy on held-out test episodes; "
        "`probe_acc_seen` = same probe restricted to training-policy labels; "
        "`knn_acc1`/`knn_acc5` = cosine kNN policy accuracy using train embeddings as the index; "
        "`gen_nmse` = MSE / target_var, scale-free (0 = perfect, 1 ≈ mean-predictor baseline); "
        "`gen_median_se` = median per-sample squared error; "
        "`deg` = degenerate runs (non-finite gen or partial finite fraction).",
        "",
        "| data | model | experiment | n | deg | probe_acc | probe_acc_seen | knn_acc@1 | knn_acc@5 | gen_nmse | gen_median_se | gen_acc | gen_nll |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (d, m, e), rs in sorted(grouped.items(), key=lambda x: str(x[0])):
        lines.append(
            f"| {d} | {m} | {e} | {len(rs)} | {degenerate_count(rs)} | "
            f"{_mstd(_finite_values(rs, 'probe_acc'))} | "
            f"{_mstd(_finite_values(rs, 'probe_acc_seen'))} | "
            f"{_mstd(_finite_values(rs, 'knn_acc1'))} | "
            f"{_mstd(_finite_values(rs, 'knn_acc5'))} | "
            f"{_mstd(_finite_values(rs, 'gen_nmse'))} | "
            f"{_mstd(_finite_values(rs, 'gen_median_se'))} | "
            f"{_mstd(_finite_values(rs, 'gen_acc'))} | "
            f"{_mstd(_finite_values(rs, 'gen_nll'))} |"
        )
    table = "\n".join(lines)
    if out_md is not None:
        Path(out_md).write_text(table)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    ap.add_argument("--md", type=str, default=None, help="Output markdown path")
    args = ap.parse_args()
    table = aggregate_runs(Path(args.root), Path(args.out) if args.out else None,
                           Path(args.md) if args.md else None)
    print(table)


if __name__ == "__main__":
    main()
