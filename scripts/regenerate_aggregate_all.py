"""Regenerate outputs/aggregate_all.md from every aggregate.csv under
outputs/, excluding any path containing Trash/ or TRASH_*, plus the HF
upload .cache/. Dedupes by (data, model, experiment, seed); on collision
keeps the row with more populated metric fields, ties broken by shorter
source path.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
OUT_MD = ROOT / "aggregate_all.md"


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


def fmt(v: object) -> str:
    if v in (None, "", "nan", "NaN"):
        return "-"
    try:
        x = float(v)
        if x != x: return "-"
        if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.3e}"
        return f"{x:.4g}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    all_files = sorted(ROOT.rglob("aggregate.csv"))
    files = [f for f in all_files if not is_trash(f.relative_to(ROOT))]
    rows = []
    for f in files:
        with f.open() as fp:
            for r in csv.DictReader(fp):
                r["__source__"] = str(f.relative_to(ROOT))
                rows.append(r)

    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("data",""), r.get("model",""), r.get("experiment",""), r.get("seed",""))
        if key not in best:
            best[key] = r
        else:
            old = best[key]
            if (populated(r), -len(r["__source__"])) > (populated(old), -len(old["__source__"])):
                best[key] = r

    deduped = list(best.values())

    def _seed_int(r):
        try: return int(r.get("seed",""))
        except (ValueError, TypeError): return 9999
    deduped.sort(key=lambda r: (r.get("data",""), r.get("model",""), r.get("experiment",""), _seed_int(r)))

    cols = ["data","model","experiment","seed",
            "probe_acc","probe_acc_seen","knn_acc1","knn_acc5",
            "novel_mean_embed_dist","gen_nmse","gen_median_se",
            "gen_acc","gen_nll","n_train_episodes","n_test_episodes_used"]

    src_counts: dict[str, int] = {}
    for r in rows:
        src_counts[r["__source__"]] = src_counts.get(r["__source__"], 0) + 1

    lines = [
        "# `outputs/aggregate_all.md` — unified results table\n",
        f"_{len(deduped)} unique rows merged from {len(files)} `aggregate.csv` files (excludes any path containing `Trash/` or `TRASH_*`, plus the HF upload `.cache/`)._\n",
        "_Dedupe key: `(data, model, experiment, seed)`. On collision the row with more populated metric fields is kept; ties broken by shorter source path._\n",
        "\nPer-source row counts (after exclusions, before dedupe):\n",
    ]
    for src, n in sorted(src_counts.items()):
        lines.append(f"- `{src}` — {n}")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in deduped:
        lines.append("| " + " | ".join(fmt(r.get(c,"")) for c in cols) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD} ({OUT_MD.stat().st_size/1024:.1f} KB, {len(deduped)} rows)")


if __name__ == "__main__":
    main()
