"""FastF1 stint-level telemetry loader.

Policies are drivers. Episodes are full stints, built by concatenating the
telemetry for consecutive laps with the same FastF1 stint id. OOD is a
held-out circuit/Grand Prix shared across selected drivers; all other
circuits are ID. Use ``shift.kind=predefined_split``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import List
import os

import numpy as np
import pandas as pd

from .base import EpisodeMeta, EpisodeStore


DEFAULT_CACHE_ROOT = Path(os.environ.get("INR_FASTF1_CACHE",
                                          Path.home() / ".cache/INR/fastf1")).expanduser()


def _session_cache_dir(cache_dir: Path) -> Path:
    d = cache_dir / "fastf1_http_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32))
        else:
            vals.append(np.zeros(len(df), dtype=np.float32))
    arr = np.stack(vals, axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def _load_session(year: int, gp: str, cache_dir: Path):
    import fastf1

    fastf1.Cache.enable_cache(str(_session_cache_dir(cache_dir)))
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def _stints_for_session(session, gp: str, min_points: int, max_points: int | None):
    records = []
    laps = session.laps
    for driver in sorted(set(laps["Driver"].dropna().astype(str))):
        driver_laps = laps.pick_drivers(driver)
        if driver_laps.empty or "Stint" not in driver_laps.columns:
            continue
        for stint_id, stint_laps in driver_laps.groupby("Stint"):
            if len(stint_laps) == 0:
                continue
            tele_parts = []
            for _, lap in stint_laps.iterlaps():
                try:
                    tel = lap.get_car_data().add_distance()
                    pos = lap.get_pos_data()
                    if len(pos):
                        tel = tel.merge_channels(pos)
                    tele_parts.append(tel)
                except Exception:
                    continue
            if not tele_parts:
                continue
            tele = pd.concat(tele_parts, ignore_index=True).sort_values("Date", kind="stable")
            if len(tele) < min_points:
                continue
            if max_points is not None and len(tele) > max_points:
                idx = np.linspace(0, len(tele) - 1, int(max_points)).round().astype(int)
                tele = tele.iloc[idx].reset_index(drop=True)
            state_cols = ["Speed", "RPM", "nGear", "Distance", "X", "Y", "Z"]
            action_cols = ["Throttle", "Brake", "nGear", "DRS"]
            states = _safe_float_array(tele, state_cols)
            actions = _safe_float_array(tele, action_cols)
            records.append({
                "driver": driver,
                "gp": gp,
                "stint": int(stint_id) if not pd.isna(stint_id) else -1,
                "states": states,
                "actions": actions,
            })
    return records


def _choose_drivers(
    records: list[dict],
    n_drivers: int,
    ood_gp: str,
    preferred_drivers: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    by_driver_gp = defaultdict(Counter)
    for r in records:
        by_driver_gp[r["driver"]][r["gp"]] += 1
    eligible = [
        d for d, gps in by_driver_gp.items()
        if gps.get(ood_gp, 0) > 0 and sum(v for k, v in gps.items() if k != ood_gp) > 0
    ]
    eligible.sort(key=lambda d: sum(by_driver_gp[d].values()), reverse=True)
    preferred = [str(d) for d in (preferred_drivers or []) if str(d) in eligible]
    if preferred:
        out = []
        for d in preferred + eligible:
            if d not in out:
                out.append(d)
            if len(out) >= n_drivers:
                return out
    if len(eligible) >= n_drivers:
        return eligible[:n_drivers]
    counts = Counter(r["driver"] for r in records)
    return [d for d, _ in counts.most_common(n_drivers)]


def build_fastf1_store(
    seasons: list[int] | tuple[int, ...] = (2023, 2024),
    gps: list[str] | tuple[str, ...] = ("Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco", "British", "Italian"),
    ood_gp: str = "Monaco",
    n_drivers: int = 3,
    preferred_drivers: list[str] | tuple[str, ...] | None = None,
    min_points: int = 32,
    max_points: int | None = 800,
    cache_dir: str | Path = DEFAULT_CACHE_ROOT,
) -> EpisodeStore:
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(x) for x in seasons) + "_" + "_".join(str(g).replace(" ", "") for g in gps)
    cache_path = cache_dir / f"fastf1_stints_{tag}_P{max_points or 'full'}.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        records = list(z["records"])
    else:
        records = []
        for year in seasons:
            for gp in gps:
                try:
                    session = _load_session(int(year), str(gp), cache_dir)
                    records.extend(_stints_for_session(session, str(gp), min_points, max_points))
                except Exception as exc:
                    print(f"[fastf1] skip {year} {gp}: {exc}", flush=True)
        np.savez_compressed(cache_path, records=np.array(records, dtype=object))

    drivers = _choose_drivers(records, int(n_drivers), str(ood_gp), preferred_drivers=preferred_drivers)
    driver_to_pid = {d: i for i, d in enumerate(drivers)}
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    meta: List[EpisodeMeta] = []
    for r in records:
        driver = r["driver"]
        if driver not in driver_to_pid:
            continue
        is_ood = str(r["gp"]) == str(ood_gp)
        states.append(np.asarray(r["states"], dtype=np.float32))
        actions.append(np.asarray(r["actions"], dtype=np.float32))
        meta.append(EpisodeMeta(
            episode_id=len(meta),
            policy_id=driver_to_pid[driver],
            is_ood=is_ood,
            source=f"fastf1/{driver}",
            extras={
                "driver": driver,
                "gp": str(r["gp"]),
                "stint": int(r["stint"]),
                "predefined_split": "OOD" if is_ood else "ID",
                "ood_gp": str(ood_gp),
            },
        ))
    if not states:
        raise RuntimeError("FastF1 builder produced no stint episodes")
    return EpisodeStore(
        states=states,
        actions=actions,
        meta=meta,
        state_dim=int(states[0].shape[-1]),
        action_dim=int(actions[0].shape[-1]),
        source="fastf1",
        action_kind="continuous",
    )
