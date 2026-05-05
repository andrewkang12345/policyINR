"""Render 2 ID + 2 OOD sequences (3 screenshots each) for every benchmark
dataset family: Minari MuJoCo (hopper, halfcheetah, walker2d, ant, humanoid),
DMLab seekavoid, and Lichess top-3.

ID = policies 0 & 1 in each family; OOD = policy 2. One screenshot per
sequence is taken at t=0, t=T/2, t=T-1.

Outputs land under <repo>/outputs/PLOTS/ with stable, sortable filenames:
  <family>_<env>_<split>_<policy>_ep<idx>_f<i>_t<step>.png
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "outputs" / "PLOTS"
PLOTS.mkdir(parents=True, exist_ok=True)

# Subfolder roots per family / split. All PNGs go under these.
MUJOCO_DIR = PLOTS / "mujoco"
DMLAB_DIR  = PLOTS / "dmlab"
LICHESS_DIR = PLOTS / "lichess"

# How many sequences per (env/player, split) to render. Doubled from the
# first pass so we span more state samples.
N_PER_SPLIT = 4

# ---------------- MuJoCo ---------------------------------------------------

MUJOCO_ENVS = ["hopper", "halfcheetah", "walker2d", "ant", "humanoid"]

# Per-env: number of leading qpos dims dropped from the observation.
# gymnasium Hopper/HalfCheetah/Walker2d drop qpos[0] (root x).
# gymnasium Ant/Humanoid drop qpos[0:2] (root x, y).
_QPOS_EXCLUDE = {
    "hopper": 1, "halfcheetah": 1, "walker2d": 1,
    "ant": 2, "humanoid": 2,
}


def _three_indices(n: int) -> list[int]:
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    return [0, n // 2, n - 1]


def _mid_index(n: int) -> int:
    """Representative timestep for the (state, next-state) pair.

    We pick n//2 so the snapshot is past the initial reset and not at
    the very last frame (which would have no "next state" for ID-style
    rollouts and is visually boring anyway).
    """
    if n <= 1:
        return 0
    return n // 2


def _meta_scalar(meta, key, default=None):
    v = meta.get(key, default)
    if v is None:
        return default
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        if len(v) == 0:
            return default
        v = v[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    return v


def render_mujoco():
    """Per env, write:

    ID (policy-specific, real Minari rollouts):
      mujoco_<env>_ID_<policy>_ep<ei>_t<t>_{state,nextstate}.png
        One per policy in each of 2 ID episodes.

    OOD (shared state sequence across policies, synthetic):
      mujoco_<env>_OOD_ssid<s>_t<t>_state.png
        The sampled state at step t. Identical across policies by
        construction, so a single file per (ssid, t) suffices.
      mujoco_<env>_OOD_ssid<s>_<policy>_t<t>_nextstate.png
        The state each policy transitions to when it executes its own
        action from that same shared starting state.
    """
    import minari

    for env_key in MUJOCO_ENVS:
        ds_id = f"inr_mujoco_action_resampled_v4/{env_key}/controlled-v0"
        try:
            ds = minari.load_dataset(ds_id)
        except Exception as exc:
            print(f"[mujoco] SKIP {ds_id}: {exc}", flush=True)
            continue

        # Scan episodes once. We only need 2 ID episodes total and the 3
        # OOD episodes for each of 2 chosen shared_sequence_ids. 384 eps
        # per policy, first policy's 256 are ID then 128 OOD; then policy
        # 1's 256 ID then 128 OOD; ditto policy 2. So scanning the whole
        # 1152 episode index is cheap enough.
        id_by_policy: dict[str, list[int]] = {}
        ood_by_ssid: dict[int, dict[str, int]] = {}
        eps_cache: dict[int, object] = {}
        for i, ep in enumerate(ds.iterate_episodes()):
            meta = ep.infos
            split = _meta_scalar(meta, "state_split")
            pname = _meta_scalar(meta, "policy_name", default="?")
            if split == "ID":
                id_by_policy.setdefault(pname, []).append(i)
            elif split == "OOD":
                ssid = _meta_scalar(meta, "shared_sequence_id", default=None)
                if ssid is None:
                    continue
                ood_by_ssid.setdefault(int(ssid), {})[pname] = i
            eps_cache[i] = ep

        # Pick N_PER_SPLIT ID episodes, spanning policies in round-robin.
        id_cases: list[tuple[str, int]] = []
        policy_round = sorted(id_by_policy.keys())
        per_policy_cursor = {p: 0 for p in policy_round}
        while len(id_cases) < N_PER_SPLIT and policy_round:
            progressed = False
            for pname in list(policy_round):
                if len(id_cases) >= N_PER_SPLIT:
                    break
                cur = per_policy_cursor[pname]
                if cur < len(id_by_policy[pname]):
                    id_cases.append((pname, id_by_policy[pname][cur]))
                    per_policy_cursor[pname] = cur + 1
                    progressed = True
            if not progressed:
                break

        # Pick N_PER_SPLIT OOD shared_sequence_ids that have episodes for
        # all 3 policies — so the same state is visible across policies.
        ood_ssids: list[int] = []
        for ssid in sorted(ood_by_ssid.keys()):
            if len(ood_by_ssid[ssid]) >= 3:
                ood_ssids.append(ssid)
            if len(ood_ssids) >= N_PER_SPLIT:
                break
        if len(ood_ssids) < N_PER_SPLIT:
            print(f"[mujoco] WARN {env_key}: only {len(ood_ssids)} full-coverage OOD ssids", flush=True)

        try:
            env = ds.recover_environment(render_mode="rgb_array")
            data = env.unwrapped.data
            n_qpos = int(data.qpos.shape[0])
            n_qvel = int(data.qvel.shape[0])
        except Exception as exc:
            print(f"[mujoco] ENV FAIL {ds_id}: {exc}", flush=True)
            continue

        q_excl = _QPOS_EXCLUDE[env_key]

        def _obs_to_qp(o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            qpos = np.zeros(n_qpos, dtype=np.float64)
            qvel = np.zeros(n_qvel, dtype=np.float64)
            qpos[q_excl:] = o[: n_qpos - q_excl]
            qvel[:]        = o[n_qpos - q_excl : n_qpos - q_excl + n_qvel]
            return qpos, qvel

        def _render_state_then_step(o: np.ndarray, a: np.ndarray) -> tuple:
            qpos, qvel = _obs_to_qp(o)
            env.unwrapped.set_state(qpos, qvel)
            fs = env.render()
            env.unwrapped.set_state(qpos, qvel)
            env.step(a)
            fn = env.render()
            return fs, fn

        id_dir = MUJOCO_DIR / "id" / env_key
        ood_dir = MUJOCO_DIR / "ood" / env_key
        id_dir.mkdir(parents=True, exist_ok=True)
        ood_dir.mkdir(parents=True, exist_ok=True)

        try:
            env.reset(seed=0)

            # ID cases: one (state, nextstate) pair per chosen episode.
            for policy, ep_idx in id_cases:
                ep = eps_cache.get(ep_idx)
                if ep is None:
                    continue
                obs = np.asarray(ep.observations, dtype=np.float64)
                acts = np.asarray(ep.actions, dtype=np.float64)
                T = min(obs.shape[0] - 1, acts.shape[0])
                if T <= 0:
                    continue
                t = _mid_index(T)
                try:
                    fs, fn = _render_state_then_step(obs[t], acts[t])
                except Exception as exc:
                    print(f"[mujoco] ID render fail {env_key} ep{ep_idx} t{t}: {exc}", flush=True)
                    continue
                base = f"{policy}_ep{ep_idx}_t{t:04d}"
                iio.imwrite(id_dir / f"{base}_state.png", fs)
                iio.imwrite(id_dir / f"{base}_nextstate.png", fn)
                print(f"[mujoco] id/{env_key}/{base}_{{state,nextstate}}.png", flush=True)

            # OOD cases: one shared-state image per ssid, one nextstate
            # image per policy for that same state.
            for ssid in ood_ssids:
                per_policy = ood_by_ssid[ssid]
                rep_pname = sorted(per_policy.keys())[0]
                rep_ep = eps_cache[per_policy[rep_pname]]
                rep_obs = np.asarray(rep_ep.observations, dtype=np.float64)
                T = rep_obs.shape[0] - 1
                if T <= 0:
                    continue
                t = _mid_index(T)
                qpos, qvel = _obs_to_qp(rep_obs[t])
                try:
                    env.unwrapped.set_state(qpos, qvel)
                    fs = env.render()
                except Exception as exc:
                    print(f"[mujoco] OOD state fail {env_key} ssid{ssid}: {exc}", flush=True)
                    continue
                state_base = f"ssid{ssid:04d}_t{t:04d}"
                iio.imwrite(ood_dir / f"{state_base}_state.png", fs)
                print(f"[mujoco] ood/{env_key}/{state_base}_state.png", flush=True)

                for policy in sorted(per_policy.keys()):
                    ep = eps_cache[per_policy[policy]]
                    acts = np.asarray(ep.actions, dtype=np.float64)
                    if t >= acts.shape[0]:
                        continue
                    try:
                        env.unwrapped.set_state(qpos, qvel)
                        env.step(acts[t])
                        fn = env.render()
                    except Exception as exc:
                        print(f"[mujoco] OOD nextstate fail {env_key} ssid{ssid} {policy}: {exc}", flush=True)
                        continue
                    next_name = f"ssid{ssid:04d}_{policy}_t{t:04d}_nextstate.png"
                    iio.imwrite(ood_dir / next_name, fn)
                    print(f"[mujoco] ood/{env_key}/{next_name}", flush=True)
        finally:
            env.close()


# ---------------- DMLab ----------------------------------------------------

DMLAB_CACHE = ROOT / ".cache" / "inr" / "dmlab"
DMLAB_POLICIES = [
    ("ID", "snapshot_0_eps_0.0", 0),
    ("ID", "snapshot_1_eps_0.0", 1),
    ("OOD", "snapshot_0_eps_0.25", 2),
]


def _load_dmlab_policy(policy_name: str):
    # Prefer the L32-128 cache (longer episodes, 20 or 60 eps).
    candidates = [
        DMLAB_CACHE / f"{policy_name}_N60_L32-128.npz",
        DMLAB_CACHE / f"{policy_name}_N20_L32-128.npz",
        DMLAB_CACHE / f"{policy_name}_N60_L301.npz",
    ]
    for p in candidates:
        if p.exists():
            return np.load(p, allow_pickle=True), p.name
    return None, None


def render_dmlab():
    # N_PER_SPLIT ID sequences (half from each of the 2 ID policies),
    # N_PER_SPLIT OOD sequences (all from the single OOD policy).
    id_policies = ("snapshot_0_eps_0.0", "snapshot_1_eps_0.0")
    ood_policy = "snapshot_0_eps_0.25"
    cases: list[tuple[str, str, int]] = []
    half = N_PER_SPLIT // 2 or 1
    for p in id_policies:
        for k in range(half):
            cases.append(("ID", p, k))
    for k in range(N_PER_SPLIT):
        cases.append(("OOD", ood_policy, k))

    for split, policy, ep_idx in cases:
        blob, src = _load_dmlab_policy(policy)
        if blob is None:
            print(f"[dmlab] SKIP {policy}: no cache found", flush=True)
            continue
        pixels = blob["pixels"]
        if ep_idx >= len(pixels):
            print(f"[dmlab] SKIP {policy} ep{ep_idx}: only {len(pixels)} eps", flush=True)
            continue
        ep_frames = np.asarray(pixels[ep_idx], dtype=np.uint8)
        T = len(ep_frames)
        if T < 2:
            continue
        t = _mid_index(T - 1)
        out_dir = DMLAB_DIR / split.lower() / policy
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"ep{ep_idx}_t{t:04d}"
        iio.imwrite(out_dir / f"{base}_state.png", ep_frames[t])
        iio.imwrite(out_dir / f"{base}_nextstate.png", ep_frames[t + 1])
        print(f"[dmlab] {split.lower()}/{policy}/{base}_{{state,nextstate}}.png (src={src})", flush=True)


# ---------------- Lichess --------------------------------------------------

LICHESS_PGN_DIR = ROOT / ".cache" / "lichess" / "pgn"
LICHESS_POLICIES = [
    ("ID", "lance5500", 0),
    ("ID", "Zhigalko_Sergei", 1),
    ("OOD", "penguingim1", 2),
]


def _collect_player_games(pgn_path: Path, player: str,
                          min_plies: int, max_plies: int,
                          tracked_player_only: bool,
                          max_games: int):
    """Yield games for `player`'s PGN file, matching build_lichess_store's
    filter: only the tracked player's own half-moves count, and a game is
    kept iff `min_plies <= len(moves) <= max_plies` once filtered.
    Returns a list of (game, positions, moves) — positions is the list
    of chess.Board snapshots, one per kept half-move (board *before* the
    player's move).
    """
    import chess
    import chess.pgn

    kept: list = []
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as fh:
        while len(kept) < max_games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            target = None
            if tracked_player_only:
                white = (game.headers.get("White") or "").lower()
                black = (game.headers.get("Black") or "").lower()
                tp = player.lower()
                if tp == white:
                    target = chess.WHITE
                elif tp == black:
                    target = chess.BLACK
                else:
                    continue
            board = game.board()
            positions: list = []
            moves: list = []
            move_objs: list = []
            for mv in game.mainline_moves():
                if target is None or board.turn == target:
                    positions.append(board.copy())
                    moves.append(mv.uci())
                    move_objs.append(mv)
                    if len(moves) >= max_plies:
                        break
                board.push(mv)
            if len(moves) < min_plies:
                continue
            kept.append((game, positions, move_objs))
    return kept


def render_lichess():
    try:
        import chess
        import chess.pgn
        import chess.svg
        import cairosvg
    except Exception as exc:
        print(f"[lichess] SKIP (missing deps): {exc}", flush=True)
        return

    # Load the cached lichess store and apply the same shared_region shift
    # the training pipeline uses, so the ID/OOD labels here match what the
    # models actually see during training.
    sys.path.insert(0, str(ROOT))
    from omegaconf import OmegaConf
    from data.lichess import build_lichess_store
    from data.shifts import SHIFTS

    cfg_path = ROOT / "configs" / "data" / "lichess_top3.yaml"
    cfg = OmegaConf.load(cfg_path)
    players = list(cfg.players)
    min_plies = int(cfg.min_plies)
    max_plies = int(cfg.max_plies)
    max_games = int(cfg.max_games_per_player)
    tracked = bool(cfg.tracked_player_only)
    pgn_dir = Path(os.path.expandvars(os.path.expanduser(str(cfg.pgn_dir))))
    if not pgn_dir.exists():
        # config points at $HOME/.cache/INR/lichess/pgn but our cache lives
        # under /mnt/data/INR/.cache/lichess/pgn — fall back to that.
        alt = ROOT / ".cache" / "lichess" / "pgn"
        if alt.exists():
            pgn_dir = alt
        else:
            print(f"[lichess] SKIP: pgn dir {pgn_dir} not found", flush=True)
            return

    store = build_lichess_store(
        pgn_dir=pgn_dir, players=players,
        max_games_per_player=max_games,
        min_plies=min_plies, max_plies=max_plies,
        tracked_player_only=tracked,
    )
    shift_fn = SHIFTS.get("shared_region")
    is_ood = shift_fn(store, ood_fraction=0.3, seed=0)
    fallback = getattr(shift_fn, "last_fallback", "?")
    print(f"[lichess] shift fallback = {fallback}; "
          f"ID={is_ood.count(False)} OOD={is_ood.count(True)}", flush=True)

    # Map episode index -> (player, per-player game index) by replaying the
    # same construction order as build_lichess_store (per_player dict in
    # insertion order of `players`).
    ep_by_player: dict[str, list[int]] = {p: [] for p in players}
    for ei, meta in enumerate(store.meta):
        ep_by_player[players[int(meta.policy_id)]].append(ei)

    # Pick N_PER_SPLIT ID + N_PER_SPLIT OOD episodes per player.
    plan: list[tuple[str, str, int, int]] = []  # (split, player, ep_idx, game_idx)
    for player in players:
        player_eps = ep_by_player[player]
        id_eps = [(e, player_eps.index(e)) for e in player_eps if not is_ood[e]]
        ood_eps = [(e, player_eps.index(e)) for e in player_eps if is_ood[e]]
        for e, gi in id_eps[:N_PER_SPLIT]:
            plan.append(("ID", player, e, gi))
        for e, gi in ood_eps[:N_PER_SPLIT]:
            plan.append(("OOD", player, e, gi))

    # Re-parse each player's games once, then render selected ones.
    games_cache: dict[str, list] = {}
    for split, player, ep_idx, game_idx in plan:
        if player not in games_cache:
            games_cache[player] = _collect_player_games(
                pgn_dir / f"{player}.pgn", player,
                min_plies, max_plies, tracked, max_games,
            )
        kept = games_cache[player]
        if game_idx >= len(kept):
            print(f"[lichess] SKIP {player} ep{ep_idx} g{game_idx}: "
                  f"only {len(kept)} games re-parsed", flush=True)
            continue
        _, positions, move_objs = kept[game_idx]
        if not positions or not move_objs:
            continue
        t = _mid_index(len(positions))
        if t >= len(move_objs):
            t = len(move_objs) - 1
        state_pos = positions[t]
        mv = move_objs[t]
        # subsequent state: position after the player plays `mv`.
        next_pos = state_pos.copy()
        try:
            next_pos.push(mv)
        except Exception as exc:
            print(f"[lichess] push fail {player} g{game_idx} t{t}: {exc}", flush=True)
            continue

        out_dir = LICHESS_DIR / split.lower() / player
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"ep{ep_idx}_g{game_idx}_t{t:04d}"
        for tag, pos, last in (("state", state_pos, None), ("nextstate", next_pos, mv)):
            svg = chess.svg.board(
                pos, size=480, lastmove=last, coordinates=True,
            )
            png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            (out_dir / f"{base}_{tag}.png").write_bytes(png)
            print(f"[lichess] {split.lower()}/{player}/{base}_{tag}.png", flush=True)


# ---------------- Main -----------------------------------------------------

def main():
    kinds = sys.argv[1:] or ["mujoco", "dmlab", "lichess"]
    if "mujoco" in kinds:
        render_mujoco()
    if "dmlab" in kinds:
        render_dmlab()
    if "lichess" in kinds:
        render_lichess()
    print(f"\nWrote screenshots to: {PLOTS}", flush=True)


if __name__ == "__main__":
    main()
