"""Lichess top-player PGN loader.

Policies are three named top players (default: Carlsen, Nakamura,
Caruana). Each player's Lichess username is used as the filter; the
loader reads games from a local PGN file (downloaded separately
from https://lichess.org/@/<user>/all or an elite-database PGN) and
produces one "episode" per game.

Per step:
  state  -> fixed 783-d vector from `chess_board_to_vector` at the
            position *before* the move is made
  action -> integer index into the fixed UCI-move vocabulary we
            derive from the union of moves observed across all games
            (kept stable across runs via a sorted vocabulary)

The UCI vocab is built from the union of moves observed in the three
players' data, sorted alphabetically, and saved next to the npz cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import io
import json
import os

import numpy as np

from .base import EpisodeMeta, EpisodeStore


DEFAULT_CACHE_ROOT = Path(os.environ.get("INR_LICHESS_CACHE",
                                          Path.home() / ".cache/INR/lichess")).expanduser()

TOP_PLAYERS_DEFAULT: Tuple[str, ...] = ("DrNykterstein", "Hikaru", "FabianoCaruana")


def _iter_games(pgn_path: Path):
    import chess.pgn
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                return
            yield g


def _game_to_positions_and_moves(game, min_plies: int, max_plies: int,
                                   tracked_player: str | None = None
                                   ) -> Tuple[np.ndarray, List[str]]:
    """Return (states, moves) for one PGN game.

    If `tracked_player` is given, keep only the half-moves where it is that
    player's turn — i.e. the moves *that player actually chose*. The state
    is the board immediately before the chosen move. This makes the episode
    a clean record of one player's policy, not a mixture with their
    opponents'. `min_plies`/`max_plies` then count the player's own plies.
    """
    import chess
    from utils.featurizers import chess_board_to_vector

    target_color: int | None = None
    if tracked_player is not None:
        white = (game.headers.get("White") or "").lower()
        black = (game.headers.get("Black") or "").lower()
        tp = tracked_player.lower()
        if tp == white:
            target_color = chess.WHITE
        elif tp == black:
            target_color = chess.BLACK
        else:
            # tracked player wasn't in this game — drop
            return np.zeros((0, 0), dtype=np.float32), []

    board = game.board()
    states: List[np.ndarray] = []
    moves: List[str] = []
    for mv in game.mainline_moves():
        keep = (target_color is None) or (board.turn == target_color)
        if keep:
            states.append(chess_board_to_vector(board))
            moves.append(mv.uci())
            if len(moves) >= max_plies:
                board.push(mv)
                break
        board.push(mv)
    if len(moves) < min_plies:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(states, axis=0).astype(np.float32), moves


def _merge_games_into_episodes(
    all_states: Sequence[np.ndarray],
    all_actions: Sequence[np.ndarray],
    policy_ids: Sequence[int],
    games_per_episode: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    if games_per_episode <= 1:
        return list(all_states), list(all_actions), list(policy_ids), [1] * len(policy_ids)

    merged_states: List[np.ndarray] = []
    merged_actions: List[np.ndarray] = []
    merged_policy_ids: List[int] = []
    merged_game_counts: List[int] = []

    cursor = 0
    while cursor < len(policy_ids):
        pid = int(policy_ids[cursor])
        end = cursor
        while end < len(policy_ids) and int(policy_ids[end]) == pid:
            end += 1
        count = end - cursor
        usable = (count // games_per_episode) * games_per_episode
        for start in range(cursor, cursor + usable, games_per_episode):
            states_chunk = [np.asarray(all_states[i], dtype=np.float32) for i in range(start, start + games_per_episode)]
            actions_chunk = [np.asarray(all_actions[i]) for i in range(start, start + games_per_episode)]
            merged_states.append(np.concatenate(states_chunk, axis=0))
            merged_actions.append(np.concatenate(actions_chunk, axis=0))
            merged_policy_ids.append(pid)
            merged_game_counts.append(games_per_episode)
        cursor = end

    return merged_states, merged_actions, merged_policy_ids, merged_game_counts


def build_lichess_store(pgn_dir: Path | str,
                        players: Sequence[str] = TOP_PLAYERS_DEFAULT,
                        max_games_per_player: int = 200,
                        min_plies: int = 20,
                        max_plies: int = 120,
                        tracked_player_only: bool = True,
                        vocab_path: Path | None = None,
                        games_per_episode: int = 1) -> EpisodeStore:
    """Build an EpisodeStore for the named Lichess players.

    pgn_dir: directory containing `{player}.pgn` (case-sensitive usernames).
              If a player's file is missing, we raise a clear error.
    tracked_player_only: when True (default), drop opponents' moves and keep
                         only the half-moves played by the named player.
                         min_plies / max_plies then count player's own plies.
    """
    pgn_dir = Path(pgn_dir).expanduser()
    DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tp_tag = "tp" if tracked_player_only else "all"
    cache_tag = "_".join(players) + f"_G{max_games_per_player}_P{min_plies}-{max_plies}_{tp_tag}"
    cache_file = DEFAULT_CACHE_ROOT / f"{cache_tag}.npz"
    vocab_file = vocab_path or (DEFAULT_CACHE_ROOT / f"{cache_tag}.vocab.json")

    if cache_file.exists() and vocab_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        all_states = list(z["states"])
        all_actions = list(z["actions"])
        with vocab_file.open() as f:
            vocab: List[str] = json.load(f)
        policy_ids = list(z["policy_ids"].astype(np.int64))
    else:
        # first pass: collect games + move strings per player
        per_player: Dict[int, List[Tuple[np.ndarray, List[str]]]] = {}
        for pid, player in enumerate(players):
            pgn_file = pgn_dir / f"{player}.pgn"
            if not pgn_file.exists():
                raise FileNotFoundError(
                    f"Expected Lichess PGN at {pgn_file}. "
                    f"Download with: curl -L 'https://lichess.org/api/games/user/{player}"
                    f"?max={max_games_per_player * 4}&moves=true&clocks=false&evals=false'"
                    f" > {pgn_file}")
            games_taken = 0
            per_player[pid] = []
            for g in _iter_games(pgn_file):
                if games_taken >= max_games_per_player:
                    break
                states, moves = _game_to_positions_and_moves(
                    g, min_plies, max_plies,
                    tracked_player=player if tracked_player_only else None,
                )
                if len(moves) == 0:
                    continue
                per_player[pid].append((states, moves))
                games_taken += 1
            if not per_player[pid]:
                raise RuntimeError(f"No usable games for player {player} in {pgn_file}")
        # build vocab: sorted union of moves seen anywhere
        uci_set = set()
        for games in per_player.values():
            for _, moves in games:
                uci_set.update(moves)
        vocab = sorted(uci_set)
        move_to_idx = {m: i for i, m in enumerate(vocab)}
        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        policy_ids: List[int] = []
        for pid, games in per_player.items():
            for states, moves in games:
                acts = np.array([move_to_idx[m] for m in moves], dtype=np.int64).reshape(-1, 1)
                all_states.append(states)
                all_actions.append(acts)
                policy_ids.append(pid)
        np.savez_compressed(cache_file,
                            states=np.array(all_states, dtype=object),
                            actions=np.array(all_actions, dtype=object),
                            policy_ids=np.array(policy_ids, dtype=np.int64))
        with vocab_file.open("w") as f:
            json.dump(vocab, f)

    all_states, all_actions, policy_ids, game_counts = _merge_games_into_episodes(
        all_states,
        all_actions,
        policy_ids,
        games_per_episode=int(games_per_episode),
    )

    all_meta: List[EpisodeMeta] = []
    for ei, (pid, game_count) in enumerate(zip(policy_ids, game_counts)):
        all_meta.append(EpisodeMeta(
            episode_id=ei, policy_id=int(pid), is_ood=False,
            source=f"lichess/{players[int(pid)]}",
            extras={"player": players[int(pid)], "games_per_episode": int(game_count)},
        ))
    state_dim = int(all_states[0].shape[-1]) if all_states else 0
    return EpisodeStore(
        states=all_states, actions=all_actions, meta=all_meta,
        state_dim=state_dim, action_dim=1,
        source="lichess",
        action_kind="discrete", n_actions=len(vocab),
    )
