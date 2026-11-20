"""Utilities for freezing solver test positions and checking mirror leakage."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .engine import ConnectFour

PHASE_PLY = {"opening": (7, 10), "middle": (11, 24), "endgame": (25, 41)}


def canonical_state_key(moves: tuple[int, ...] | list[int]) -> str:
    state = ConnectFour()
    for action in moves:
        state.step(int(action))
    board = tuple(tuple(row) for row in state.board)
    mirror = tuple(tuple(reversed(row)) for row in board)
    payload = json.dumps([min(board, mirror), int(state.current_player)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_sample(
    log_path: str | Path, per_stage: int = 500, seed: int = 20261122
) -> dict[str, list[dict[str, Any]]]:
    """Reproduce the original frozen sampler and restore source-game identity."""
    rng = random.Random(seed)
    candidates: dict[str, list[tuple[str, tuple[int, ...]]]] = {
        name: [] for name in PHASE_PLY
    }
    with Path(log_path).open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_type") != "game":
                continue
            actions = tuple(int(a) for a in record["actions"])
            pair = record["pair"]
            source_game = f"{pair[0]}|{pair[1]}|{record['game_index']}"
            for phase, (first_ply, last_ply) in PHASE_PLY.items():
                available_last = min(last_ply, len(actions) - 1)
                for ply in range(first_ply, available_last + 1):
                    candidates[phase].append((source_game, actions[:ply]))

    result: dict[str, list[dict[str, Any]]] = {}
    for phase, phase_candidates in candidates.items():
        rng.shuffle(phase_candidates)
        seen_histories: set[tuple[int, ...]] = set()
        unique = []
        for source_game, moves in phase_candidates:
            mirror = tuple(6 - action for action in moves)
            canonical = min(moves, mirror)
            if canonical in seen_histories:
                continue
            seen_histories.add(canonical)
            unique.append((source_game, moves))
        if len(unique) < per_stage:
            raise ValueError(f"Only {len(unique)} unique {phase} positions; requested {per_stage}")
        chosen: list[tuple[str, tuple[int, ...]]] = []
        per_game: dict[str, int] = {}
        for source_game, moves in unique:
            if per_game.get(source_game, 0) >= 2:
                continue
            chosen.append((source_game, moves))
            per_game[source_game] = per_game.get(source_game, 0) + 1
            if len(chosen) == per_stage:
                break
        if len(chosen) != per_stage:
            raise ValueError(f"Only {len(chosen)} {phase} positions after source-game cap")
        result[phase] = [
            {"stage": phase, "source_game": source_game, "moves": list(moves),
             "state_sha256_mirror_canonical": canonical_state_key(moves)}
            for source_game, moves in chosen
        ]
    return result


def build_frozen_manifest(
    positions_path: str | Path,
    game_log_path: str | Path,
    per_stage: int = 500,
    seed: int = 20261122,
) -> dict[str, Any]:
    positions_path, game_log_path = Path(positions_path), Path(game_log_path)
    frozen = json.loads(positions_path.read_text(encoding="utf-8"))
    reconstructed = reconstruct_sample(game_log_path, per_stage, seed)
    for phase, entries in reconstructed.items():
        observed = [entry["moves"] for entry in entries]
        if observed != frozen.get(phase):
            raise ValueError(f"Reconstructed {phase} sample differs from frozen file")
    states = [entry for group in reconstructed.values() for entry in group]
    keys = [entry["state_sha256_mirror_canonical"] for entry in states]
    return {
        "dataset_id": "solver_eval_seed20261122_frozen",
        "frozen": True,
        "positions_file": str(positions_path),
        "positions_sha256": sha256_file(positions_path),
        "source_log": str(game_log_path),
        "source_log_sha256": sha256_file(game_log_path),
        "sampler": {"seed": seed, "positions_per_stage": per_stage,
                    "phase_ply": {name: list(span) for name, span in PHASE_PLY.items()},
                    "reconstruction_matches_frozen_order": True},
        "positions": states,
        "position_count": len(states),
        "unique_mirror_canonical_states": len(set(keys)),
        "duplicate_canonical_entries": len(keys) - len(set(keys)),
    }
