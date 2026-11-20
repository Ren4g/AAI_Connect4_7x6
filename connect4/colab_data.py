"""Prepare solver-labeled train/validation data with a frozen test exclusion."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .engine import ConnectFour
from .evaluation_data import PHASE_PLY, canonical_state_key, sha256_file


def split_rows_by_game(rows: list[dict[str, Any]], validation_fraction: float,
                       seed: int) -> dict[str, list[dict[str, Any]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    game_ids = sorted({row["source_game"] for row in rows})
    if len(game_ids) < 2:
        raise ValueError("at least two source games are required for a split")
    random.Random(seed).shuffle(game_ids)
    validation_count = max(1, min(len(game_ids) - 1, round(len(game_ids) * validation_fraction)))
    validation_games = set(game_ids[:validation_count])
    splits = {
        "train": [row for row in rows if row["source_game"] not in validation_games],
        "validation": [row for row in rows if row["source_game"] in validation_games],
    }
    if {row["state_sha256"] for row in splits["train"]} & {
        row["state_sha256"] for row in splits["validation"]
    }:
        raise ValueError("position or mirror leakage across train/validation")
    if {row["source_game"] for row in splits["train"]} & {
        row["source_game"] for row in splits["validation"]
    }:
        raise ValueError("source-game leakage across train/validation")
    return splits


def collect_positions(log_paths: list[str | Path], frozen_manifest_path: str | Path,
                      positions_per_stage: int, seed: int) -> tuple[list[dict], dict]:
    frozen_path = Path(frozen_manifest_path)
    frozen_manifest = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not frozen_manifest.get("frozen"):
        raise ValueError("A frozen test manifest is required")
    frozen_keys = {row["state_sha256_mirror_canonical"]
                   for row in frozen_manifest["positions"]}
    blocked_source_ids = {row["source_game"] for row in frozen_manifest["positions"]}
    rng = random.Random(seed)
    candidates: dict[str, list[dict]] = {stage: [] for stage in PHASE_PLY}
    input_hashes = {}

    for log_path in log_paths:
        path = Path(log_path)
        log_hash = sha256_file(path)
        input_hashes[str(path)] = log_hash
        same_test_source = log_hash == frozen_manifest["source_log_sha256"]
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("record_type") != "game":
                    continue
                pair = record["pair"]
                base_id = f"{pair[0]}|{pair[1]}|{record['game_index']}"
                if same_test_source and base_id in blocked_source_ids:
                    continue
                actions = tuple(int(action) for action in record["actions"])
                source_game = f"{log_hash[:16]}|{base_id}"
                for stage, (low, high) in PHASE_PLY.items():
                    last = min(high, len(actions) - 1)
                    if last < low:
                        continue
                    plies = list(range(low, last + 1))
                    rng.shuffle(plies)
                    candidates[stage].append({
                        "source_game": source_game,
                        "moves": list(actions[:plies[0]]),
                        "stage": stage,
                    })

    selected: list[dict] = []
    globally_seen: set[str] = set()
    selection_meta = {}
    for stage, stage_candidates in candidates.items():
        rng.shuffle(stage_candidates)
        selected_games = set()
        accepted = 0
        frozen_rejected = duplicate_rejected = 0
        for row in stage_candidates:
            if row["source_game"] in selected_games:
                continue
            state_key = canonical_state_key(row["moves"])
            if state_key in frozen_keys:
                frozen_rejected += 1
                continue
            if state_key in globally_seen:
                duplicate_rejected += 1
                continue
            row["state_sha256"] = state_key
            row["history_sha256"] = hashlib.sha256(
                bytes(row["moves"])
            ).hexdigest()
            selected.append(row)
            globally_seen.add(state_key)
            selected_games.add(row["source_game"])
            accepted += 1
            if accepted >= positions_per_stage:
                break
        selection_meta[stage] = {
            "candidates": len(stage_candidates), "selected": accepted,
            "frozen_state_rejections": frozen_rejected,
            "mirror_duplicate_rejections": duplicate_rejected,
        }
        if accepted < positions_per_stage:
            raise ValueError(
                f"Only {accepted} clean {stage} positions available; requested {positions_per_stage}"
            )
    return selected, {
        "frozen_dataset_id": frozen_manifest["dataset_id"],
        "frozen_positions_sha256": frozen_manifest["positions_sha256"],
        "frozen_state_count": len(frozen_keys),
        "input_logs_sha256": input_hashes,
        "selection": selection_meta,
        "frozen_overlap_count": 0,
    }


def check_pack_disjoint(rows: list[dict], frozen_manifest_path: str | Path) -> None:
    frozen_manifest = json.loads(Path(frozen_manifest_path).read_text(encoding="utf-8"))
    frozen_keys = {row["state_sha256_mirror_canonical"]
                   for row in frozen_manifest["positions"]}
    overlap = frozen_keys & {row["state_sha256"] for row in rows}
    if overlap:
        raise ValueError(f"Dataset intersects frozen test in {len(overlap)} positions")
