from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .agents import Agent, AlphaBetaAgent
from .engine import ConnectFour, GameResult
from .state import GameState

StateFactory = Callable[[], GameState]


@dataclass(frozen=True)
class PositionRecord:
    board: tuple[tuple[int, ...], ...]
    player: int
    action: int
    value: float
    game_id: int = 0


def generate_games(
    games: int,
    agent: Agent | None = None,
    seed: int | None = None,
    epsilon: float = 0.05,
    engine_factory: StateFactory = ConnectFour,
) -> list[PositionRecord]:
    """Generate expert-labelled games with optional seeded behavior exploration."""
    if games < 1:
        raise ValueError("games must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    expert = agent or AlphaBetaAgent(depth=3)
    rng = random.Random(seed)
    records: list[PositionRecord] = []

    for game_id in range(games):
        state = engine_factory()
        history: list[tuple[tuple[tuple[int, ...], ...], int, int]] = []
        while not state.is_terminal():
            actions = state.legal_actions()
            if not actions:
                raise RuntimeError("Non-terminal game state has no legal actions")
            player = int(state.current_player)
            expert_action = expert.choose_action(state)
            if expert_action not in actions:
                raise ValueError(f"Expert selected illegal action {expert_action}")
            history.append((tuple(tuple(row) for row in state.board), player, expert_action))
            behavior_action = rng.choice(actions) if rng.random() < epsilon else expert_action
            state.step(behavior_action)

        result = state.result()
        for board, player, action in history:
            value = 0.0 if result == GameResult.DRAW else (1.0 if int(result) == player else -1.0)
            records.append(PositionRecord(board, player, action, value, game_id))
    return records


def mirror(record: PositionRecord) -> PositionRecord:
    board = tuple(tuple(reversed(row)) for row in record.board)
    return PositionRecord(board, record.player, 6 - record.action, record.value, record.game_id)


def _position_key(record: PositionRecord) -> tuple[tuple[tuple[int, ...], ...], int]:
    mirrored = tuple(tuple(reversed(row)) for row in record.board)
    return min(record.board, mirrored), record.player


def deduplicate(records: Iterable[PositionRecord]) -> list[PositionRecord]:
    """Merge repeated positions without arbitrarily discarding value labels."""
    grouped: dict[tuple[tuple[tuple[int, ...], ...], int], list[PositionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.board, record.player)].append(record)

    unique: list[PositionRecord] = []
    for (board, player), duplicates in grouped.items():
        action_counts = Counter(record.action for record in duplicates)
        action = min(action_counts, key=lambda move: (-action_counts[move], move))
        value = sum(record.value for record in duplicates) / len(duplicates)
        unique.append(PositionRecord(board, player, action, value, duplicates[0].game_id))
    return unique


def split_training_records(
    games: int,
    seed: int = 42,
    epsilon: float = 0.05,
    agent: Agent | None = None,
    engine_factory: StateFactory = ConnectFour,
) -> dict[str, list[PositionRecord]]:
    """Split complete games first, then remove mirrored position leakage."""
    if games < 3:
        raise ValueError("At least 3 games are required for train/validation/test splits")
    records = generate_games(games, agent=agent, seed=seed, epsilon=epsilon,
                             engine_factory=engine_factory)
    by_game: dict[int, list[PositionRecord]] = defaultdict(list)
    for record in records:
        by_game[record.game_id].append(record)
    game_ids = list(by_game)
    random.Random(seed).shuffle(game_ids)

    train_count = max(1, round(games * 0.8))
    validation_count = max(1, round(games * 0.1))
    if train_count + validation_count >= games:
        train_count = games - 2
        validation_count = 1
    train_ids = game_ids[:train_count]
    validation_ids = game_ids[train_count:train_count + validation_count]
    test_ids = game_ids[train_count + validation_count:]

    train = [record for game_id in train_ids for record in by_game[game_id]]
    validation = [record for game_id in validation_ids for record in by_game[game_id]]
    test = [record for game_id in test_ids for record in by_game[game_id]]

    train_keys = {_position_key(record) for record in train}
    validation = [record for record in validation if _position_key(record) not in train_keys]
    validation_keys = {_position_key(record) for record in validation}
    test = [record for record in test
            if _position_key(record) not in train_keys | validation_keys]
    if not validation or not test:
        raise ValueError(
            "Game-level split has no independent validation or test positions; "
            "increase --games or --epsilon to generate more varied games"
        )

    train = deduplicate(train + [mirror(record) for record in train])
    validation = deduplicate(validation)
    test = deduplicate(test)
    return {"train": train, "validation": validation, "test": test}


def save_csv(records: Iterable[PositionRecord], path: str | Path) -> None:
    rows = list(records)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["board", "player", "action", "value", "game_id"])
        for record in rows:
            board = "/".join("".join(map(str, row)) for row in record.board)
            writer.writerow([board, record.player, record.action, record.value, record.game_id])


def build_training_records(
    games: int,
    seed: int | None = None,
    epsilon: float = 0.05,
    engine_factory: StateFactory = ConnectFour,
) -> list[PositionRecord]:
    """Build a single augmented dataset; use split_training_records for model evaluation."""
    records = generate_games(games, seed=seed, epsilon=epsilon, engine_factory=engine_factory)
    return deduplicate(records + [mirror(record) for record in records])
