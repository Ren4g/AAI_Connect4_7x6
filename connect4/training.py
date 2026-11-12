from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .agents import Agent, AlphaBetaAgent
from .engine import ConnectFour, GameResult


@dataclass(frozen=True)
class PositionRecord:
    board: tuple[tuple[int, ...], ...]
    player: int
    action: int
    value: float


def generate_games(games: int, agent: Agent | None = None, seed: int | None = None) -> list[PositionRecord]:
    if games < 1:
        raise ValueError("games must be positive")
    expert = agent or AlphaBetaAgent(depth=3)
    records: list[PositionRecord] = []
    for _ in range(games):
        state = ConnectFour()
        history: list[tuple[tuple[tuple[int, ...], ...], int, int, int]] = []
        while not state.is_terminal():
            player = int(state.current_player)
            action = expert.choose_action(state)
            history.append((tuple(tuple(row) for row in state.board), player, action))
            state.step(action)
        for board, player, action in history:
            result = state.result()
            value = 0.0 if result is GameResult.DRAW else (1.0 if int(result) == player else -1.0)
            records.append(PositionRecord(board, player, action, value))
    return records


def mirror(record: PositionRecord) -> PositionRecord:
    board = tuple(tuple(reversed(row)) for row in record.board)
    return PositionRecord(board, record.player, 6 - record.action, record.value)


def deduplicate(records: Iterable[PositionRecord]) -> list[PositionRecord]:
    unique: dict[tuple[tuple[tuple[int, ...], ...], int], PositionRecord] = {}
    for record in records:
        unique.setdefault((record.board, record.player), record)
    return list(unique.values())


def save_csv(records: Iterable[PositionRecord], path: str | Path) -> None:
    rows = list(records)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["board", "player", "action", "value"])
        for record in rows:
            board = "/".join("".join(map(str, row)) for row in record.board)
            writer.writerow([board, record.player, record.action, record.value])


def build_training_records(games: int, seed: int | None = None) -> list[PositionRecord]:
    records = generate_games(games, seed=seed)
    augmented = records + [mirror(record) for record in records]
    return deduplicate(augmented)
