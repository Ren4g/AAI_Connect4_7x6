from __future__ import annotations

import time
from dataclasses import dataclass

from .agents import Agent
from .engine import ConnectFour, GameResult, Player


@dataclass(frozen=True)
class GameMetrics:
    result: GameResult
    moves: int
    elapsed_seconds: float
    average_move_seconds: float


def play_game(first: Agent, second: Agent) -> GameMetrics:
    state = ConnectFour()
    agents = {Player.ONE: first, Player.TWO: second}
    started = time.perf_counter()
    moves = 0
    while not state.is_terminal():
        state.step(agents[state.current_player].choose_action(state))
        moves += 1
    elapsed = time.perf_counter() - started
    return GameMetrics(state.result(), moves, elapsed, elapsed / moves)


def round_robin(agents: dict[str, Agent], games_per_pair: int = 100) -> dict[tuple[str, str], list[GameMetrics]]:
    if games_per_pair < 1:
        raise ValueError("games_per_pair must be positive")
    names = list(agents)
    results: dict[tuple[str, str], list[GameMetrics]] = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            key = (first_name, second_name)
            results[key] = []
            for game in range(games_per_pair):
                if game % 2:
                    results[key].append(play_game(agents[second_name], agents[first_name]))
                else:
                    results[key].append(play_game(agents[first_name], agents[second_name]))
    return results

