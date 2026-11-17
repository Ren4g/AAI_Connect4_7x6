from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .agents import Agent
from .engine import ConnectFour, GameResult, Player
from .state import GameState

StateFactory = Callable[[], GameState]


@dataclass(frozen=True)
class GameMetrics:
    result: GameResult
    moves: int
    elapsed_seconds: float
    average_move_seconds: float
    move_times: tuple[float, ...] = ()
    nodes: int = 0
    simulations: int = 0
    first_agent: Optional[str] = None
    second_agent: Optional[str] = None
    budget_violations: int = 0


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total, with total positive")
    if confidence != 0.95:
        raise ValueError("Only 95% Wilson intervals are currently supported")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def play_game(
    first: Agent,
    second: Agent,
    first_agent: Optional[str] = None,
    second_agent: Optional[str] = None,
    time_budget_seconds: Optional[float] = None,
    engine_factory: StateFactory = ConnectFour,
) -> GameMetrics:
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")
    state = engine_factory()
    agents = {Player.ONE: first, Player.TWO: second}
    started = time.perf_counter()
    move_times: list[float] = []
    nodes = 0
    simulations = 0
    budget_violations = 0
    moves = 0
    while not state.is_terminal():
        agent = agents[state.current_player]
        move_started = time.perf_counter()
        action = agent.choose_action(state)
        move_elapsed = time.perf_counter() - move_started
        move_times.append(move_elapsed)
        if time_budget_seconds is not None and move_elapsed > time_budget_seconds:
            budget_violations += 1
        nodes += int(getattr(agent, "nodes", 0))
        simulations += int(getattr(agent, "simulations_run", 0))
        state.step(action)
        moves += 1
    elapsed = time.perf_counter() - started
    return GameMetrics(
        state.result(),
        moves,
        elapsed,
        elapsed / moves,
        tuple(move_times),
        nodes,
        simulations,
        first_agent,
        second_agent,
        budget_violations,
    )


def round_robin(
    agents: dict[str, Agent],
    games_per_pair: int = 100,
    time_budget_seconds: Optional[float] = None,
    engine_factory: StateFactory = ConnectFour,
) -> dict[tuple[str, str], list[GameMetrics]]:
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
                    results[key].append(
                        play_game(
                            agents[second_name],
                            agents[first_name],
                            second_name,
                            first_name,
                            time_budget_seconds,
                            engine_factory,
                        )
                    )
                else:
                    results[key].append(
                        play_game(
                            agents[first_name],
                            agents[second_name],
                            first_name,
                            second_name,
                            time_budget_seconds,
                            engine_factory,
                        )
                    )
    return results


def summarize_pair(
    first_name: str,
    second_name: str,
    games: list[GameMetrics],
) -> dict[str, object]:
    if not games:
        raise ValueError("games must not be empty")
    wins = draws = losses = 0
    move_times: list[float] = []
    nodes = simulations = violations = 0
    for game in games:
        if game.result is GameResult.DRAW:
            draws += 1
        elif game.first_agent == first_name:
            if game.result is GameResult.PLAYER_ONE:
                wins += 1
            else:
                losses += 1
        else:
            if game.result is GameResult.PLAYER_TWO:
                wins += 1
            else:
                losses += 1
        move_times.extend(game.move_times)
        nodes += game.nodes
        simulations += game.simulations
        violations += game.budget_violations
    total = len(games)
    score = (wins + 0.5 * draws) / total
    return {
        "first_agent": first_name,
        "second_agent": second_name,
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "win_rate_wilson_95": wilson_interval(wins, total),
        "average_move_seconds": sum(move_times) / len(move_times) if move_times else 0.0,
        "p95_move_seconds": percentile(move_times, 95),
        "nodes": nodes,
        "simulations": simulations,
        "budget_violations": violations,
    }


def summarize_results(
    results: dict[tuple[str, str], list[GameMetrics]],
) -> list[dict[str, object]]:
    return [
        summarize_pair(first_name, second_name, games)
        for (first_name, second_name), games in results.items()
    ]


def save_results(
    results: dict[tuple[str, str], list[GameMetrics]],
    path: str | Path,
) -> None:
    report = summarize_results(results)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

