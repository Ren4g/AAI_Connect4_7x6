from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .agents import Agent
from .engine import COLS, ROWS, ConnectFour, GameResult, Player
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
    actions: tuple[int, ...] = ()
    move_players: tuple[int, ...] = ()
    nodes_by_player: tuple[int, int] = (0, 0)
    simulations_by_player: tuple[int, int] = (0, 0)
    budget_violations_by_player: tuple[int, int] = (0, 0)
    opening_actions: tuple[int, ...] = ()


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
    opening_actions: tuple[int, ...] = (),
) -> GameMetrics:
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")
    state = engine_factory()
    for action in opening_actions:
        state.step(action)
        if state.is_terminal():
            raise ValueError("Opening actions must not end the game")
    agents = {Player.ONE: first, Player.TWO: second}
    started = time.perf_counter()
    move_times: list[float] = []
    move_players: list[int] = []
    nodes = 0
    simulations = 0
    budget_violations = 0
    nodes_by_player = [0, 0]
    simulations_by_player = [0, 0]
    violations_by_player = [0, 0]
    moves = len(opening_actions)
    actions: list[int] = list(opening_actions)
    while not state.is_terminal():
        player = state.current_player
        agent = agents[player]
        move_started = time.perf_counter()
        action = agent.choose_action(state)
        move_elapsed = time.perf_counter() - move_started
        move_times.append(move_elapsed)
        move_players.append(int(player))
        if time_budget_seconds is not None and move_elapsed > time_budget_seconds:
            budget_violations += 1
            violations_by_player[int(player) - 1] += 1
        move_nodes = int(getattr(agent, "nodes", 0))
        move_simulations = int(getattr(agent, "simulations_run", 0))
        nodes += move_nodes
        simulations += move_simulations
        nodes_by_player[int(player) - 1] += move_nodes
        simulations_by_player[int(player) - 1] += move_simulations
        state.step(action)
        actions.append(action)
        moves += 1
    elapsed = time.perf_counter() - started
    return GameMetrics(
        state.result(),
        moves,
        elapsed,
        elapsed / len(move_times) if move_times else 0.0,
        tuple(move_times),
        nodes,
        simulations,
        first_agent,
        second_agent,
        budget_violations,
        tuple(actions),
        tuple(move_players),
        tuple(nodes_by_player),
        tuple(simulations_by_player),
        tuple(violations_by_player),
        tuple(opening_actions),
    )


def _make_openings(
    games: int,
    plies: int,
    seed: int,
    engine_factory: StateFactory,
    apply_fair_filter: bool = True,
    diagnostics: dict[str, int] | None = None,
) -> list[tuple[int, ...]]:
    if plies == 0:
        if diagnostics is not None:
            diagnostics.update({
                "attempts": games,
                "accepted": games,
                "incomplete": 0,
                "would_reject_immediate_win": 0,
                "would_reject_no_safe_action": 0,
                "duplicate_position": 0,
                "filtered_immediate_win": 0,
                "filtered_no_safe_action": 0,
            })
        return [()] * games
    rng = random.Random(seed)
    openings: list[tuple[int, ...]] = []
    seen_positions: set[tuple[tuple[int, ...], ...]] = set()
    attempts = 0
    max_attempts = max(1000, games * 100)
    counts = {
        "attempts": 0,
        "accepted": 0,
        "incomplete": 0,
        "would_reject_immediate_win": 0,
        "would_reject_no_safe_action": 0,
        "duplicate_position": 0,
        "filtered_immediate_win": 0,
        "filtered_no_safe_action": 0,
    }
    while len(openings) < games and attempts < max_attempts:
        attempts += 1
        state = engine_factory()
        actions: list[int] = []
        for _ in range(plies):
            legal = state.legal_actions()
            if not legal or state.is_terminal():
                break
            action = rng.choice(legal)
            actions.append(action)
            state.step(action)
        if len(actions) != plies or state.is_terminal():
            counts["incomplete"] += 1
            continue
        # Do not hand either search agent a win before it has made a decision,
        # and reject starts where every move concedes an immediate reply win.
        has_immediate_win = _has_immediate_win(state)
        has_safe_action = _has_safe_action(state)
        if has_immediate_win:
            counts["would_reject_immediate_win"] += 1
        elif not has_safe_action:
            counts["would_reject_no_safe_action"] += 1
        if apply_fair_filter and (has_immediate_win or not has_safe_action):
            if has_immediate_win:
                counts["filtered_immediate_win"] += 1
            else:
                counts["filtered_no_safe_action"] += 1
            continue
        position = tuple(tuple(row) for row in state.board)
        if position in seen_positions:
            counts["duplicate_position"] += 1
            continue
        seen_positions.add(position)
        openings.append(tuple(actions))
        counts["accepted"] += 1
    if len(openings) != games:
        raise ValueError(
            f"Could generate only {len(openings)} unique {plies}-ply openings for {games} games; "
            "increase opening_plies"
        )
    counts["attempts"] = attempts
    if diagnostics is not None:
        diagnostics.update(counts)
    return openings


def _has_immediate_win(state: GameState) -> bool:
    player = state.current_player
    for action in state.legal_actions():
        child = state.clone()
        child.step(action)
        if child.result() is GameResult(int(player)):
            return True
    return False


def _has_safe_action(state: GameState) -> bool:
    player = state.current_player
    for action in state.legal_actions():
        child = state.clone()
        child.step(action)
        if child.is_terminal():
            if child.result() is GameResult.DRAW:
                return True
            continue
        if not _has_immediate_win(child):
            return True
    return False


def round_robin(
    agents: dict[str, Agent],
    games_per_pair: int = 100,
    time_budget_seconds: Optional[float] = None,
    engine_factory: StateFactory = ConnectFour,
    seed: int = 42,
    opening_plies: int = 6,
    apply_fair_filter: bool = True,
    opening_diagnostics: dict[str, int] | None = None,
    paired_openings: bool = False,
) -> dict[tuple[str, str], list[GameMetrics]]:
    if games_per_pair < 1:
        raise ValueError("games_per_pair must be positive")
    if games_per_pair % 2:
        raise ValueError("games_per_pair must be even to balance seats")
    if not 0 <= opening_plies < ROWS * COLS:
        raise ValueError("opening_plies must be between 0 and 41")
    names = list(agents)
    opening_count = games_per_pair // 2 if paired_openings else games_per_pair
    openings = _make_openings(
        opening_count, opening_plies, seed, engine_factory,
        apply_fair_filter=apply_fair_filter, diagnostics=opening_diagnostics,
    )
    results: dict[tuple[str, str], list[GameMetrics]] = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            key = (first_name, second_name)
            results[key] = []
            for game in range(games_per_pair):
                opening_index = game // 2 if paired_openings else game
                if game % 2:
                    results[key].append(
                        play_game(
                            agents[second_name],
                            agents[first_name],
                            second_name,
                            first_name,
                            time_budget_seconds,
                            engine_factory,
                            openings[opening_index],
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
                            openings[opening_index],
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
    per_agent: dict[str, dict[str, object]] = {
        name: {"move_times": [], "nodes_total": 0, "simulations_total": 0,
              "budget_violations": 0, "moves": 0}
        for name in (first_name, second_name)
    }
    score_by_seat: dict[int, list[float]] = {1: [], 2: []}
    score_by_opening: dict[tuple[int, ...], list[float]] = defaultdict(list)
    budget_violations = 0
    opening_positions: set[tuple[int, ...]] = set()
    for game in games:
        game_first = game.first_agent or first_name
        game_second = game.second_agent or second_name
        if game.result is GameResult.DRAW:
            draws += 1
        elif game_first == first_name:
            if game.result is GameResult.PLAYER_ONE:
                wins += 1
            else:
                losses += 1
        else:
            if game.result is GameResult.PLAYER_TWO:
                wins += 1
            else:
                losses += 1
        if game.result is GameResult.DRAW:
            score = 0.5
        elif (game_first == first_name and game.result is GameResult.PLAYER_ONE) or (
            game_second == first_name and game.result is GameResult.PLAYER_TWO
        ):
            score = 1.0
        else:
            score = 0.0
        seat = 1 if game_first == first_name else 2
        score_by_seat[seat].append(score)
        opening_positions.add(game.opening_actions)
        score_by_opening[game.opening_actions].append(score)
        budget_violations += game.budget_violations
        for index, player_value in enumerate(game.move_players):
            agent_name = game_first if player_value == 1 else game_second
            if agent_name not in per_agent:
                continue
            metrics = per_agent[agent_name]
            metrics["move_times"].append(game.move_times[index])
            metrics["moves"] += 1
        for player_index, agent_name in enumerate((game_first, game_second)):
            if agent_name not in per_agent:
                continue
            metrics = per_agent[agent_name]
            metrics["nodes_total"] += game.nodes_by_player[player_index]
            metrics["simulations_total"] += game.simulations_by_player[player_index]
            metrics["budget_violations"] += game.budget_violations_by_player[player_index]
    total = len(games)
    score = (wins + 0.5 * draws) / total
    for metrics in per_agent.values():
        times = metrics.pop("move_times")
        count = metrics["moves"]
        metrics["average_move_seconds"] = sum(times) / len(times) if times else 0.0
        metrics["p95_move_seconds"] = percentile(times, 95)
        metrics["nodes_per_move"] = metrics["nodes_total"] / count if count else 0.0
        metrics["simulations_per_move"] = metrics["simulations_total"] / count if count else 0.0
    paired = len(games) > 1 and all(len(scores) == 2 for scores in score_by_opening.values())
    score_interval = (
        _bootstrap_clustered_score_interval(list(score_by_opening.values()))
        if paired else _bootstrap_score_interval(score_by_seat)
    )
    return {
        "first_agent": first_name,
        "second_agent": second_name,
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        # Draws are included in total and count as non-wins for this interval.
        "win_rate_wilson_95": wilson_interval(wins, total),
        "decisive_win_rate_wilson_95": (
            wilson_interval(wins, wins + losses) if wins + losses else None
        ),
        "score_bootstrap_95": score_interval,
        "score_bootstrap_unit": "paired opening" if paired else "seat-stratified game",
        "score_bootstrap_replicates": 10000,
        "score_bootstrap_seed": 20261119,
        "unique_openings": len(opening_positions),
        "per_agent": per_agent,
        "budget_violations": budget_violations,
    }


def _bootstrap_score_interval(
    score_by_seat: dict[int, list[float]],
    replicates: int = 10000,
    seed: int = 20261119,
) -> tuple[float, float]:
    rng = random.Random(seed)
    groups = [scores for scores in score_by_seat.values() if scores]
    total = sum(len(scores) for scores in groups)
    estimates = []
    for _ in range(replicates):
        resampled_total = sum(
            sum(rng.choices(scores, k=len(scores))) for scores in groups
        )
        estimates.append(resampled_total / total)
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]


def _bootstrap_clustered_score_interval(
    score_groups: list[list[float]],
    replicates: int = 10000,
    seed: int = 20261119,
) -> tuple[float, float]:
    """Bootstrap paired games as opening clusters, preserving their dependence."""
    if not score_groups or any(not group for group in score_groups):
        raise ValueError("score_groups must contain non-empty groups")
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(score_groups) for _ in range(len(score_groups))]
        values = [score for group in sampled for score in group]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]


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


def save_game_logs(
    results: dict[tuple[str, str], list[GameMetrics]],
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    """Write run metadata and one replayable JSONL record per game."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"record_type": "run", **(metadata or {})}, ensure_ascii=False) + "\n")
        for pair, games in results.items():
            for game_index, game in enumerate(games):
                file.write(json.dumps({
                    "record_type": "game",
                    "pair": list(pair),
                    "game_index": game_index,
                    "player_one_agent": game.first_agent,
                    "player_two_agent": game.second_agent,
                    "result": int(game.result),
                    "plies": game.moves,
                    "actions": list(game.actions),
                    "opening_actions": list(game.opening_actions),
                    "move_players": list(game.move_players),
                    "elapsed_seconds": game.elapsed_seconds,
                    "move_times_seconds": list(game.move_times),
                    "nodes_total": game.nodes,
                    "nodes_by_player": list(game.nodes_by_player),
                    "simulations_total": game.simulations,
                    "simulations_by_player": list(game.simulations_by_player),
                    "budget_violations": game.budget_violations,
                    "budget_violations_by_player": list(game.budget_violations_by_player),
                }, ensure_ascii=False) + "\n")

