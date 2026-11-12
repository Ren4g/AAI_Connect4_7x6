from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect4.agents import AlphaBetaAgent, MCTSAgent, RandomAgent
from connect4.experiments import play_game, percentile


def run_ablation(games: int = 10, depth: int = 4) -> list[dict[str, object]]:
    if games < 1:
        raise ValueError("games must be positive")
    configurations: list[tuple[str, object]] = [
        ("ab-pruning-ordering", AlphaBetaAgent(depth, pruning=True, move_ordering=True)),
        ("ab-no-pruning", AlphaBetaAgent(depth, pruning=False, move_ordering=True)),
        ("ab-no-move-ordering", AlphaBetaAgent(depth, pruning=True, move_ordering=False)),
    ]
    configurations.extend(
        (f"mcts-{simulations}", MCTSAgent(simulations=simulations, seed=42))
        for simulations in (50, 100, 500, 1000, 5000)
    )
    report: list[dict[str, object]] = []
    for name, agent in configurations:
        games_played = [
            play_game(agent, RandomAgent(seed=game), name, "Random")
            for game in range(games)
        ]
        wins = sum(game.result.value == 1 for game in games_played)
        move_times = [move_time for game in games_played for move_time in game.move_times]
        report.append(
            {
                "configuration": name,
                "games": games,
                "wins_as_first_player": wins,
                "win_rate": wins / games,
                "average_move_seconds": sum(move_times) / len(move_times),
                "p95_move_seconds": percentile(move_times, 95),
                "nodes": sum(game.nodes for game in games_played),
                "simulations": sum(game.simulations for game in games_played),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpha-Beta and MCTS ablations")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()
    report = run_ablation(args.games, args.depth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ablation report to {output}")


if __name__ == "__main__":
    main()
