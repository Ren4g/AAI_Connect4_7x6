from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect4.agents import AlphaBetaAgent, MCTSAgent, RandomAgent
from connect4.experiments import play_game, summarize_pair
from connect4.engine import ConnectFour
from connect4.engine_bit import ConnectFour as BitboardConnectFour


def run_ablation(
    games: int = 10,
    depth: int = 4,
    engine_factory=ConnectFour,
    seed: int = 42,
    engine_name: str = "grid",
) -> list[dict[str, object]]:
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
        games_played = []
        for game_index in range(games):
            random_agent = RandomAgent(seed=seed + game_index)
            if game_index % 2:
                games_played.append(
                    play_game(random_agent, agent, "Random", name, engine_factory=engine_factory)
                )
            else:
                games_played.append(
                    play_game(agent, random_agent, name, "Random", engine_factory=engine_factory)
                )
        summary = summarize_pair(name, "Random", games_played)
        agent_metrics = summary["per_agent"][name]
        report.append(
            {
                "configuration": name,
                "games": games,
                "seed": seed,
                "depth": depth,
                "engine": engine_name,
                "wins": summary["wins"],
                "draws": summary["draws"],
                "losses": summary["losses"],
                "score": summary["score"],
                "agent_metrics": agent_metrics,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpha-Beta and MCTS ablations")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()
    factory = BitboardConnectFour if args.engine == "bitboard" else ConnectFour
    report = run_ablation(args.games, args.depth, factory, args.seed, args.engine)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ablation report to {output}")


if __name__ == "__main__":
    main()
