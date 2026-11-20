from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect4.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from connect4.experiments import _make_openings, play_game, summarize_pair
from connect4.engine import ConnectFour
from connect4.engine_bit import ConnectFour as BitboardConnectFour


def run_ablation(
    games: int = 10,
    depth: int = 4,
    engine_factory=ConnectFour,
    seed: int = 42,
    engine_name: str = "grid",
    opponent_name: str = "Random",
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    if games < 1:
        raise ValueError("games must be positive")
    seeds = seeds or (seed,)
    configurations: list[tuple[str, object]] = [
        ("ab-pruning-ordering", lambda: AlphaBetaAgent(depth, pruning=True, move_ordering=True)),
        ("ab-no-pruning", lambda: AlphaBetaAgent(depth, pruning=False, move_ordering=True)),
        ("ab-no-move-ordering", lambda: AlphaBetaAgent(depth, pruning=True, move_ordering=False)),
    ]
    configurations.extend(
        (f"mcts-{simulations}", lambda simulations=simulations: MCTSAgent(simulations=simulations, seed=seed + 3))
        for simulations in (50, 100, 500, 1000, 5000)
    )
    report: list[dict[str, object]] = []
    if opponent_name not in {"Random", "Heuristic", "AlphaBeta", "MCTS"}:
        raise ValueError("unsupported opponent")
    for current_seed in seeds:
        openings = _make_openings(games // 2, 6, current_seed, engine_factory)
        for name, agent_factory in configurations:
            agent = (
                MCTSAgent(simulations=int(name.split("-")[1]), seed=current_seed + 3)
                if name.startswith("mcts-") else agent_factory()
            )
            if opponent_name == "Random":
                opponent = RandomAgent(seed=current_seed + 100)
            elif opponent_name == "Heuristic":
                opponent = HeuristicAgent()
            elif opponent_name == "AlphaBeta":
                opponent = AlphaBetaAgent(depth=4)
            else:
                opponent = MCTSAgent(simulations=500, seed=current_seed + 200)
            games_played = []
            for game_index in range(games):
                opening = openings[game_index // 2]
                if game_index % 2:
                    games_played.append(play_game(
                        opponent, agent, opponent_name, name,
                        engine_factory=engine_factory, opening_actions=opening,
                    ))
                else:
                    games_played.append(play_game(
                        agent, opponent, name, opponent_name,
                        engine_factory=engine_factory, opening_actions=opening,
                    ))
            summary = summarize_pair(name, opponent_name, games_played)
            report.append({
                "configuration": name,
                "opponent": opponent_name,
                "games": games,
                "seed": current_seed,
                "depth": depth,
                "engine": engine_name,
                "unique_openings": summary["unique_openings"],
                "wins": summary["wins"],
                "draws": summary["draws"],
                "losses": summary["losses"],
                "score": summary["score"],
                "score_bootstrap_95": summary["score_bootstrap_95"],
                "agent_metrics": summary["per_agent"][name],
            })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alpha-Beta and MCTS ablations")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--opponent", choices=("Random", "Heuristic", "AlphaBeta", "MCTS"), default="Random")
    parser.add_argument("--output", default="benchmarks/ablation/ablation.json")
    args = parser.parse_args()
    factory = BitboardConnectFour if args.engine == "bitboard" else ConnectFour
    report = run_ablation(
        args.games, args.depth, factory, args.seed, args.engine, args.opponent,
        tuple(args.seeds) if args.seeds else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ablation report to {output}")


if __name__ == "__main__":
    main()
