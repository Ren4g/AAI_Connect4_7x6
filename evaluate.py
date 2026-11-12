from __future__ import annotations

import argparse

from connect4.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from connect4.experiments import round_robin, save_results


def make_agents() -> dict[str, object]:
    return {
        "Random": RandomAgent(seed=42),
        "Heuristic": HeuristicAgent(),
        "AlphaBeta-depth4": AlphaBetaAgent(depth=4),
        "MCTS-500": MCTSAgent(simulations=500, seed=42),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Connect Four round-robin evaluation")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--output", default="results/round_robin.json")
    args = parser.parse_args()
    agents = make_agents()
    results = round_robin(
        agents,
        games_per_pair=args.games,
        time_budget_seconds=None if args.budget_ms is None else args.budget_ms / 1000,
    )
    save_results(results, args.output)
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
