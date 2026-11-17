from __future__ import annotations

import argparse

from connect4.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, PolicyAgent, RandomAgent
from connect4.experiments import round_robin, save_results
from connect4.engine import ConnectFour
from connect4.engine_bit import ConnectFour as BitboardConnectFour


def make_agents(checkpoint: str | None = None, device: str = "cpu") -> dict[str, object]:
    agents: dict[str, object] = {
        "Random": RandomAgent(seed=42),
        "Heuristic": HeuristicAgent(),
        "AlphaBeta-depth4": AlphaBetaAgent(depth=4),
        "MCTS-500": MCTSAgent(simulations=500, seed=42),
    }
    if checkpoint:
        agents["CNN-Policy"] = PolicyAgent.from_checkpoint(checkpoint, device=device)
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Connect Four round-robin evaluation")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--checkpoint", default=None, help="Optional trained policy/value checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device used for checkpoint inference")
    parser.add_argument("--output", default="results/round_robin.json")
    args = parser.parse_args()
    agents = make_agents(args.checkpoint, args.device)
    results = round_robin(
        agents,
        games_per_pair=args.games,
        time_budget_seconds=None if args.budget_ms is None else args.budget_ms / 1000,
        engine_factory=BitboardConnectFour if args.engine == "bitboard" else ConnectFour,
    )
    save_results(results, args.output)
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
