from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from connect4.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, PolicyAgent, RandomAgent
from connect4.experiments import round_robin, save_game_logs, save_results
from connect4.engine import COLS, ROWS, ConnectFour
from connect4.engine_bit import ConnectFour as BitboardConnectFour


def make_agents(
    checkpoint: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    time_budget_seconds: float | None = None,
) -> dict[str, object]:
    depth = ROWS * COLS if time_budget_seconds is not None else 4
    simulations = 1_000_000_000 if time_budget_seconds is not None else 500
    ab_name = (
        f"AlphaBeta-time{time_budget_seconds:g}s"
        if time_budget_seconds is not None else f"AlphaBeta-depth{depth}"
    )
    agents: dict[str, object] = {
        "Random": RandomAgent(seed=seed),
        "Heuristic": HeuristicAgent(),
        ab_name: AlphaBetaAgent(
            depth=depth, time_budget_seconds=time_budget_seconds
        ),
        (f"MCTS-time{time_budget_seconds:g}s" if time_budget_seconds is not None else f"MCTS-{simulations}"):
            MCTSAgent(simulations=simulations, seed=seed + 3,
                      time_budget_seconds=time_budget_seconds),
    }
    if checkpoint:
        agents["CNN-Policy"] = PolicyAgent.from_checkpoint(checkpoint, device=device)
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Connect Four round-robin evaluation")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-plies", type=int, default=6,
                        help="Seeded random opening length; set 0 to disable")
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--checkpoint", default=None, help="Optional trained policy/value checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device used for checkpoint inference")
    parser.add_argument(
        "--include", nargs="*", choices=("Random", "Heuristic", "AlphaBeta", "MCTS", "CNN-Policy"),
        default=None, help="Optional agent subset; e.g. --include AlphaBeta MCTS",
    )
    parser.add_argument("--output", default="results/round_robin.json")
    parser.add_argument("--game-log", default=None, help="JSONL replay log; defaults beside --output")
    args = parser.parse_args()
    budget_seconds = None if args.budget_ms is None else args.budget_ms / 1000
    agents = make_agents(args.checkpoint, args.device, args.seed, budget_seconds)
    if args.include is not None:
        agents = {
            name: agent for name, agent in agents.items()
            if any(name == prefix or name.startswith(prefix + "-") for prefix in args.include)
        }
        if len(agents) < 2:
            parser.error("--include must select at least two agents; CNN-Policy requires --checkpoint")
    results = round_robin(
        agents,
        games_per_pair=args.games,
        time_budget_seconds=budget_seconds,
        engine_factory=BitboardConnectFour if args.engine == "bitboard" else ConnectFour,
        seed=args.seed,
        opening_plies=args.opening_plies,
    )
    save_results(results, args.output)
    game_log = args.game_log or str(Path(args.output).with_suffix(".games.jsonl"))
    save_game_logs(results, game_log, {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "games_per_pair": args.games,
        "time_budget_seconds": budget_seconds,
        "opening_plies": args.opening_plies,
        "opening_filter": "no immediate win for side to move; at least one move avoids an immediate losing reply",
        "engine": args.engine,
        "agents": list(agents),
        "timed_search_agents": [
            name for name in agents if name.startswith("AlphaBeta-") or name.startswith("MCTS-time")
        ] if budget_seconds is not None else [],
        "time_budget_semantics": (
            "maximum per decision; AlphaBeta and MCTS use timed search, other agents run normally"
            if budget_seconds is not None else "fixed depth/simulation settings"
        ),
        "checkpoint": args.checkpoint,
    })
    print(f"Saved evaluation report to {args.output}")
    print(f"Saved replayable game log to {game_log}")


if __name__ == "__main__":
    main()
