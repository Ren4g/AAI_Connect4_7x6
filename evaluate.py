from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from connect4.agents import (AlphaBetaAgent, HeuristicAgent, MCTSAgent, PolicyAgent,
                             PUCTAgent, RandomAgent)
from connect4.experiments import round_robin, save_game_logs, save_results
from connect4.engine import ConnectFour
from connect4.engine_bit import ConnectFour as BitboardConnectFour


def make_agents(
    checkpoint: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    time_budget_seconds: float | None = None,
    alpha_beta_depth: int | None = None,
    puct_checkpoint: str | None = None,
    puct_simulations: int = 200,
) -> dict[str, object]:
    if alpha_beta_depth is None:
        if time_budget_seconds is None or time_budget_seconds <= 0.1:
            depth = 4
        elif time_budget_seconds < 0.4:
            depth = 6
        else:
            depth = 8
    else:
        depth = alpha_beta_depth
    simulations = 1_000_000_000 if time_budget_seconds is not None else 500
    ab_name = (
        f"AlphaBeta-time{time_budget_seconds:g}s-depth{depth}"
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
    if puct_checkpoint:
        agents["PUCT"] = PUCTAgent.from_checkpoint(
            puct_checkpoint, simulations=puct_simulations, device=device
        )
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Connect Four round-robin evaluation")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--preset", choices=("fast", "standard"), default=None,
                        help="Fast=100ms/depth4; standard=500ms/depth8")
    parser.add_argument("--ab-depth", type=int, default=None,
                        help="Override the AlphaBeta iterative-deepening ceiling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-plies", type=int, default=6,
                        help="Seeded random opening length; set 0 to disable")
    parser.add_argument("--opening-filter", choices=("fair", "none"), default="fair",
                        help="Fair filter removes starts with an immediate win or no safe reply")
    parser.add_argument("--paired-openings", action="store_true",
                        help="Replay each opening twice, once from each seat")
    parser.add_argument("--engine", choices=("grid", "bitboard"), default="grid")
    parser.add_argument("--checkpoint", default=None, help="Optional trained policy/value checkpoint")
    parser.add_argument("--puct-checkpoint", default=None,
                        help="Optional policy/value checkpoint searched with PUCT")
    parser.add_argument("--puct-simulations", type=int, default=200)
    parser.add_argument("--device", default="cpu", help="PyTorch device used for checkpoint inference")
    parser.add_argument(
        "--include", nargs="*", choices=("Random", "Heuristic", "AlphaBeta", "MCTS", "CNN-Policy", "PUCT"),
        default=None, help="Optional agent subset; e.g. --include AlphaBeta MCTS",
    )
    parser.add_argument("--output", default="benchmarks/round_robin/round_robin.json")
    parser.add_argument("--game-log", default=None, help="JSONL replay log; defaults beside --output")
    args = parser.parse_args()
    if args.preset is not None:
        if args.budget_ms is not None or args.ab_depth is not None:
            parser.error("--preset cannot be combined with --budget-ms or --ab-depth")
        args.budget_ms = 100 if args.preset == "fast" else 500
        args.ab_depth = 4 if args.preset == "fast" else 8
    budget_seconds = None if args.budget_ms is None else args.budget_ms / 1000
    agents = make_agents(args.checkpoint, args.device, args.seed, budget_seconds, args.ab_depth,
                         args.puct_checkpoint, args.puct_simulations)
    if args.include is not None:
        agents = {
            name: agent for name, agent in agents.items()
            if any(name == prefix or name.startswith(prefix + "-") for prefix in args.include)
        }
        if len(agents) < 2:
            parser.error("--include must select at least two agents; CNN-Policy requires --checkpoint")
    output_path = Path(args.output)
    game_log_path = Path(args.game_log) if args.game_log else output_path.with_suffix(".games.jsonl")
    metadata_path = output_path.with_suffix(".meta.json")
    if any(path.exists() for path in (output_path, game_log_path, metadata_path)):
        raise FileExistsError("round-robin outputs are immutable; choose new file paths")
    opening_diagnostics: dict[str, int] = {}
    results = round_robin(
        agents,
        games_per_pair=args.games,
        time_budget_seconds=budget_seconds,
        engine_factory=BitboardConnectFour if args.engine == "bitboard" else ConnectFour,
        seed=args.seed,
        opening_plies=args.opening_plies,
        apply_fair_filter=args.opening_filter == "fair",
        opening_diagnostics=opening_diagnostics,
        paired_openings=args.paired_openings,
    )
    save_results(results, args.output)
    game_log = args.game_log or str(Path(args.output).with_suffix(".games.jsonl"))
    run_metadata = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "games_per_pair": args.games,
        "time_budget_seconds": budget_seconds,
        "preset": args.preset,
        "opening_plies": args.opening_plies,
        "paired_openings": args.paired_openings,
        "opening_filter": args.opening_filter,
        "opening_filter_definition": (
            "reject starts with an immediate win for side to move or no move that avoids an immediate losing reply"
        ),
        "opening_generation": opening_diagnostics,
        "engine": args.engine,
        "agents": list(agents),
        "cnn_warmup_runs": getattr(agents.get("CNN-Policy"), "warmup_runs", 0),
        "timed_search_agents": [
            name for name in agents if name.startswith("AlphaBeta-") or name.startswith("MCTS-time")
        ] if budget_seconds is not None else [],
        "time_budget_semantics": (
            "maximum per decision; AlphaBeta and MCTS use timed search, other agents run normally"
            if budget_seconds is not None else "fixed depth/simulation settings"
        ),
        "checkpoint": args.checkpoint,
        "puct_checkpoint": args.puct_checkpoint,
        "puct_simulations": args.puct_simulations if args.puct_checkpoint else None,
    }
    save_game_logs(results, game_log, run_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {args.output}")
    print(f"Saved replayable game log to {game_log}")
    print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
