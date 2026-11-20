from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from connect4.agents import AlphaBetaAgent
from connect4.engine import ConnectFour
from connect4.solver import PonsSolver


PHASES = {"opening": (7, 10), "middle": (11, 24), "endgame": (25, 41)}


def sample(log_path: str, per_stage: int, seed: int):
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for game_id, line in enumerate(Path(log_path).open(encoding="utf-8")):
        record = json.loads(line)
        if record.get("record_type") != "game":
            continue
        actions = record["actions"]
        for phase, (low, high) in PHASES.items():
            high = min(high, len(actions) - 1)
            for ply in range(low, high + 1):
                moves = tuple(actions[:ply])
                mirror = tuple(6 - x for x in moves)
                candidates[phase].append((game_id, min(moves, mirror), moves))
    result = {}
    for phase, rows in candidates.items():
        rng.shuffle(rows)
        chosen, seen, games = [], set(), defaultdict(int)
        for game, canonical, moves in rows:
            if canonical in seen or games.get(game, 0) >= 2:
                continue
            seen.add(canonical)
            chosen.append((game, moves))
            games[game] += 1
            if len(chosen) == per_stage:
                break
        if len(chosen) < per_stage:
            raise ValueError(f"only {len(chosen)} {phase} positions available")
        result[phase] = chosen
    return result


def bootstrap_by_game(rows, field: str, seed: int = 0, rounds: int = 2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_game"]].append(row[field])
    groups = list(grouped.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(rounds):
        sample_groups = [rng.choice(groups) for _ in groups]
        values = [value for group in sample_groups for value in group]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return [estimates[int(0.025 * rounds)], estimates[int(0.975 * rounds)]]


def main():
    parser = argparse.ArgumentParser(description="AlphaBeta depth curve against exact solver")
    parser.add_argument("--game-log", default="benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl")
    parser.add_argument("--solver", default="solver/connect4/c4solver.exe")
    parser.add_argument("--per-stage", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", default="results/solver_depth_curve_seed20261125.json")
    args = parser.parse_args()
    samples = sample(args.game_log, args.per_stage, args.seed)
    report = {"seed": args.seed, "per_stage": args.per_stage, "results": {}}
    with PonsSolver(args.solver, timeout_seconds=300) as solver:
        for depth in range(1, args.max_depth + 1):
            by_stage = {}
            all_rows = []
            for phase, positions in samples.items():
                rows = []
                agent = AlphaBetaAgent(depth=depth)
                for source_game, moves in positions:
                    state = ConnectFour()
                    for move in moves:
                        state.step(move)
                    scores = solver.scores(moves)
                    legal = state.legal_actions()
                    best = max(scores[a] for a in legal)
                    action = agent.choose_action(state)
                    row = {
                        "source_game": source_game,
                        "optimal": int(scores[action] == best),
                        "value_preserving": int((scores[action] > 0) - (scores[action] < 0) == (best > 0) - (best < 0)),
                    }
                    rows.append(row)
                    all_rows.append(row)
                by_stage[phase] = summarize(rows, seed=args.seed + depth)
            report["results"][str(depth)] = {
                "overall": summarize(all_rows, seed=args.seed + depth),
                "by_stage": by_stage,
            }
            print(f"depth={depth}: {report['results'][str(depth)]['overall']}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def summarize(rows, seed):
    return {
        "positions": len(rows),
        "optimal_action_rate": sum(row["optimal"] for row in rows) / len(rows),
        "optimal_action_ci95_clustered": bootstrap_by_game(rows, "optimal", seed),
        "value_preserving_rate": sum(row["value_preserving"] for row in rows) / len(rows),
        "value_preserving_ci95_clustered": bootstrap_by_game(rows, "value_preserving", seed + 1),
    }


if __name__ == "__main__":
    main()
