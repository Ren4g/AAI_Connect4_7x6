from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from connect4.engine import ConnectFour
from connect4.solver import phase
from connect4.agents import PolicyAgent
from connect4.agents import AlphaBetaAgent
from train_solver import sample_log, split_train_val, canonical_state


def main():
    parser = argparse.ArgumentParser(description="Compare CNN checkpoints on the same solver-held-out positions")
    parser.add_argument("--train-log", default="benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl")
    parser.add_argument("--test-log", default="benchmarks/round_robin/round_robin_filtered_paired_seed20261124.games.jsonl")
    parser.add_argument("--labels", default="results/solver_labels_seed20261201.json")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ab-depths", nargs="*", type=int, default=[3])
    parser.add_argument("--per-stage", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/solver_checkpoint_comparison.json")
    args = parser.parse_args()

    train_sample, _ = sample_log(args.train_log, args.per_stage, args.seed)
    test_sample, test_meta = sample_log(args.test_log, args.per_stage, args.seed + 1)
    train_rows, validation_rows, train_keys, validation_keys, _ = split_train_val(
        train_sample, 0.15, args.seed
    )
    test_rows = [row for rows in test_sample.values() for row in rows
                 if canonical_state(row[1]) not in train_keys | validation_keys]
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["records"]
    label_map = {(row["source_game"], tuple(row["moves"])): row for row in labels}
    agents = {str(path): PolicyAgent.from_checkpoint(path, device=args.device)
              for path in args.checkpoints}
    agents.update({f"AlphaBeta-depth{depth}": AlphaBetaAgent(depth=depth)
                   for depth in args.ab_depths})
    tallies = {name: {"all": [0, 0], "by_stage": defaultdict(lambda: [0, 0])}
               for name in agents}
    observations = {name: [] for name in agents}
    for source_game, moves in test_rows:
        target = label_map[(source_game, moves)]
        state = ConnectFour()
        for action in moves:
            state.step(action)
        stage = phase(len(moves))
        for path, agent in agents.items():
            action = agent.choose_action(state)
            optimal = action in target["optimal_actions"]
            counts = tallies[path]
            counts["all"][0] += int(optimal)
            counts["all"][1] += 1
            counts["by_stage"][stage][0] += int(optimal)
            counts["by_stage"][stage][1] += 1
            observations[path].append({"source_game": source_game, "stage": stage,
                                       "optimal": int(optimal)})

    results = {}
    for path, counts in tallies.items():
        results[path] = {
            "overall": {"optimal": counts["all"][0], "positions": counts["all"][1],
                        "exact_optimal_rate": counts["all"][0] / counts["all"][1]},
            "by_stage": {
                key: {"optimal": value[0], "positions": value[1],
                      "exact_optimal_rate": value[0] / value[1],
                      "cluster_bootstrap_95": cluster_interval(
                          [row for row in observations[path] if row["stage"] == key],
                          seed=args.seed + len(key),
                      )}
                for key, value in counts["by_stage"].items()
            },
            "cluster_bootstrap_95": cluster_interval(observations[path], seed=args.seed),
        }
    baseline = str(args.checkpoints[0])
    paired_differences = {}
    references = [name for name in agents if name.startswith("AlphaBeta-depth")]
    for checkpoint in [str(path) for path in args.checkpoints[1:]]:
        paired_differences[checkpoint] = paired_difference_interval(
            observations[baseline], observations[checkpoint], seed=args.seed + 17
        )
    for reference in references:
        paired_differences[f"{baseline} - {reference}"] = paired_difference_interval(
            observations[reference], observations[baseline], seed=args.seed + 23
        )
        paired_differences[f"{baseline} - {reference} by stage"] = {
            stage: paired_difference_interval(
                [row for row in observations[reference] if row["stage"] == stage],
                [row for row in observations[baseline] if row["stage"] == stage],
                seed=args.seed + 31 + len(stage),
            )
            for stage in sorted({row["stage"] for row in observations[reference]})
        }
        for checkpoint_name in [str(path) for path in args.checkpoints]:
            paired_differences[f"{checkpoint_name} - {reference}"] = paired_difference_interval(
                observations[reference], observations[checkpoint_name], seed=args.seed + 41
            )
            paired_differences[f"{checkpoint_name} - {reference} by stage"] = {
                stage: paired_difference_interval(
                    [row for row in observations[reference] if row["stage"] == stage],
                    [row for row in observations[checkpoint_name] if row["stage"] == stage],
                    seed=args.seed + 47 + len(stage),
                )
                for stage in sorted({row["stage"] for row in observations[reference]})
            }
    output = {
        "train_log": args.train_log, "test_log": args.test_log, "labels": args.labels,
        "seed": args.seed, "per_stage_requested": args.per_stage,
        "test_sampling": test_meta, "test_positions_after_split_filter": len(test_rows),
        "checkpoints": results,
        "paired_difference_vs_first_checkpoint": paired_differences,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved comparison to {args.output}")


def cluster_interval(rows, seed: int, replicates: int = 3000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_game"]].append(row["optimal"])
    game_ids = list(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        values = [value for game in (rng.choice(game_ids) for _ in game_ids)
                  for value in grouped[game]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    return [estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]]


def paired_difference_interval(left_rows, right_rows, seed: int, replicates: int = 3000):
    left = defaultdict(list)
    right = defaultdict(list)
    for row in left_rows:
        left[row["source_game"]].append(row["optimal"])
    for row in right_rows:
        right[row["source_game"]].append(row["optimal"])
    shared = sorted(set(left) & set(right))
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(shared) for _ in shared]
        l_values = [value for game in sampled for value in left[game]]
        r_values = [value for game in sampled for value in right[game]]
        estimates.append(sum(r_values) / len(r_values) - sum(l_values) / len(l_values))
    estimates.sort()
    return [estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]]


if __name__ == "__main__":
    main()
