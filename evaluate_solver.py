from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from connect4.agents import (
    AlphaBetaAgent,
    HeuristicAgent,
    MCTSAgent,
    PolicyAgent,
    PUCTAgent,
    RandomAgent,
)
from connect4.engine import ConnectFour
from connect4.evaluation_data import sha256_file
from connect4.solver import PonsSolver


def clustered_rate(rows: list[dict], metric: str, seed: int, replicates: int):
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["source_game"]].append(int(row[metric]))
    if not grouped:
        return None
    ids = sorted(grouped)
    values = np.asarray([sum(grouped[key]) for key in ids], dtype=np.float64)
    sizes = np.asarray([len(grouped[key]) for key in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(ids), size=(replicates, len(ids)))
    estimates = values[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def paired_difference(left: list[dict], right: list[dict], metric: str,
                      seed: int, replicates: int):
    left_groups: dict[str, list[int]] = defaultdict(list)
    right_groups: dict[str, list[int]] = defaultdict(list)
    for row in left:
        left_groups[row["source_game"]].append(int(row[metric]))
    for row in right:
        right_groups[row["source_game"]].append(int(row[metric]))
    ids = sorted(set(left_groups) & set(right_groups))
    if not ids:
        return None
    if any(len(left_groups[key]) != len(right_groups[key]) for key in ids):
        raise ValueError("paired observations do not align within source-game clusters")
    diffs = np.asarray([
        sum(r - l for l, r in zip(left_groups[key], right_groups[key]))
        for key in ids
    ], dtype=np.float64)
    sizes = np.asarray([len(left_groups[key]) for key in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(ids), size=(replicates, len(ids)))
    estimates = diffs[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def source_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_agents(checkpoints: list[str], puct_checkpoints: list[str], device: str,
                seed: int, puct_simulations: int):
    agents = {
        "Random": RandomAgent(seed),
        "Heuristic": HeuristicAgent(),
        **{f"AlphaBeta-depth{depth}": AlphaBetaAgent(depth=depth)
           for depth in range(1, 7)},
        "MCTS-500": MCTSAgent(simulations=500, seed=seed + 3),
    }
    checkpoint_hashes = {}
    for specification in checkpoints:
        if "=" not in specification:
            raise ValueError("Checkpoint format is LABEL=PATH")
        label, filename = specification.split("=", 1)
        if not label or label in ("Random", "Heuristic"):
            raise ValueError(f"Invalid checkpoint label {label!r}")
        filename_path = Path(filename)
        policy_agent = PolicyAgent.from_checkpoint(filename_path, device=device)
        checkpoint_hashes[label] = {
            "path": str(filename_path), "sha256": sha256_file(filename_path),
        }

        class RawPolicy:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def choose_action(self, state):
                legal = state.legal_actions()
                scores = self.wrapped.policy(state)
                return max(legal, key=lambda action: (float(scores[action]), -action))

        agents[f"CNN-{label}-raw"] = RawPolicy(policy_agent)
        agents[f"CNN-{label}-policy"] = policy_agent
    for specification in puct_checkpoints:
        if "=" not in specification:
            raise ValueError("PUCT checkpoint format is LABEL=PATH")
        label, filename = specification.split("=", 1)
        if not label or label in agents:
            raise ValueError(f"Invalid or duplicate PUCT label {label!r}")
        filename_path = Path(filename)
        agents[f"PUCT-{label}-raw"] = PUCTAgent.from_checkpoint(
            filename_path, simulations=puct_simulations, device=device,
            safety_filter=False,
        )
        agents[f"PUCT-{label}-filtered"] = PUCTAgent.from_checkpoint(
            filename_path, simulations=puct_simulations, device=device,
            safety_filter=True,
        )
        checkpoint_hashes[f"PUCT-{label}"] = {
            "path": str(filename_path), "sha256": sha256_file(filename_path),
        }
    return agents, checkpoint_hashes


def _solve_batch(payload):
    worker_index, solver_exe, rows, timeout_seconds = payload
    from connect4.solver import PonsSolver
    started = time.perf_counter()
    with PonsSolver(str(solver_exe), timeout_seconds=timeout_seconds) as solver:
        scores = solver.scores_batch([row["moves"] for row in rows])
    return worker_index, [row["index"] for row in rows], scores, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agents on the immutable, solver-labeled Connect Four test set"
    )
    parser.add_argument("--manifest", default="results/solver_eval_seed20261122.manifest.json")
    parser.add_argument("--solver-root", default="solver")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Optional clean checkpoints as LABEL=PATH; emits raw and safety-filtered policies")
    parser.add_argument("--puct-checkpoints", nargs="*", default=[],
                        help="Optional policy/value checkpoints as LABEL=PATH")
    parser.add_argument("--puct-simulations", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--solver-workers", type=int, default=8,
                        help="Parallel Pons processes; use 1 to run sequentially")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate a prefix only (smoke/diagnostic; never the final report)")
    parser.add_argument("--output", default="results/solver_eval_final_20261120_seed20261125.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest["positions_sha256"] != sha256_file(manifest["positions_file"]):
        raise ValueError("Frozen test manifest is missing or its source file hash changed")
    frozen_rows = manifest["positions"]
    if len(frozen_rows) != manifest["position_count"]:
        raise ValueError("Frozen test manifest row count is inconsistent")
    rows = frozen_rows if args.limit is None else frozen_rows[:args.limit]
    if not rows:
        raise ValueError("No evaluation positions selected")
    if args.solver_workers < 1:
        parser.error("--solver-workers must be positive")
    if Path(args.output).exists():
        raise FileExistsError("solver evaluation outputs are immutable; choose a new --output path")

    agents, checkpoint_hashes = make_agents(args.checkpoints, args.puct_checkpoints,
                                            args.device, args.seed, args.puct_simulations)
    observations: dict[str, list[dict]] = {name: [] for name in agents}
    operation_times: dict[str, list[float]] = {name: [] for name in agents}
    nodes: dict[str, list[int]] = {name: [] for name in agents}
    simulations: dict[str, list[int]] = {name: [] for name in agents}
    labels: list[dict] = []
    raw_observations: dict[str, list[dict]] = {name: [] for name in agents}
    solver_root = Path(args.solver_root)
    solver_binary = (solver_root / "connect4" / "c4solver.exe").resolve()
    started = time.perf_counter()
    worker_count = min(args.solver_workers, len(rows))
    indexed_rows = [{**row, "index": index} for index, row in enumerate(rows)]
    chunks = [indexed_rows[index::worker_count] for index in range(worker_count)]
    scores_by_index: dict[int, tuple[int, ...]] = {}
    worker_times = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_solve_batch, (index, str(solver_binary), chunk, 120.0))
                   for index, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            worker_index, indexes, batch_scores, worker_seconds = future.result()
            worker_times[str(worker_index)] = worker_seconds
            scores_by_index.update(zip(indexes, batch_scores))
            print(f"solver workers complete={len(worker_times)}/{worker_count}; "
                  f"positions={len(scores_by_index)}/{len(rows)}", flush=True)
    solver_elapsed = time.perf_counter() - started
    for index, row in enumerate(rows):
        state = ConnectFour()
        for action in row["moves"]:
            state.step(action)
        if state.is_terminal():
            raise ValueError(f"Frozen test contains terminal history at row {index + 1}")
        scores = scores_by_index[index]
        legal = state.legal_actions()
        best_score = max(scores[action] for action in legal)
        best_wdl = (best_score > 0) - (best_score < 0)
        action_scores = {action: scores[action] for action in legal}
        decision = any(((score > 0) - (score < 0)) != best_wdl
                       for score in action_scores.values())
        labels.append({
            "stage": row["stage"], "source_game": row["source_game"],
            "theoretical_wdl": best_wdl, "decision_position": decision,
            "best_score": best_score, "legal_actions": legal,
            "scores_by_column": list(scores),
        })
        for name, agent in agents.items():
            before = time.perf_counter()
            action = int(agent.choose_action(state))
            elapsed = time.perf_counter() - before
            if action not in legal:
                raise ValueError(f"{name} returned illegal move {action} at row {index + 1}")
            optimal = int(scores[action] == best_score)
            preserving = int(((scores[action] > 0) - (scores[action] < 0)) == best_wdl)
            observations[name].append({
                "source_game": row["source_game"], "stage": row["stage"],
                "theoretical_wdl": best_wdl, "decision_position": int(decision),
                "optimal": optimal, "wdl_preserving": preserving,
            })
            operation_times[name].append(elapsed)
            nodes[name].append(int(getattr(agent, "nodes", 0)))
            simulations[name].append(int(getattr(agent, "simulations_run", 0)))
            raw_observations[name].append({
                **observations[name][-1], "action": action,
                "action_solver_score": scores[action], "best_solver_score": best_score,
                "seconds": elapsed, "nodes": nodes[name][-1],
                "simulations": simulations[name][-1],
            })
        if (index + 1) % 100 == 0:
            print(f"evaluated agents on {index+1}/{len(rows)} frozen positions", flush=True)

    metric_specs = {
        "all": lambda row: True,
        "decision_positions": lambda row: row["decision_position"] == 1,
        "theoretical_win": lambda row: row["theoretical_wdl"] == 1,
        "theoretical_draw": lambda row: row["theoretical_wdl"] == 0,
        "theoretical_loss": lambda row: row["theoretical_wdl"] == -1,
        **{f"stage_{stage}": lambda row, stage=stage: row["stage"] == stage
           for stage in ("opening", "middle", "endgame")},
    }
    metrics = {}
    for agent_name, agent_rows in observations.items():
        metrics[agent_name] = {
            group_name: {
                "positions": sum(1 for row in agent_rows if selector(row)),
                **{
                    metric: {
                        "successes": sum(row[metric] for row in agent_rows if selector(row)),
                        "rate": (sum(row[metric] for row in agent_rows if selector(row)) /
                                 max(1, sum(1 for row in agent_rows if selector(row)))),
                        "cluster_bootstrap_95": clustered_rate(
                            [row for row in agent_rows if selector(row)], metric,
                            args.seed + len(agent_name) + len(group_name) + len(metric), args.bootstrap,
                        ),
                    }
                    for metric in ("optimal", "wdl_preserving")
                },
            }
            for group_name, selector in metric_specs.items()
        }
        move_times = operation_times[agent_name]
        metrics[agent_name]["local_cost"] = {
            "moves": len(move_times), "mean_seconds_per_move": sum(move_times) / len(move_times),
            "p95_seconds_per_move": float(np.quantile(move_times, .95)),
            "mean_nodes_per_move": sum(nodes[agent_name]) / len(nodes[agent_name]),
            "mean_simulations_per_move": sum(simulations[agent_name]) / len(simulations[agent_name]),
            "measurement_machine_only": True,
        }

    paired = {}
    baseline = observations.get("AlphaBeta-depth3")
    if baseline is not None:
        for agent_name, agent_rows in observations.items():
            paired[agent_name] = {}
            for metric in ("optimal", "wdl_preserving"):
                paired[agent_name][metric] = paired_difference(
                    baseline, agent_rows, metric, args.seed + 101 + len(agent_name), args.bootstrap
                )
                for stage in ("opening", "middle", "endgame"):
                    left = [row for row in baseline if row["stage"] == stage]
                    right = [row for row in agent_rows if row["stage"] == stage]
                    paired[agent_name][f"{metric}_{stage}"] = paired_difference(
                        left, right, metric, args.seed + 211 + len(agent_name) + len(stage), args.bootstrap
                    )

    root_commit = source_revision(Path.cwd())
    solver_commit = source_revision(solver_root / "connect4")
    output = {
        "run_id": f"solver_eval_frozen_20261120_seed{args.seed}",
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "root_commit": root_commit, "solver_commit": solver_commit,
        "source_tree_sha256": source_tree_hash(Path.cwd()),
        "solver_root": str(solver_root), "solver_binary_sha256": sha256_file(solver_binary),
        "frozen_dataset_id": manifest["dataset_id"],
        "frozen_positions_sha256": manifest["positions_sha256"],
        "positions": len(rows), "requested_frozen_positions": len(frozen_rows),
        "partial_diagnostic_only": args.limit is not None,
        "unique_mirror_canonical_states": manifest["unique_mirror_canonical_states"],
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed,
                      "cluster": "source game; paired difference on identical positions"},
        "agents": list(agents), "checkpoints": checkpoint_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "device": args.device},
        "solver_wall_seconds": solver_elapsed, "solver_worker_seconds": worker_times,
        "solver_workers": worker_count, "scores_by_position": labels,
        "observations_by_agent": raw_observations,
        "metrics": metrics, "paired_difference_vs_ab3": paired,
    }
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved solver evaluation to {output_path}; solver wall={solver_elapsed:.1f}s")


if __name__ == "__main__":
    main()
