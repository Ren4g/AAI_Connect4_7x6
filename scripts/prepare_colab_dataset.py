from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from connect4.colab_data import check_pack_disjoint, collect_positions, split_rows_by_game
from connect4.encoding import encode_planes
from connect4.engine import ConnectFour
from connect4.evaluation_data import sha256_file


def _label_batch(payload):
    worker_index, solver_binary, rows, temperature, timeout = payload
    from connect4.solver import PonsSolver
    with PonsSolver(str(solver_binary), timeout_seconds=timeout) as solver:
        scores = solver.scores_batch([row["moves"] for row in rows])
    return worker_index, [row["state_sha256"] for row in rows], scores


def git_commit(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("connect4/*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def label_rows(rows, solver_root: Path, temperature: float, workers: int,
               timeout: float):
    executable = (solver_root / "connect4" / "c4solver.exe").resolve()
    chunks = [rows[index::workers] for index in range(workers)]
    labels = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_label_batch,
                                   (index, str(executable), chunk, temperature, timeout))
                   for index, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            worker, hashes, score_rows = future.result()
            labels.update(zip(hashes, score_rows))
            print(f"solver labels complete: {len(labels)}/{len(rows)} rows; worker={worker}", flush=True)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create test-disjoint solver-labeled train/validation data")
    parser.add_argument("--logs", nargs="+", default=[
        "benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl",
        "benchmarks/round_robin/round_robin_filtered_paired_seed20261124.games.jsonl",
        "benchmarks/round_robin/round_robin_unfiltered_paired_seed20261124.games.jsonl",
        "benchmarks/head_to_head/benchmark_ab_mcts_standard_100_paired_seed20261126.games.jsonl",
        "benchmarks/head_to_head/benchmark_heuristic_ab6_seed20261130.games.jsonl",
    ])
    parser.add_argument("--frozen-manifest", default="results/solver_eval_seed20261122.manifest.json")
    parser.add_argument("--solver-root", default="solver")
    parser.add_argument("--positions-per-stage", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--smoke", action="store_true", help="Tiny local pipeline run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.smoke:
        args.positions_per_stage = 6
        args.workers = min(args.workers, 2)
    if args.positions_per_stage < 1 or args.workers < 1 or args.temperature <= 0:
        parser.error("positions, workers, and temperature must be positive")

    started = time.perf_counter()
    rows, selection = collect_positions(
        args.logs, args.frozen_manifest, args.positions_per_stage, args.seed
    )
    splits = split_rows_by_game(rows, args.validation_fraction, args.seed + 1)
    check_pack_disjoint(rows, args.frozen_manifest)
    labels = label_rows(rows, Path(args.solver_root), args.temperature, args.workers, args.timeout)

    from connect4.engine import COLS
    features, policies, values, legal_masks, optimal_masks = [], [], [], [], []
    source_games, stages, split_names = [], [], []
    for split_name, split_rows in splits.items():
        for row in split_rows:
            state = ConnectFour()
            for action in row["moves"]:
                state.step(action)
            score_vector = labels[row["state_sha256"]]
            legal = state.legal_actions()
            best = max(score_vector[action] for action in legal)
            scaled = [score_vector[action] / args.temperature for action in legal]
            maximum = max(scaled)
            weights = {action: math.exp(value - maximum) for action, value in zip(legal, scaled)}
            normalizer = sum(weights.values())
            policy = [weights.get(action, 0.0) / normalizer for action in range(COLS)]
            features.append(encode_planes(state.board, int(state.current_player)))
            policies.append(policy)
            values.append(float((best > 0) - (best < 0)))
            legal_masks.append([action in legal for action in range(COLS)])
            optimal_masks.append([action in legal and score_vector[action] == best
                                  for action in range(COLS)])
            source_games.append(row["source_game"])
            stages.append(row["stage"])
            split_names.append(split_name)

    output = Path(args.output) if args.output else Path("colab/data") / (
        f"solverlabels_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
        + ("_smoke" if args.smoke else "")
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset package {output}")
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "train_validation.npz"
    np.savez_compressed(
        data_path,
        features=np.asarray(features, dtype=np.float32),
        policy=np.asarray(policies, dtype=np.float32),
        value=np.asarray(values, dtype=np.float32),
        legal_mask=np.asarray(legal_masks, dtype=np.bool_),
        optimal_mask=np.asarray(optimal_masks, dtype=np.bool_),
        split=np.asarray(split_names), stage=np.asarray(stages),
    )
    hash_rows = []
    for split_name, split_rows in splits.items():
        for row in split_rows:
            hash_rows.append({
                "source_game": row["source_game"], "stage": row["stage"],
                "split": split_name, "moves": row["moves"],
                "state_sha256_mirror_canonical": row["state_sha256"],
                "history_sha256": row["history_sha256"],
            })
    hashes_path = output / "hashes.json"
    hashes_path.write_text(json.dumps({"records": hash_rows}, indent=2), encoding="utf-8")
    frozen_manifest = json.loads(Path(args.frozen_manifest).read_text(encoding="utf-8"))
    frozen_hashes_path = output / "frozen_test_hashes.json"
    frozen_hashes_path.write_text(json.dumps({
        "dataset_id": frozen_manifest["dataset_id"],
        "positions_sha256": frozen_manifest["positions_sha256"],
        "state_sha256_mirror_canonical": sorted({
            row["state_sha256_mirror_canonical"] for row in frozen_manifest["positions"]
        }),
        "source_games": sorted({row["source_game"] for row in frozen_manifest["positions"]}),
    }, indent=2), encoding="utf-8")
    solver_root = Path(args.solver_root)
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "phase": "solver_labelled_cnn_data",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "source_code_commit": git_commit(Path.cwd()),
        "source_code_sha256": source_hash(Path.cwd()),
        "solver_commit": git_commit(solver_root / "connect4"),
        "solver_binary_sha256": sha256_file(solver_root / "connect4" / "c4solver.exe"),
        "frozen_test_dataset_id": selection["frozen_dataset_id"],
        "frozen_test_sha256": selection["frozen_positions_sha256"],
        "frozen_state_overlap_count": 0,
        "blocked_frozen_source_games": True,
        "configuration": {
            "input_logs": selection["input_logs_sha256"],
            "positions_per_stage": args.positions_per_stage,
            "validation_fraction": args.validation_fraction,
            "policy_temperature": args.temperature,
            "solver_workers": args.workers,
            "smoke": args.smoke,
        },
        "selection": selection["selection"],
        "splits": {
            name: {"positions": len(group),
                   "source_games": len({row["source_game"] for row in group}),
                   "by_stage": {stage: sum(row["stage"] == stage for row in group)
                                for stage in ("opening", "middle", "endgame")}}
            for name, group in splits.items()
        },
        "files_sha256": {
            data_path.name: sha256_file(data_path), hashes_path.name: sha256_file(hashes_path),
            frozen_hashes_path.name: sha256_file(frozen_hashes_path),
        },
        "local_label_seconds": time.perf_counter() - started,
        "producer_environment": {"python": platform.python_version(),
                                "platform": platform.platform()},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": output.name, "splits": manifest["splits"],
                      "frozen_overlap_count": manifest["frozen_state_overlap_count"],
                      "local_label_seconds": manifest["local_label_seconds"]}, indent=2))
    print(f"Dataset package: {output}")


if __name__ == "__main__":
    main()
