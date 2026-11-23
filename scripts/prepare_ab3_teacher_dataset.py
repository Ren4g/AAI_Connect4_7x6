from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from connect4.agents import AlphaBetaAgent
from connect4.engine import ConnectFour
from connect4.evaluation_data import sha256_file
from connect4.train_cnn import verify_dataset


def label_worker(payload: tuple[int, list[dict], int]) -> tuple[int, list[int]]:
    start, rows, depth = payload
    agent = AlphaBetaAgent(depth=depth)
    actions = []
    for row in rows:
        state = ConnectFour()
        for action in row["moves"]:
            state.step(int(action))
        actions.append(int(agent.choose_action(state)))
    return start, actions


def prepare(source: Path, output: Path, workers: int, depth: int = 3) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_manifest, arrays, rows = verify_dataset(source)
    if depth != 3:
        raise ValueError("This registered ablation uses AlphaBeta depth 3")
    started = time.perf_counter()
    chunks = []
    for worker in range(workers):
        indices = list(range(worker, len(rows), workers))
        if indices:
            chunks.append((min(indices), [rows[index] for index in indices], depth))
    actions_by_index = np.full(len(rows), -1, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(label_worker, chunk) for chunk in chunks]
        for future in as_completed(futures):
            start, actions = future.result()
            for index, action in zip(range(start, len(rows), workers), actions):
                actions_by_index[index] = action
            print(f"AlphaBeta-3 policy labels completed: "
                  f"{np.count_nonzero(actions_by_index >= 0)}/{len(rows)}", flush=True)
    if np.any(actions_by_index < 0):
        raise RuntimeError("AlphaBeta label coverage is incomplete")

    policies = np.zeros_like(arrays["policy"], dtype=np.float32)
    for index, action in enumerate(actions_by_index):
        if not arrays["legal_mask"][index, action]:
            raise ValueError(f"Teacher chose an illegal action at position {index}")
        policies[index, action] = 1.0
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "train_validation.npz"
    saved = {name: arrays[name] for name in arrays.files}
    saved["policy"] = policies
    np.savez_compressed(data_path, **saved)
    hash_path = output / "hashes.json"
    frozen_path = output / "frozen_test_hashes.json"
    hash_path.write_bytes((source / "hashes.json").read_bytes())
    frozen_path.write_bytes((source / "frozen_test_hashes.json").read_bytes())
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or None
    record = {
        "schema_version": 1,
        "run_id": output.name,
        "phase": "ab3_policy_teacher_ablation",
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "source_code_commit": commit,
        "source_script_sha256": script_hash,
        "parent_dataset_id": source_manifest["run_id"],
        "parent_manifest_sha256": sha256_file(source / "manifest.json"),
        "frozen_test_sha256": source_manifest["frozen_test_sha256"],
        "frozen_state_overlap_count": 0,
        "configuration": {
            "teacher": "AlphaBetaAgent",
            "depth": depth,
            "workers": workers,
            "policy_targets": "one-hot teacher action",
            "value_targets": "unchanged Pons exact sign from parent dataset",
            "positions": len(rows),
            "train_validation_source_game_split": "unchanged from parent dataset",
        },
        "files_sha256": {
            "train_validation.npz": sha256_file(data_path),
            "hashes.json": sha256_file(hash_path),
            "frozen_test_hashes.json": sha256_file(frozen_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "cpu_count": os.cpu_count(),
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired AlphaBeta-3 policy-label ablation dataset")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(prepare(Path(args.source), Path(args.output), args.workers, args.depth), indent=2))


if __name__ == "__main__":
    main()
