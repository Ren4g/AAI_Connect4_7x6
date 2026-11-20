from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from connect4.evaluation_data import sha256_file
from connect4.solver import PonsSolver


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local exact-solver cost on a stratified sample")
    parser.add_argument("--manifest", default="results/solver_eval_seed20261122.manifest.json")
    parser.add_argument("--solver-root", default="solver")
    parser.add_argument("--positions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="results/solver_probe_20261120_seed2701_50.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    counts = {"opening": args.positions // 3 + (args.positions % 3 > 0),
              "middle": args.positions // 3 + (args.positions % 3 > 1),
              "endgame": args.positions // 3}
    chosen = []
    for stage, count in counts.items():
        candidates = [row for row in manifest["positions"] if row["stage"] == stage]
        rng.shuffle(candidates)
        stage_seen = set()
        for row in candidates:
            key = row["state_sha256_mirror_canonical"]
            if key in stage_seen:
                continue
            stage_seen.add(key)
            chosen.append(row)
            if sum(item["stage"] == stage for item in chosen) == count:
                break
    solver_root = Path(args.solver_root)
    executable = solver_root / "connect4" / "c4solver.exe"
    measurements = []
    started = time.perf_counter()
    with PonsSolver(str(executable), timeout_seconds=args.timeout) as solver:
        for index, row in enumerate(chosen, 1):
            before = time.perf_counter()
            scores = solver.scores(row["moves"])
            elapsed = time.perf_counter() - before
            measurements.append({
                "stage": row["stage"], "source_game": row["source_game"],
                "moves": row["moves"], "elapsed_seconds": elapsed,
                "legal_score_count": sum(score != -100 for score in scores),
            })
            if index % 10 == 0:
                print(f"{index}/{len(chosen)} positions in {time.perf_counter()-started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    grouped = defaultdict(list)
    for item in measurements:
        grouped[item["stage"]].append(item["elapsed_seconds"])
    solver_commit = subprocess.run(
        ["git", "-C", str(solver_root / "connect4"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or None
    try:
        root_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        root_commit = None
    output = {
        "run_id": f"solver_probe_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}",
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "root_commit": root_commit,
        "solver_commit": solver_commit,
        "solver_root": str(solver_root),
        "solver_binary_sha256": sha256_file(executable),
        "frozen_positions_sha256": manifest["positions_sha256"],
        "seed": args.seed,
        "positions": len(measurements),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_position": elapsed / len(measurements),
        "mean_solver_call_seconds": sum(x["elapsed_seconds"] for x in measurements) / len(measurements),
        "p95_solver_call_seconds": percentile([x["elapsed_seconds"] for x in measurements], .95),
        "by_stage": {
            stage: {"positions": len(values),
                    "mean_seconds": sum(values) / len(values),
                    "p95_seconds": percentile(values, .95),
                    "max_seconds": max(values)}
            for stage, values in grouped.items()
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "measurements": measurements,
    }
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "run_id", "positions", "elapsed_seconds", "mean_seconds_per_position",
        "mean_solver_call_seconds", "p95_solver_call_seconds", "by_stage",
    )}, indent=2))
    print(f"Saved probe to {output_path}")


if __name__ == "__main__":
    main()
