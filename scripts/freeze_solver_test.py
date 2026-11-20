from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect4.evaluation_data import build_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore source-game IDs for a frozen solver test set")
    parser.add_argument("--positions", default="results/solver_eval_seed20261122.positions.json")
    parser.add_argument("--game-log", default="benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl")
    parser.add_argument("--per-stage", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", default="results/solver_eval_seed20261122.manifest.json")
    args = parser.parse_args()
    manifest = build_frozen_manifest(args.positions, args.game_log, args.per_stage, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "dataset_id", "positions_sha256", "position_count",
        "unique_mirror_canonical_states", "duplicate_canonical_entries",
    )}, indent=2))
    print(f"Saved frozen test manifest to {output}")


if __name__ == "__main__":
    main()
