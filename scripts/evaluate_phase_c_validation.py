"""Evaluate all registered local Phase C snapshots on solver-labelled validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.agents import CENTER_ORDER, HeuristicAgent
from connect4.engine import ConnectFour
from connect4.evaluation_data import canonical_state_key, sha256_file
from connect4.model import ConnectFourCNN
from connect4.selfplay import BatchedPUCT

RUNS = {
    "warm": ("az_v2_warm_seed20261121", 20261121),
    "scratch": ("az_v2_scratch_seed20261122", 20261122),
}


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def interval(rows: list[dict], metric: str, replicates: int, seed: int) -> list[float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["source_game"]].append(int(row[metric]))
    ids = sorted(groups)
    values = np.asarray([sum(groups[key]) for key in ids], dtype=np.float64)
    sizes = np.asarray([len(groups[key]) for key in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ids), size=(replicates, len(ids)))
    rates = values[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return [float(v) for v in np.quantile(rates, [0.025, 0.975])]


def paired_delta(left: list[dict], right: list[dict], metric: str,
                 replicates: int, seed: int) -> list[float]:
    left_groups: dict[str, list[int]] = defaultdict(list)
    right_groups: dict[str, list[int]] = defaultdict(list)
    for row in left:
        left_groups[row["source_game"]].append(int(row[metric]))
    for row in right:
        right_groups[row["source_game"]].append(int(row[metric]))
    ids = sorted(set(left_groups) & set(right_groups))
    if any(len(left_groups[key]) != len(right_groups[key]) for key in ids):
        raise ValueError("paired validation rows do not align within source game")
    values = np.asarray([sum(b - a for a, b in zip(left_groups[k], right_groups[k]))
                         for k in ids], dtype=np.float64)
    sizes = np.asarray([len(left_groups[k]) for k in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ids), size=(replicates, len(ids)))
    rates = values[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return [float(v) for v in np.quantile(rates, [0.025, 0.975])]


def load_model(torch, checkpoint: Path, device: str):
    model = ConnectFourCNN().to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model


def safe_actions(state: ConnectFour) -> list[int]:
    legal = state.legal_actions()
    winning = HeuristicAgent._winning_actions(state, state.current_player)
    if winning:
        return winning
    safe = []
    for action in legal:
        child = state.clone()
        child.step(action)
        if not child.is_terminal() and not HeuristicAgent._winning_actions(
                child, state.current_player.opponent):
            safe.append(action)
    return safe or legal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("colab/data/solverlabels_20261121_seed20261121_v3_30k"))
    parser.add_argument("--runs-root", type=Path, default=Path("results/phase_c_local"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase_c_local/validation_curve_20261123.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="independent PUCT roots evaluated in one batched search")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error("batch-size and bootstrap must be positive")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(12)
    dataset_manifest_path = args.dataset / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    for relative, expected in dataset_manifest["files_sha256"].items():
        if sha256_file(args.dataset / relative) != expected:
            raise ValueError(f"dataset checksum mismatch: {relative}")
    frozen_hash = dataset_manifest["frozen_test_sha256"]
    arrays = np.load(args.dataset / "train_validation.npz", allow_pickle=False)
    records = json.loads((args.dataset / "hashes.json").read_text(encoding="utf-8"))["records"]
    if len(records) != len(arrays["features"]):
        raise ValueError("hash record and feature rows differ")

    validation_indices = [i for i, row in enumerate(records) if row["split"] == "validation"]
    validation_rows = [records[i] for i in validation_indices]
    if len(validation_rows) != 4530:
        raise ValueError(f"expected 4530 source-game validation rows, got {len(validation_rows)}")
    frozen_hashes_path = args.dataset / "frozen_test_hashes.json"
    frozen_hashes = set(json.loads(frozen_hashes_path.read_text(encoding="utf-8"))[
        "state_sha256_mirror_canonical"])
    if any(row["state_sha256_mirror_canonical"] in frozen_hashes for row in validation_rows):
        raise ValueError("validation contains a frozen-test state or mirror")

    states = []
    optimal_masks = np.asarray(arrays["optimal_mask"][validation_indices], dtype=bool)
    for row in validation_rows:
        state = ConnectFour()
        for action in row["moves"]:
            state.step(int(action))
        states.append(state)
    progress_path = args.output.with_suffix(".progress.json")
    progress = {"created_at": datetime.now().astimezone().isoformat(),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                "frozen_test_sha256": frozen_hash,
                "source_code_sha256": source_hash(ROOT), "completed": {},
                "validation_positions": len(validation_rows),
                "source_game_clusters": len({r["source_game"] for r in validation_rows}),
                "mirror_state_overlap_with_test": 0,
                "bootstrap": {"replicates": args.bootstrap, "seed": args.seed,
                              "cluster": "source_game"}}
    if progress_path.exists():
        old = json.loads(progress_path.read_text(encoding="utf-8"))
        if (old.get("dataset_manifest_sha256") != progress["dataset_manifest_sha256"]
                or old.get("source_code_sha256") != progress["source_code_sha256"]
                or old.get("frozen_test_sha256") != frozen_hash):
            raise ValueError("existing progress file belongs to a different dataset/source")
        progress = old

    def save_progress():
        progress["updated_at"] = datetime.now().astimezone().isoformat()
        tmp = progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        tmp.replace(progress_path)

    def score_rows(actions: list[int], selected_indices: list[int]) -> list[dict]:
        rows = []
        for idx, action in zip(selected_indices, actions):
            rows.append({"source_game": validation_rows[idx]["source_game"],
                         "state_key": validation_rows[idx]["state_sha256_mirror_canonical"],
                         "exact": int(optimal_masks[idx, action])})
        return rows

    configs = []
    for mode, (run_id, seed) in RUNS.items():
        run_dir = args.runs_root / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("iteration") != 40:
            raise ValueError(f"run is incomplete: {run_id}")
        if manifest.get("frozen_test_sha256") != frozen_hash:
            raise ValueError(f"frozen-test hash mismatch in {run_id}")
        progress.setdefault("training_source_hashes", {})[mode] = manifest.get("source_code_sha256")
        replay_hashes = set()
        latest = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        replay_hashes.update(latest.get("training_state_hashes", []))
        replay_hashes.update(row.get("state_sha256_mirror_canonical")
                             for game in latest.get("active", []) for row in game.get("rows", []))
        overlap = [row["state_sha256_mirror_canonical"] in replay_hashes for row in validation_rows]
        eligible = [i for i, is_overlap in enumerate(overlap) if not is_overlap]
        if not eligible:
            raise ValueError(f"all validation rows overlap replay for {run_id}")
        snapshots = run_dir / "checkpoints"
        expected_iterations = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        for iteration in expected_iterations:
            checkpoint = snapshots / f"checkpoint_iteration_{iteration:04d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_sha = sha256_file(checkpoint)
            expected_checkpoint_sha = manifest.get("files_sha256", {}).get(
                checkpoint.relative_to(run_dir).as_posix())
            if expected_checkpoint_sha and checkpoint_sha != expected_checkpoint_sha:
                raise ValueError(f"snapshot checksum mismatch: {checkpoint}")
            key = f"{mode}_iter{iteration:02d}"
            if key not in progress["completed"]:
                configs.append((mode, run_id, seed, iteration, checkpoint, checkpoint_sha, eligible, overlap))

    for mode, run_id, seed, iteration, checkpoint, checkpoint_sha, eligible, overlap in configs:
        started = time.perf_counter()
        model = load_model(torch, checkpoint, args.device)
        features = np.asarray(arrays["features"][validation_indices], dtype=np.float32)
        logits_all = []
        values_all = []
        with torch.inference_mode():
            for offset in range(0, len(eligible), 512):
                ids = eligible[offset:offset + 512]
                x = torch.as_tensor(features[ids], device=args.device)
                logits, values = model(x)
                logits_all.extend(logits.detach().cpu().numpy())
                values_all.extend(values.detach().cpu().numpy())
        # Policy-only raw and policy with tactical immediate-win/block filter.
        policy_raw, policy_filtered = [], []
        for local_idx, global_idx in enumerate(eligible):
            state = states[global_idx]
            legal = state.legal_actions()
            scores = logits_all[local_idx]
            policy_raw.append(max(legal, key=lambda a: (float(scores[a]), -CENTER_ORDER.index(a))))
            winning = HeuristicAgent._winning_actions(state, state.current_player)
            candidates = winning if winning else safe_actions(state)
            policy_filtered.append(max(candidates,
                                       key=lambda a: (float(scores[a]), -CENTER_ORDER.index(a))))

        search = BatchedPUCT(model, args.device, simulations=400, c_puct=1.5,
                             dirichlet_alpha=0.3, dirichlet_epsilon=0.0)
        puct_raw, puct_filtered = [], []
        rng = np.random.default_rng(args.seed + seed + iteration)
        for offset in range(0, len(eligible), args.batch_size):
            ids = eligible[offset:offset + args.batch_size]
            policies, _ = search.search([states[i] for i in ids], 1.0, rng)
            for global_idx, visit_policy in zip(ids, policies):
                state = states[global_idx]
                legal = state.legal_actions()
                puct_raw.append(max(legal, key=lambda a: (visit_policy[a], -a)))
                winning = HeuristicAgent._winning_actions(state, state.current_player)
                candidates = winning if winning else safe_actions(state)
                puct_filtered.append(max(candidates,
                                         key=lambda a: (visit_policy[a], -a)))

        observations = {}
        for label, actions in (("policy_raw", policy_raw), ("policy_filtered", policy_filtered),
                               ("puct_raw", puct_raw), ("puct_filtered", puct_filtered)):
            observations[label] = score_rows(actions, eligible)
        summary = {label: {
            "positions": len(rows), "replay_state_overlap_excluded": len(overlap) - len(eligible),
            "exact_optimal_rate": float(np.mean([r["exact"] for r in rows])),
            "cluster_bootstrap_95": interval(rows, "exact", args.bootstrap,
                                               args.seed + seed + iteration + len(label)),
            "observations": rows,
        } for label, rows in observations.items()}
        progress["completed"][f"{mode}_iter{iteration:02d}"] = {
            "run_id": run_id, "seed": seed, "iteration": iteration,
            "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha,
            "replay_state_overlap_excluded": len(overlap) - len(eligible),
            "replay_training_state_count": len(latest.get("training_state_hashes", [])),
            "source_game_clusters": len({validation_rows[i]["source_game"] for i in eligible}),
            "metrics": {name: {k: v for k, v in metric.items() if k != "observations"}
                        for name, metric in summary.items()},
            "observations": observations,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(f"{mode} iteration {iteration}: n={len(eligible)}/{len(validation_rows)}, "
              f"PUCT raw={summary['puct_raw']['exact_optimal_rate']:.3f}, "
              f"filtered={summary['puct_filtered']['exact_optimal_rate']:.3f}, "
              f"elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
        save_progress()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Paired deltas versus each run's iteration-0 initialization, by source-game cluster.
    for mode, (_, seed) in RUNS.items():
        baseline = progress["completed"][f"{mode}_iter00"]["observations"]
        for iteration in (5, 10, 15, 20, 25, 30, 35, 40):
            item = progress["completed"][f"{mode}_iter{iteration:02d}"]
            item["paired_delta_vs_iteration_0"] = {}
            for label in ("policy_raw", "policy_filtered", "puct_raw", "puct_filtered"):
                left, right = baseline[label], item["observations"][label]
                # align only states eligible for both checkpoints
                left_map = {r["state_key"]: r for r in left}
                right_map = {r["state_key"]: r for r in right}
                common = sorted(set(left_map) & set(right_map))
                left_aligned = [left_map[key] for key in common]
                right_aligned = [right_map[key] for key in common]
                if not common:
                    continue
                item["paired_delta_vs_iteration_0"][label] = {
                    "paired_positions": len(common),
                    "delta": float(np.mean([b["exact"] - a["exact"]
                                             for a, b in zip(left_aligned, right_aligned)])),
                    "cluster_bootstrap_95": paired_delta(left_aligned, right_aligned, "exact",
                                                           args.bootstrap, args.seed + seed + iteration),
                }
        save_progress()
    progress["completed_at"] = datetime.now().astimezone().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(f"Saved validation curve: {args.output}", flush=True)


if __name__ == "__main__":
    main()
