"""Calibrate a policy/value checkpoint's value head against frozen Pons labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from connect4.encoding import encode_planes
from connect4.engine import ConnectFour
from connect4.evaluation_data import sha256_file
from connect4.model import ConnectFourCNN


def clustered_intervals(rows: list[dict], replicates: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_game"]].append(row)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    mse_estimates, accuracy_estimates = [], []
    for _ in range(replicates):
        selected = rng.integers(0, len(ids), size=len(ids))
        sample = [row for index in selected for row in groups[ids[index]]]
        mse_estimates.append(np.mean([(row["prediction"] - row["target"]) ** 2 for row in sample]))
        accuracy_estimates.append(np.mean([row["class_correct"] for row in sample]))
    return {
        "value_mse_95": [float(v) for v in np.quantile(mse_estimates, [0.025, 0.975])],
        "wdl_accuracy_95": [float(v) for v in np.quantile(accuracy_estimates, [0.025, 0.975])],
        "bootstrap_unit": "source_game",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Ingested results/colab/<run_id> directory")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--frozen-manifest", type=Path,
                        default=Path("results/solver_eval_seed20261122.manifest.json"))
    parser.add_argument("--solver-result", type=Path,
                        default=Path("results/solver_eval_final_20261120_seed20261125.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.batch_size < 1:
        parser.error("bootstrap and batch-size must be positive")

    run_manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete" or run_manifest.get("phase") != "cnn_supervised_solver_labels":
        raise ValueError("value-head calibration requires a completed ingested supervised CNN run")
    checkpoint = args.run / args.checkpoint
    expected = run_manifest["files_sha256"].get(checkpoint.name)
    if not expected or sha256_file(checkpoint) != expected:
        raise ValueError("checkpoint is not covered by the ingested manifest hashes")
    if args.output.exists():
        raise FileExistsError("evaluation outputs are immutable; choose a new path")
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    solver = json.loads(args.solver_result.read_text(encoding="utf-8"))
    positions_file = Path(frozen["positions_file"])
    if not positions_file.is_absolute():
        positions_file = args.frozen_manifest.parent.parent / positions_file
    if sha256_file(positions_file) != frozen["positions_sha256"]:
        raise ValueError("frozen positions file checksum mismatch")
    if (solver["frozen_positions_sha256"] != frozen["positions_sha256"]
            or len(solver["scores_by_position"]) != frozen["position_count"]):
        raise ValueError("solver labels do not match the frozen test manifest")
    if run_manifest["frozen_test_sha256"] != frozen["positions_sha256"]:
        raise ValueError("checkpoint was trained against a different frozen test set")

    import torch

    model = ConnectFourCNN().to(args.device)
    blob = torch.load(checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    positions = frozen["positions"]
    labels = solver["scores_by_position"]
    if len(positions) != len(labels):
        raise ValueError("test position order/length differs from solver labels")
    features, targets = [], []
    for row, target in zip(positions, labels):
        state = ConnectFour()
        for action in row["moves"]:
            state.step(int(action))
        features.append(encode_planes(state.board, int(state.current_player)))
        targets.append(target["theoretical_wdl"])
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(x), args.batch_size):
            batch = x[offset:offset + args.batch_size].to(args.device)
            _, values = model(batch)
            predictions.extend(values.detach().cpu().tolist())

    observations = []
    for pos, label, prediction in zip(positions, labels, predictions):
        target = int(label["theoretical_wdl"])
        pred_class = 1 if prediction > 1 / 3 else -1 if prediction < -1 / 3 else 0
        observations.append({
            "source_game": pos["source_game"], "stage": pos["stage"],
            "theoretical_wdl": target, "prediction": float(prediction),
            "target": target, "class_correct": int(pred_class == target),
        })

    groups = {"all": observations, **{
        stage: [row for row in observations if row["stage"] == stage]
        for stage in ("opening", "middle", "endgame")
    }}
    metrics = {}
    for name, rows in groups.items():
        y = np.asarray([row["target"] for row in rows], dtype=float)
        yhat = np.asarray([row["prediction"] for row in rows], dtype=float)
        classes = np.asarray([row["class_correct"] for row in rows], dtype=float)
        majority = int(np.bincount((y.astype(int) + 1), minlength=3).argmax()) - 1
        bins = np.linspace(-1.0, 1.0, 6)
        calibration = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected = (yhat >= lower) & (yhat < upper if upper < 1 else yhat <= upper)
            if selected.any():
                calibration.append({"prediction_mean": float(yhat[selected].mean()),
                                    "target_mean": float(y[selected].mean()),
                                    "positions": int(selected.sum())})
        metrics[name] = {
            "positions": len(rows), "value_mse": float(np.mean((yhat - y) ** 2)),
            "wdl_accuracy": float(classes.mean()),
            "majority_baseline_accuracy": float(np.mean(y == majority)),
            "majority_baseline_class": majority,
            "calibration_bins": calibration,
            "cluster_bootstrap_95": clustered_intervals(rows, args.bootstrap,
                                                         args.seed + len(name)),
        }
    result = {
        "run_id": f"value_head_{run_manifest['run_id']}_{args.seed}",
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "checkpoint_sha256": expected, "frozen_positions_sha256": frozen["positions_sha256"],
        "solver_result": str(args.solver_result), "bootstrap": args.bootstrap,
        "bootstrap_seed": args.seed, "class_thresholds": "value > 1/3 win; value < -1/3 loss; otherwise draw",
        "interpretation": "value head predicts side-to-move WDL target; PUCT uses it as a leaf estimate",
        "metrics": metrics, "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved value-head report to {args.output}")


if __name__ == "__main__":
    main()
