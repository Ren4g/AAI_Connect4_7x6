"""Resumable policy/value CNN training CLI for local or Colab execution."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _total_ram_bytes() -> int | None:
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def _atomic_torch_save(torch, value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _rng_state(torch) -> dict:
    return {
        "python": random.getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(torch, state: dict) -> None:
    random.setstate(state["python"])
    __import__("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def verify_dataset(dataset_dir: Path, frozen_sha256: str | None = None):
    import numpy as np

    manifest_path = dataset_dir / "manifest.json"
    data_path = dataset_dir / "train_validation.npz"
    hashes_path = dataset_dir / "hashes.json"
    frozen_path = dataset_dir / "frozen_test_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))["records"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_sha = frozen_sha256 or manifest["frozen_test_sha256"]
    if frozen["positions_sha256"] != expected_sha or manifest["frozen_test_sha256"] != expected_sha:
        raise ValueError("Frozen test checksum does not match dataset manifest")
    for name, expected in manifest["files_sha256"].items():
        file_path = dataset_dir / name
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise ValueError(f"Dataset input hash mismatch: {name}")
    frozen_keys = set(frozen["state_sha256_mirror_canonical"])
    if frozen_keys & {row["state_sha256_mirror_canonical"] for row in hashes}:
        raise ValueError("Train/validation data overlaps the frozen test states")
    if any(row["split"] not in {"train", "validation"} for row in hashes):
        raise ValueError("Dataset contains a test split; only train and validation are allowed")
    train_games = {row["source_game"] for row in hashes if row["split"] == "train"}
    val_games = {row["source_game"] for row in hashes if row["split"] == "validation"}
    if train_games & val_games:
        raise ValueError("Train/validation source-game leakage")
    arrays = np.load(data_path)
    split = arrays["split"].astype(str)
    if set(split) != {"train", "validation"}:
        raise ValueError("Data arrays must contain train and validation only")
    if len(split) != len(hashes):
        raise ValueError("Data array and hash manifest lengths differ")
    for name in ("features", "policy", "value", "optimal_mask", "stage"):
        if len(arrays[name]) != len(split):
            raise ValueError(f"Array length mismatch: {name}")
    return manifest, arrays, hashes


def train(args) -> dict:
    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Install PyTorch and NumPy to train the CNN") from error
    from .model import ConnectFourCNN
    from .resource_metrics import peak_vram_bytes, reset_peak_vram_stats

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.dataset)
    dataset_manifest, arrays, hash_rows = verify_dataset(dataset_dir, args.frozen_sha256)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_id = args.run_id or f"cnn_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
    output = Path(args.output_root) / run_id
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output}")
    elif output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "checkpoint_latest.pt"
    best_path = output / "checkpoint_best.pt"
    log_path = output / "train_log.jsonl"
    manifest_path = output / "manifest.json"
    previous_peak_vram = 0
    if args.resume and manifest_path.exists():
        previous_peak_vram = int(json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "peak_vram_bytes", 0
        ))

    config = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "patience": args.patience,
        "value_weight": args.value_weight, "mirror_augmentation": True,
        "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
        "smoke": args.smoke,
    }
    if args.smoke:
        config.update({"epochs": 2, "batch_size": min(args.batch_size, 8), "patience": 2})

    model = ConnectFourCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_idx = np.flatnonzero(arrays["split"].astype(str) == "train")
    val_idx = np.flatnonzero(arrays["split"].astype(str) == "validation")
    if args.smoke:
        train_idx = train_idx[:min(32, len(train_idx))]
        val_idx = val_idx[:min(16, len(val_idx))]
    x = torch.as_tensor(arrays["features"], dtype=torch.float32, device=device)
    p = torch.as_tensor(arrays["policy"], dtype=torch.float32, device=device)
    v = torch.as_tensor(arrays["value"], dtype=torch.float32, device=device)
    optimal = torch.as_tensor(arrays["optimal_mask"], dtype=torch.bool, device=device)
    # Gravity makes a legal column exactly one whose top cell is empty.
    legal = (torch.as_tensor(arrays["legal_mask"], dtype=torch.bool, device=device)
             if "legal_mask" in arrays.files else x[:, :, 0, :].sum(dim=1) == 0)

    start_epoch, cursor, order = 0, 0, None
    global_step, best_val, best_epoch, stale_epochs = 0, float("inf"), 0, 0
    best_state = None
    best_metrics = None
    epoch_loss_sum = 0.0
    epoch_samples = 0
    resume_count = 0
    started_at = datetime.now().astimezone().isoformat()
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["dataset_sha256"] != dataset_manifest["files_sha256"]["train_validation.npz"]:
            raise ValueError("Resume checkpoint belongs to a different dataset")
        if state["config"] != config:
            raise ValueError("Resume configuration differs from saved run")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, cursor = state["epoch"], state["cursor"]
        order = state["order"]
        global_step, best_val = state["global_step"], state["best_val"]
        best_epoch, stale_epochs = state["best_epoch"], state["stale_epochs"]
        best_state = state["best_state_dict"]
        best_metrics = state.get("best_metrics")
        epoch_loss_sum, epoch_samples = state["epoch_loss_sum"], state["epoch_samples"]
        resume_count = state["resume_count"] + 1
        started_at = state["started_at"]
        _restore_rng(torch, state["rng_state"])

    manifest = {
        "schema_version": 1, "run_id": run_id, "phase": "cnn_supervised_solver_labels",
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "frozen_test_sha256": dataset_manifest["frozen_test_sha256"],
        "test_overlap_count": 0, "source_code_commit": _git_commit(Path.cwd()),
        "source_code_sha256": _source_tree_sha256(Path.cwd()),
        "solver_commit": dataset_manifest.get("solver_commit"),
        "seed": args.seed, "configuration": {
            **config, "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
            "dataset_npz_sha256": dataset_manifest["files_sha256"]["train_validation.npz"],
            "dataset_id": dataset_manifest["run_id"], "max_minutes_per_segment": args.max_minutes,
        }, "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ram_bytes": _total_ram_bytes(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu_count": os.cpu_count(), "resume_count": resume_count,
        "resumed": bool(args.resume), "start_time": started_at,
        "status": "running", "completed_steps": global_step,
    }

    def evaluate(indices):
        model.eval()
        if len(indices) == 0:
            raise ValueError("train and validation splits must be non-empty")
        total = policy_sum = value_sum = 0.0
        correct = count = 0
        with torch.inference_mode():
            for offset in range(0, len(indices), config["batch_size"]):
                ids = torch.as_tensor(indices[offset:offset + config["batch_size"]],
                                      dtype=torch.long, device=device)
                logits, value_pred = model(x[ids])
                policy_loss = -(p[ids] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(value_pred.reshape(-1), v[ids])
                batch_loss = policy_loss + config["value_weight"] * value_loss
                n = len(ids)
                total += float(batch_loss) * n
                policy_sum += float(policy_loss) * n
                value_sum += float(value_loss) * n
                pred = logits.masked_fill(~legal[ids], float("-inf")).argmax(dim=1)
                correct += int(optimal[ids, pred].sum())
                count += n
        return {"loss": total / count, "policy_loss": policy_sum / count,
                "value_mse": value_sum / count, "exact_optimal_top1": correct / count,
                "positions": count}

    reset_peak_vram_stats(torch, device)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_minutes * 60 if args.max_minutes else float("inf")
    last_save = time.monotonic()
    stop_early = False
    epochs = config["epochs"]

    def save_latest() -> None:
        _atomic_torch_save(torch, {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": start_epoch, "cursor": cursor, "order": order,
            "global_step": global_step, "best_val": best_val,
            "best_epoch": best_epoch, "stale_epochs": stale_epochs,
            "best_state_dict": best_state,
            "best_metrics": best_metrics,
            "epoch_loss_sum": epoch_loss_sum, "epoch_samples": epoch_samples,
            "resume_count": resume_count, "started_at": started_at,
            "rng_state": _rng_state(torch), "config": config,
            "dataset_sha256": dataset_manifest["files_sha256"]["train_validation.npz"],
        }, latest_path)

    while start_epoch < epochs:
        if order is None:
            order = torch.randperm(len(train_idx), device="cpu").tolist()
            cursor, epoch_loss_sum, epoch_samples = 0, 0.0, 0
        model.train()
        while cursor < len(order):
            batch_order = order[cursor:cursor + config["batch_size"]]
            ids = torch.as_tensor(train_idx[np.asarray(batch_order)], dtype=torch.long, device=device)
            inputs, targets, values, masks = x[ids], p[ids], v[ids], optimal[ids]
            flip = torch.rand(len(ids), device=device) < 0.5
            if flip.any():
                inputs = inputs.clone(); targets = targets.clone(); masks = masks.clone()
                inputs[flip] = inputs[flip].flip(-1)
                targets[flip] = targets[flip].flip(-1)
                masks[flip] = masks[flip].flip(-1)
            logits, value_pred = model(inputs)
            policy_loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value_pred.reshape(-1), values)
            loss = policy_loss + config["value_weight"] * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(ids)
            epoch_loss_sum += float(loss.detach()) * count
            epoch_samples += count
            cursor += count
            global_step += 1

            now = time.monotonic()
            if now >= deadline:
                stop_early = True
                break
            if now - last_save >= config["checkpoint_interval_seconds"]:
                save_latest()
                last_save = now
        if stop_early:
            break
        train_loss = epoch_loss_sum / max(1, epoch_samples)
        val_metrics = evaluate(val_idx)
        completed_epoch = start_epoch + 1
        row = {"epoch": completed_epoch, "global_step": global_step,
               "train_loss": train_loss, "validation": val_metrics,
               "timestamp_local": datetime.now().astimezone().isoformat()}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
            log.flush(); os.fsync(log.fileno())
        if val_metrics["loss"] < best_val:
            best_val, best_epoch, stale_epochs = val_metrics["loss"], completed_epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_metrics = val_metrics
            _atomic_torch_save(torch, {
                "model_state_dict": best_state,
                "run_id": run_id, "seed": args.seed, "best_epoch": best_epoch,
                "best_validation_loss": best_val,
                "frozen_test_sha256": dataset_manifest["frozen_test_sha256"],
            }, best_path)
        else:
            stale_epochs += 1
        print(json.dumps(row), flush=True)
        start_epoch, cursor, order = completed_epoch, 0, None
        epoch_loss_sum, epoch_samples = 0.0, 0
        if stale_epochs >= config["patience"]:
            break

    save_latest()
    complete = start_epoch >= epochs or stale_epochs >= config["patience"]
    if complete:
        manifest["status"] = "complete"
        manifest["end_time"] = datetime.now().astimezone().isoformat()
        manifest["best_epoch"] = best_epoch
        manifest["best_validation_loss"] = best_val
        manifest["best_validation_metrics"] = best_metrics
    else:
        manifest["status"] = "paused"
        manifest["end_time"] = datetime.now().astimezone().isoformat()
        manifest["pause_time"] = manifest["end_time"]
    manifest["completed_steps"] = global_step
    manifest["resume_count"] = resume_count
    manifest["local_elapsed_seconds_this_segment"] = time.monotonic() - started_clock
    manifest["peak_vram_bytes"] = peak_vram_bytes(torch, device, previous_peak_vram)
    manifest["files_sha256"] = {
        path.name: sha256_file(path) for path in (latest_path, log_path, best_path)
        if path.exists()
    }
    _save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=60)
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.resume == "latest":
        run_id = args.run_id or f"cnn_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
        args.resume = str(Path(args.output_root) / run_id / "checkpoint_latest.pt")
    if args.epochs < 1 or args.batch_size < 1 or args.max_minutes < 0:
        parser.error("epochs/batch-size must be positive; max-minutes cannot be negative")
    result = train(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
