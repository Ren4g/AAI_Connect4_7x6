"""Small REINFORCE/actor-critic experiment against Random, resumable on Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .encoding import encode_planes
from .engine import ConnectFour, GameResult
from .evaluation_data import canonical_state_key, sha256_file
from .model import ConnectFourCNN
from .resource_metrics import peak_vram_bytes, reset_peak_vram_stats


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _save(torch_module, data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(data, temporary)
    os.replace(temporary, path)


def _rng(torch_module) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch_module.get_rng_state(),
            "torch_cuda": torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else None}


def _restore_rng(torch_module, state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch_module.cuda.is_available():
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def train(args) -> dict:
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError("Install PyTorch to run the policy-gradient trainer") from error
    torch_module = torch
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch_module.manual_seed(seed)
    if torch_module.cuda.is_available(): torch_module.cuda.manual_seed_all(seed)
    device = torch_module.device(args.device)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = Path(args.dataset)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in dataset_manifest["files_sha256"].items():
        if sha256_file(dataset / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    frozen = json.loads((dataset / "frozen_test_hashes.json").read_text(encoding="utf-8"))
    if frozen["positions_sha256"] != dataset_manifest["frozen_test_sha256"]:
        raise ValueError("Frozen test hash differs from dataset manifest")
    frozen_keys = set(frozen["state_sha256_mirror_canonical"])
    dataset_sha = sha256_file(dataset / "manifest.json")

    run_id = args.run_id or f"rl_{datetime.now().strftime('%Y%m%d')}_seed{seed}"
    output = Path(args.output_root) / run_id
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    latest_path, policy_path = output / "checkpoint_latest.pt", output / "checkpoint_policy.pt"
    log_path, manifest_path = output / "train_log.jsonl", output / "manifest.json"
    previous_peak_vram = 0
    if args.resume and manifest_path.exists():
        previous_peak_vram = int(json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "peak_vram_bytes", 0
        ))
    log_path.touch(exist_ok=True)
    model = ConnectFourCNN().to(device)
    optimizer = torch_module.optim.Adam(model.parameters(), lr=args.learning_rate)
    episode = 0
    wins = losses = draws = excluded_positions = resume_count = 0
    baseline = 0.0
    training_state_hashes: set[str] = set()
    started_at = datetime.now().astimezone().isoformat()
    if args.init_checkpoint:
        initial = torch_module.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(initial.get("model_state_dict", initial))
    if args.resume:
        checkpoint = torch_module.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["dataset_manifest_sha256"] != dataset_sha or checkpoint["config"] != args.config:
            raise ValueError("Resume data/config differs from the saved RL run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode, wins, losses, draws = (checkpoint[k] for k in ("episode", "wins", "losses", "draws"))
        excluded_positions, baseline = checkpoint["excluded_positions"], checkpoint["baseline"]
        training_state_hashes = set(checkpoint["training_state_hashes"])
        resume_count = checkpoint["resume_count"] + 1
        started_at = checkpoint["started_at"]
        _restore_rng(torch_module, checkpoint["rng"])

    reset_peak_vram_stats(torch_module, device)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_minutes * 60 if args.max_minutes else float("inf")
    last_save = time.monotonic()

    def save(status: str) -> dict:
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "episode": episode, "wins": wins, "losses": losses, "draws": draws,
                 "excluded_positions": excluded_positions, "baseline": baseline,
                 "training_state_hashes": sorted(training_state_hashes),
                 "resume_count": resume_count, "started_at": started_at,
                 "rng": _rng(torch_module), "config": args.config,
                 "dataset_manifest_sha256": dataset_sha}
        _save(torch_module, state, latest_path)
        _save(torch_module, {"model_state_dict": model.state_dict(), "run_id": run_id,
                             "episode": episode, "seed": seed,
                             "frozen_test_sha256": dataset_manifest["frozen_test_sha256"]}, policy_path)
        files = {p.name: sha256_file(p) for p in (latest_path, policy_path, log_path)}
        manifest = {
            "schema_version": 1, "run_id": run_id, "phase": "rl_policy_gradient",
            "dataset_manifest_sha256": dataset_sha,
            "frozen_test_sha256": dataset_manifest["frozen_test_sha256"],
            "test_overlap_count": 0, "excluded_test_positions": excluded_positions,
            "source_code_commit": _commit(), "source_code_sha256": _tree_hash(Path.cwd()),
            "solver_commit": dataset_manifest.get("solver_commit"), "seed": seed,
            "configuration": {**args.config, "dataset_id": dataset_manifest["run_id"],
                               "dataset_manifest_sha256": dataset_sha,
                               "max_minutes_per_segment": args.max_minutes,
                               "init_checkpoint_sha256": sha256_file(Path(args.init_checkpoint)) if args.init_checkpoint else None},
            "python": platform.python_version(), "torch": torch_module.__version__,
            "cuda": torch_module.version.cuda, "gpu_name": torch_module.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cpu_count": os.cpu_count(), "ram_bytes": _ram_bytes(), "resume_count": resume_count,
            "resumed": bool(args.resume), "start_time": started_at,
            "status": status, "completed_steps": episode, "end_time": datetime.now().astimezone().isoformat(),
            "pause_time": datetime.now().astimezone().isoformat() if status == "paused" else None,
            "local_elapsed_seconds_this_segment": time.monotonic() - started_clock,
            "peak_vram_bytes": peak_vram_bytes(torch_module, device, previous_peak_vram),
            "files_sha256": files,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
        return manifest

    while episode < args.config["episodes"]:
        side = 1 if episode % 2 == 0 else 2
        state = ConnectFour()
        history: list[int] = []
        records = []
        model.eval()
        while not state.is_terminal():
            legal = state.legal_actions()
            if state.current_player == side:
                key = canonical_state_key(history)
                training_row = None
                if key not in frozen_keys:
                    training_row = {"features": encode_planes(state.board, side),
                                    "legal": legal, "player": side,
                                    "state_sha256_mirror_canonical": key}
                    records.append(training_row)
                    training_state_hashes.add(key)
                else:
                    excluded_positions += 1
                x = torch_module.as_tensor(encode_planes(state.board, side), dtype=torch_module.float32,
                                           device=device).unsqueeze(0)
                with torch_module.inference_mode():
                    logits, _ = model(x)
                    masked = logits[0].clone()
                    mask = torch_module.ones(7, dtype=torch_module.bool, device=device)
                    mask[legal] = False
                    masked[mask] = float("-inf")
                    probs = torch_module.softmax(masked, dim=0)
                    action = int(torch_module.multinomial(probs, 1).item())
                    if training_row is not None:
                        training_row["action"] = action
            else:
                action = random.choice(legal)
            state.step(action)
            history.append(action)

        result = state.result()
        reward = 0.0 if result is GameResult.DRAW else (1.0 if int(result) == side else -1.0)
        wins += int(reward > 0); losses += int(reward < 0); draws += int(reward == 0)
        if records:
            xs = torch_module.as_tensor(np.stack([row["features"] for row in records]),
                                        dtype=torch_module.float32, device=device)
            legal_masks = torch_module.zeros((len(records), 7), dtype=torch_module.bool, device=device)
            for index, row in enumerate(records): legal_masks[index, row["legal"]] = True
            actions = torch_module.as_tensor([row["action"] for row in records], dtype=torch_module.long, device=device)
            logits, values = model(xs)
            masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
            log_probs = torch_module.log_softmax(masked_logits, dim=1)
            selected_log_probs = log_probs.gather(1, actions[:, None]).squeeze(1)
            advantage = torch_module.full_like(values, reward - baseline)
            entropy = -(torch_module.softmax(masked_logits, dim=1) * log_probs.nan_to_num()).sum(dim=1).mean()
            policy_loss = -(selected_log_probs * advantage).mean()
            value_loss = F.mse_loss(values, torch_module.full_like(values, reward))
            loss = policy_loss + args.config["value_weight"] * value_loss - args.config["entropy_weight"] * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            baseline = 0.95 * baseline + 0.05 * reward
        else:
            loss = torch_module.tensor(0.0)
        episode += 1
        model.train()
        row = {"episode": episode, "reward": reward, "loss": float(loss.detach().cpu()),
               "wins": wins, "losses": losses, "draws": draws,
               "agent_side": side, "actions": history,
               "excluded_positions": excluded_positions,
               "timestamp_local": datetime.now().astimezone().isoformat()}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n"); log.flush(); os.fsync(log.fileno())
        if time.monotonic() >= deadline:
            return save("paused")
        if time.monotonic() - last_save >= args.config["checkpoint_interval_seconds"]:
            save("running"); last_save = time.monotonic()

    return save("complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke: args.episodes = 2
    if args.episodes < 1 or args.max_minutes < 0: parser.error("episodes must be positive; time nonnegative")
    args.config = {"episodes": args.episodes, "learning_rate": args.learning_rate,
                   "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
                   "value_weight": 0.5, "entropy_weight": 0.01, "smoke": args.smoke,
                   "init_checkpoint_sha256": sha256_file(Path(args.init_checkpoint)) if args.init_checkpoint else None}
    if args.resume == "latest":
        run_id = args.run_id or f"rl_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
        args.resume = str(Path(args.output_root) / run_id / "checkpoint_latest.pt")
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
