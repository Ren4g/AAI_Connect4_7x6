"""Batched PUCT self-play and resumable AlphaZero-inspired training CLI."""
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

from .agents import CENTER_ORDER, _PUCTNode
from .encoding import encode_planes
from .engine import ConnectFour, GameResult
from .evaluation_data import canonical_state_key, sha256_file
from .model import ConnectFourCNN
from .resource_metrics import peak_vram_bytes, reset_peak_vram_stats


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_save(torch, value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state(torch) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(torch, state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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


def _states_from_histories(histories: list[list[int]]) -> list[ConnectFour]:
    states = []
    for history in histories:
        state = ConnectFour()
        for action in history:
            state.step(int(action))
        states.append(state)
    return states


class BatchedPUCT:
    """Synchronous PUCT over independent roots, batching leaf inference."""

    def __init__(self, model, device, simulations: int, c_puct: float,
                 dirichlet_alpha: float, dirichlet_epsilon: float):
        self.model, self.device = model, device
        self.simulations, self.c_puct = simulations, c_puct
        self.dirichlet_alpha, self.dirichlet_epsilon = dirichlet_alpha, dirichlet_epsilon

    def _infer(self, states: list[ConnectFour]):
        import torch
        x = torch.as_tensor(np.stack([
            encode_planes(state.board, int(state.current_player)) for state in states
        ]), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits, values = self.model(x)
        return logits.detach().cpu().numpy(), values.detach().cpu().numpy()

    def _expand(self, node: _PUCTNode, logits: np.ndarray) -> None:
        legal = node.state.legal_actions()
        maximum = max(float(logits[action]) for action in legal)
        weights = {a: float(np.exp(np.clip(float(logits[a]) - maximum, -80, 0))) for a in legal}
        normalizer = sum(weights.values())
        for action in legal:
            child_state = node.state.clone()
            child_state.step(action)
            node.children[action] = _PUCTNode(child_state, node, action,
                                              weights[action] / normalizer)

    def _score(self, child: _PUCTNode) -> float:
        parent = child.parent
        assert parent is not None
        if child.visits:
            q = child.value_sum / child.visits
            if child.state.current_player != parent.state.current_player:
                q = -q
        else:
            q = 0.0
        return q + self.c_puct * child.prior * np.sqrt(max(parent.visits, 1)) / (1 + child.visits)

    @staticmethod
    def _terminal_value(state: ConnectFour) -> float:
        result = state.result()
        if result is GameResult.DRAW:
            return 0.0
        return 1.0 if result is GameResult(int(state.current_player)) else -1.0

    def search(self, states: list[ConnectFour], temperature: float | list[float],
               rng: np.random.Generator) -> tuple[list[list[float]], list[int]]:
        roots = [_PUCTNode(state.clone()) for state in states]
        logits, _ = self._infer([root.state for root in roots])
        for root, policy_logits in zip(roots, logits):
            self._expand(root, policy_logits)
            if root.children and self.dirichlet_epsilon:
                actions = list(root.children)
                noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
                for action, sample in zip(actions, noise):
                    child = root.children[action]
                    child.prior = ((1 - self.dirichlet_epsilon) * child.prior
                                   + self.dirichlet_epsilon * float(sample))

        for _ in range(self.simulations):
            leaves: list[_PUCTNode] = []
            paths: list[list[_PUCTNode]] = []
            terminal_values: list[float | None] = []
            for root in roots:
                node = root
                path = [node]
                while node.children and not node.state.is_terminal():
                    node = max(node.children.values(), key=self._score)
                    path.append(node)
                paths.append(path)
                if node.state.is_terminal():
                    leaves.append(node)
                    terminal_values.append(self._terminal_value(node.state))
                else:
                    leaves.append(node)
                    terminal_values.append(None)

            pending = [i for i, value in enumerate(terminal_values) if value is None]
            if pending:
                batch_logits, batch_values = self._infer([leaves[i].state for i in pending])
                for row, path_index in enumerate(pending):
                    self._expand(leaves[path_index], batch_logits[row])
                    terminal_values[path_index] = float(np.clip(batch_values[row], -1, 1))

            for path, value in zip(paths, terminal_values):
                assert value is not None
                leaf = path[-1]
                leaf.visits += 1
                leaf.value_sum += value
                for node, parent in zip(reversed(path[1:]), reversed(path[:-1])):
                    # Convert the child-side value into the parent's side-to-move perspective.
                    if node.state.current_player != parent.state.current_player:
                        value = -value
                    parent.visits += 1
                    parent.value_sum += value
                # A root path has no parent; record its value once as well.
                if len(path) == 1:
                    root = path[0]
                    root.visits += 1
                    root.value_sum += value

        temperatures = ([float(temperature)] * len(roots)
                        if isinstance(temperature, (int, float)) else temperature)
        policies, actions = [], []
        for root, root_temperature in zip(roots, temperatures):
            legal = root.state.legal_actions()
            visits = np.asarray([root.children[a].visits for a in legal], dtype=np.float64)
            if visits.sum() == 0:
                visits = np.ones_like(visits)
            if root_temperature <= 1e-6:
                probs = np.zeros_like(visits)
                probs[int(np.argmax(visits))] = 1.0
            else:
                powered = visits ** (1.0 / root_temperature)
                probs = powered / powered.sum()
            policy = [0.0] * 7
            for action, probability in zip(legal, probs):
                policy[action] = float(probability)
            selected = int(rng.choice(legal, p=probs))
            policies.append(policy)
            actions.append(selected)
        return policies, actions


def train(args) -> dict:
    import torch
    from torch.nn import functional as F

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset_dir = Path(args.dataset)
    data_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen_path = dataset_dir / "frozen_test_hashes.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_keys = set(frozen["state_sha256_mirror_canonical"])
    for name, expected in data_manifest["files_sha256"].items():
        if sha256_file(dataset_dir / name) != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    if frozen["positions_sha256"] != data_manifest["frozen_test_sha256"]:
        raise ValueError("Frozen test manifest mismatch")

    run_id = args.run_id or f"az_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    replay: list[dict] = []
    training_state_hashes: set[str] = set()
    active: list[dict] = []
    examples: list[dict] = []
    iteration, games_completed, games_started = 0, 0, 0
    train_steps_done = 0
    training_step_cursor = 0
    pending_train_losses: list[float] = []
    iteration_game_plies: list[int] = []
    iteration_game_results: list[int] = []
    iteration_game_actions: list[list[int]] = []
    excluded_test_positions = 0
    resume_count = 0
    started_at = datetime.now().astimezone().isoformat()
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(initial.get("model_state_dict", initial))
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["dataset_manifest_sha256"] != sha256_file(dataset_dir / "manifest.json"):
            raise ValueError("Resume checkpoint has a different frozen dataset")
        if state["config"] != args.config:
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        replay, active = state["replay"], state["active"]
        training_state_hashes = set(state["training_state_hashes"])
        examples = state["examples"]
        iteration, games_completed, games_started = state["iteration"], state["games_completed"], state["games_started"]
        train_steps_done = state["train_steps_done"]
        training_step_cursor = state["training_step_cursor"]
        pending_train_losses = state["pending_train_losses"]
        iteration_game_plies = state["iteration_game_plies"]
        iteration_game_results = state["iteration_game_results"]
        iteration_game_actions = state["iteration_game_actions"]
        excluded_test_positions = state["excluded_test_positions"]
        resume_count = state["resume_count"] + 1
        started_at = state["started_at"]
        _restore_rng(torch, state["rng"])

    config = args.config
    search = BatchedPUCT(model, device, config["simulations"], config["c_puct"],
                         config["dirichlet_alpha"], config["dirichlet_epsilon"])
    target_iterations = config["iterations"]
    games_per_iteration = config["games_per_iteration"]
    reset_peak_vram_stats(torch, device)
    started_clock = time.monotonic()
    deadline = started_clock + args.max_minutes * 60 if args.max_minutes else float("inf")
    status = "running"
    last_save = time.monotonic()

    def save(status_value: str) -> dict:
        checkpoint = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "replay": replay, "active": active, "examples": examples,
            "training_state_hashes": sorted(training_state_hashes),
            "iteration": iteration, "games_completed": games_completed,
            "games_started": games_started, "train_steps_done": train_steps_done,
            "training_step_cursor": training_step_cursor,
            "pending_train_losses": pending_train_losses,
            "iteration_game_plies": iteration_game_plies,
            "iteration_game_results": iteration_game_results,
            "iteration_game_actions": iteration_game_actions,
            "excluded_test_positions": excluded_test_positions, "resume_count": resume_count,
            "started_at": started_at, "rng": _rng_state(torch), "config": config,
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        }
        _atomic_save(torch, checkpoint, latest_path)
        _atomic_save(torch, {"model_state_dict": model.state_dict(), "run_id": run_id,
                             "iteration": iteration, "seed": args.seed,
                             "frozen_test_sha256": data_manifest["frozen_test_sha256"]}, policy_path)
        manifest = {
            "schema_version": 1, "run_id": run_id, "phase": "alphazero_selfplay",
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
            "frozen_test_sha256": data_manifest["frozen_test_sha256"],
            "test_overlap_count": 0, "excluded_test_positions": excluded_test_positions,
            "training_position_states": len(training_state_hashes),
            "source_code_commit": _git_commit(Path.cwd()), "source_code_sha256": _source_hash(Path.cwd()),
            "solver_commit": data_manifest.get("solver_commit"), "seed": args.seed,
            "configuration": {**config, "dataset_id": data_manifest["run_id"],
                               "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
                               "max_minutes_per_segment": args.max_minutes},
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cpu_count": os.cpu_count(), "ram_bytes": _total_ram_bytes(), "resume_count": resume_count,
            "resumed": bool(args.resume), "start_time": started_at,
            "status": status_value, "completed_steps": train_steps_done,
            "local_elapsed_seconds_this_segment": time.monotonic() - started_clock,
            "peak_vram_bytes": peak_vram_bytes(torch, device, previous_peak_vram),
            "iteration": iteration, "games_completed": games_completed,
            "end_time": datetime.now().astimezone().isoformat(),
            "pause_time": datetime.now().astimezone().isoformat() if status_value == "paused" else None,
            "files_sha256": {p.name: sha256_file(p) for p in (latest_path, policy_path, log_path) if p.exists()},
        }
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp, manifest_path)
        return manifest

    while iteration < target_iterations:
        model.eval()
        while games_completed < (iteration + 1) * games_per_iteration:
            while len(active) < config["parallel_games"] and games_started < (iteration + 1) * games_per_iteration:
                active.append({"moves": [], "rows": []})
                games_started += 1
            states = _states_from_histories([game["moves"] for game in active])
            temperatures = [1.0 if len(game["moves"]) < config["temperature_moves"] else 0.0
                            for game in active]
            policies, actions = search.search(states, temperatures,
                                               np.random.default_rng(random.randrange(2**32)))
            keep_active = []
            for game, state, policy, action in zip(active, states, policies, actions):
                state_key = canonical_state_key(game["moves"])
                if state_key not in frozen_keys:
                    game["rows"].append({"features": encode_planes(state.board, int(state.current_player)),
                                         "policy": policy, "player": int(state.current_player),
                                         "state_sha256_mirror_canonical": state_key})
                else:
                    excluded_test_positions += 1
                state.step(int(action)); game["moves"].append(int(action))
                if state.is_terminal():
                    result = state.result()
                    winner = int(result) if result in (GameResult.PLAYER_ONE, GameResult.PLAYER_TWO) else 0
                    for row in game["rows"]:
                        row["value"] = float((winner > 0) - (winner < 0)) if winner == 0 else (1.0 if winner == row["player"] else -1.0)
                        replay.append(row)
                        training_state_hashes.add(row["state_sha256_mirror_canonical"])
                    games_completed += 1
                    iteration_game_plies.append(len(game["moves"]))
                    iteration_game_results.append(winner)
                    iteration_game_actions.append(list(game["moves"]))
                else:
                    keep_active.append(game)
            active = keep_active
            if len(replay) > config["replay_capacity"]:
                replay = replay[-config["replay_capacity"]:]
            if time.monotonic() >= deadline:
                status = "paused"
                save(status)
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            if time.monotonic() - last_save >= config["checkpoint_interval_seconds"]:
                save("running")
                last_save = time.monotonic()

        # Train from the replay buffer; random sampling and augmentations are checkpointed RNGs.
        model.train()
        for _ in range(training_step_cursor, config["training_steps"]):
            if not replay:
                raise RuntimeError("Self-play produced no training positions")
            batch = random.choices(replay, k=config["batch_size"])
            x = np.stack([row["features"] for row in batch]).astype(np.float32)
            p = np.asarray([row["policy"] for row in batch], dtype=np.float32)
            v = np.asarray([row["value"] for row in batch], dtype=np.float32)
            flip = np.random.random(len(batch)) < 0.5
            x[flip] = x[flip, :, :, ::-1].copy()
            p[flip] = p[flip, ::-1].copy()
            tx = torch.as_tensor(x, device=device)
            tp = torch.as_tensor(p, device=device)
            tv = torch.as_tensor(v, device=device)
            logits, value_pred = model(tx)
            policy_loss = -(tp * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred.reshape(-1), tv)
            loss = policy_loss + config["value_weight"] * value_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            pending_train_losses.append(float(loss.detach().cpu()))
            train_steps_done += 1
            training_step_cursor += 1
            if time.monotonic() >= deadline:
                save("paused")
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            if time.monotonic() - last_save >= config["checkpoint_interval_seconds"]:
                save("running")
                last_save = time.monotonic()

        iteration += 1
        training_step_cursor = 0
        row = {"iteration": iteration, "games_completed": games_completed,
               "replay_positions": len(replay), "training_steps": config["training_steps"],
               "mean_loss": float(np.mean(pending_train_losses)),
               "mean_game_plies": float(np.mean(iteration_game_plies)) if iteration_game_plies else None,
               "selfplay_wins_player1": sum(result == 1 for result in iteration_game_results),
               "selfplay_wins_player2": sum(result == 2 for result in iteration_game_results),
               "selfplay_draws": sum(result == 0 for result in iteration_game_results),
               "game_actions": iteration_game_actions,
               "timestamp_local": datetime.now().astimezone().isoformat()}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n"); log.flush(); os.fsync(log.fileno())
        print(json.dumps(row), flush=True)
        games_started = games_completed
        pending_train_losses = []
        iteration_game_plies = []
        iteration_game_results = []
        iteration_game_actions = []
        if time.monotonic() >= deadline:
            save("paused")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iteration", type=int, default=32)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.games_per_iteration = 1, 2
        args.parallel_games, args.simulations = 2, 2
        args.training_steps, args.batch_size, args.replay_capacity = 2, 4, 128
    if args.max_minutes < 0 or min(args.iterations, args.games_per_iteration, args.parallel_games,
                                    args.simulations, args.training_steps, args.batch_size) < 1:
        parser.error("all work limits must be positive")
    args.config = {"iterations": args.iterations, "games_per_iteration": args.games_per_iteration,
                   "parallel_games": args.parallel_games, "simulations": args.simulations,
                   "training_steps": args.training_steps, "batch_size": args.batch_size,
                   "replay_capacity": args.replay_capacity, "learning_rate": args.learning_rate,
                   "value_weight": 0.5, "c_puct": 1.5, "dirichlet_alpha": 0.3,
                   "dirichlet_epsilon": 0.25, "temperature_moves": 12,
                   "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
                   "smoke": args.smoke,
                   "init_checkpoint_sha256": sha256_file(Path(args.init_checkpoint)) if args.init_checkpoint else None}
    if args.resume == "latest":
        run_id = args.run_id or f"az_{datetime.now().strftime('%Y%m%d')}_seed{args.seed}"
        args.resume = str(Path(args.output_root) / run_id / "checkpoint_latest.pt")
    result = train(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
