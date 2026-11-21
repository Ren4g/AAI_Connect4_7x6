"""Measure local agent peak RSS in isolated full-game runs; summarize Colab VRAM manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_agent(name: str, checkpoint: str | None):
    from connect4.agents import (AlphaBetaAgent, HeuristicAgent, MCTSAgent,
                                 PUCTAgent, RandomAgent, PolicyAgent)

    if name == "Random":
        return RandomAgent(seed=20261121)
    if name == "Heuristic":
        return HeuristicAgent()
    if name == "AlphaBeta-depth4":
        return AlphaBetaAgent(depth=4)
    if name == "AlphaBeta-depth6":
        return AlphaBetaAgent(depth=6)
    if name == "MCTS-500":
        return MCTSAgent(simulations=500, seed=20261124)
    if name == "CNN-Policy":
        return PolicyAgent.from_checkpoint(checkpoint, device="cpu")
    if name == "RL-Policy":
        return PolicyAgent.from_checkpoint(checkpoint, device="cpu")
    if name == "PUCT-500":
        return PUCTAgent.from_checkpoint(checkpoint, simulations=500, device="cpu")
    raise ValueError(f"Unknown agent: {name}")


def _worker(name: str, checkpoint: str | None) -> int:
    from connect4.agents import HeuristicAgent
    from connect4.experiments import play_game
    from connect4.engine_bit import ConnectFour

    agent = _make_agent(name, checkpoint)
    print(json.dumps({"ready": True}), flush=True)
    if sys.stdin.readline().strip() != "GO":
        return 2
    game = play_game(agent, HeuristicAgent(), name, "Heuristic", engine_factory=ConnectFour)
    print(json.dumps({
        "result": game.result.value,
        "moves": game.moves,
        "measured_agent_decisions": sum(1 for player in game.move_players if player == 1),
        "action_count": len(game.actions),
    }), flush=True)
    return 0


def measure_one(name: str, checkpoint: str | None, interval: float) -> dict:
    venv_python = REPO_ROOT / ".venv-gpu" / "Scripts" / "python.exe"
    worker_python = str(venv_python) if venv_python.is_file() else sys.executable
    command = [worker_python, str(Path(__file__).resolve()), "--worker", name]
    if checkpoint:
        command += ["--worker-checkpoint", checkpoint]
    child = subprocess.Popen(command, cwd=REPO_ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        ready_line = child.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"{name} worker failed to initialize: {child.stderr.read()}")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"{name} worker did not become ready")
        baseline = _rss_for_pid(child.pid)
        peak = baseline
        child.stdin.write("GO\n")
        child.stdin.flush()
        started = time.perf_counter()
        while child.poll() is None:
            try:
                peak = max(peak, _rss_for_pid(child.pid))
            except ProcessLookupError:
                break
            time.sleep(interval)
        remainder = child.stdout.read()
        stderr = child.stderr.read()
        if child.returncode:
            raise RuntimeError(f"{name} worker exited {child.returncode}: {stderr}")
        outcome = json.loads(remainder.strip().splitlines()[-1])
        peak = max(peak, baseline)
        return {
            "agent": name,
            "measurement": "isolated process-tree peak RSS sampled during one complete game",
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "peak_increment_bytes": max(0, peak - baseline),
            "peak_rss_mib": round(peak / (1024 ** 2), 2),
            "peak_increment_mib": round(max(0, peak - baseline) / (1024 ** 2), 2),
            "sampling_interval_ms": interval * 1000,
            "game_seconds": round(time.perf_counter() - started, 3),
            **outcome,
            "checkpoint": checkpoint,
        }
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def _rss_for_pid(pid: int) -> int:
    import psutil
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(int(process.memory_info().rss) for process in processes
                   if process.is_running())
    except psutil.NoSuchProcess as error:
        raise ProcessLookupError(pid) from error


def colab_memory(manifest_paths: list[Path]) -> list[dict]:
    rows = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run_id": manifest["run_id"],
            "phase": manifest["phase"],
            "gpu": manifest.get("gpu_name"),
            "peak_vram_bytes": manifest.get("peak_vram_bytes"),
            "peak_vram_mib": (round(manifest["peak_vram_bytes"] / (1024 ** 2), 2)
                              if manifest.get("peak_vram_bytes") is not None else None),
            "ram_bytes": manifest.get("ram_bytes"),
            "ram_mib": (round(manifest["ram_bytes"] / (1024 ** 2), 2)
                        if manifest.get("ram_bytes") is not None else None),
            "manifest": path.as_posix(),
        })
    return rows


def render_report(data: dict) -> str:
    lines = [
        "# Đo tài nguyên agent",
        "",
        f"Ngày đo local: {data['timestamp_local']}; commit nền: `{data['source_commit']}`; SHA256 công cụ: `{data['script_sha256']}`.",
        "",
        "## RAM tiến trình local",
        "",
        "Mỗi agent chạy riêng trong một cây tiến trình qua một ván đầy đủ trên bitboard, đấu Heuristic. Working Set cộng trên tiến trình launcher và các tiến trình con được lấy mẫu mỗi 100 ms sau khi agent khởi tạo; peak gồm runtime, model và agent. `peak_increment` là phần tăng thêm so với giá trị ngay trước ván. Windows có thể đếm trùng trang chia sẻ giữa các tiến trình trong cây, nên tổng Working Set là giới hạn trên gần đúng. Một ván cho mỗi agent là phép đo mô tả, không phải benchmark sức chơi.",
        "",
        "| Agent | RSS trước ván (MiB) | Peak RSS (MiB) | Phần tăng (MiB) | Số quyết định | Thời gian ván (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in data["local_agents"]:
        lines.append(
            f"| {row['agent']} | {row['baseline_rss_bytes'] / (1024 ** 2):.2f} | "
            f"{row['peak_rss_mib']:.2f} | {row['peak_increment_mib']:.2f} | "
            f"{row['measured_agent_decisions']} | {row['game_seconds']:.3f} |"
        )
    lines += [
        "",
        "## Tài nguyên run Colab",
        "",
        "RAM là giá trị máy khả dụng ghi trong manifest khi job chạy; VRAM là peak allocated ghi bởi PyTorch khi manifest có trường tương ứng. Thiếu VRAM peak được ghi là chưa được đo, không suy ra từ model size.",
        "",
        "| Run | Nhánh | GPU | RAM máy (MiB) | Peak VRAM allocated (MiB) |",
        "|---|---|---|---:|---:|",
    ]
    for row in data["colab_runs"]:
        ram = f"{row['ram_mib']:.0f}" if row["ram_mib"] is not None else "không ghi"
        vram = f"{row['peak_vram_mib']:.2f}" if row["peak_vram_mib"] is not None else "không đo/không ghi"
        lines.append(f"| `{row['run_id']}` | {row['phase']} | {row['gpu'] or 'không ghi'} | {ram} | {vram} |")
    lines += [
        "",
        "Số liệu local và Colab thuộc hai máy khác nhau; chỉ RAM/VRAM Colab phản ánh lúc huấn luyện, còn RSS local phản ánh tiến trình suy luận/đấu ván trên máy cố định của dự án.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-checkpoint", type=Path, default=Path("results/colab/cnn_seed20261121_colab1/checkpoint_best.pt"))
    parser.add_argument("--rl-checkpoint", type=Path, default=Path("results/colab/rl_seed20261120_run1/checkpoint_policy.pt"))
    parser.add_argument("--puct-checkpoint", type=Path, default=Path("results/colab/az_seed20261120_run1/checkpoint_policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/resource_profile_20261121.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/RESOURCE_PROFILE.md"))
    parser.add_argument("--interval-ms", type=float, default=100)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(_worker(args.worker, args.worker_checkpoint))
    agents = [
        ("Random", None), ("Heuristic", None), ("AlphaBeta-depth4", None),
        ("MCTS-500", None), ("AlphaBeta-depth6", None),
        ("CNN-Policy", str(args.cnn_checkpoint)), ("RL-Policy", str(args.rl_checkpoint)),
        ("PUCT-500", str(args.puct_checkpoint)),
    ]
    for _, checkpoint in agents:
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Resource outputs are immutable; choose new paths")
    local = [measure_one(name, checkpoint, args.interval_ms / 1000) for name, checkpoint in agents]
    manifests = sorted(Path("results/colab").glob("*/manifest.json"))
    data = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True, check=False).stdout.strip() or None,
        "source_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_machine": {"python": sys.version, "platform": sys.platform},
        "local_agents": local,
        "colab_runs": colab_memory(manifests),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    args.report.write_text(render_report(data), encoding="utf-8")
    print(f"Wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
