"""Run the registered Phase C V2 warm/scratch jobs locally with resumable segments."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "colab/data/solverlabels_20261121_seed20261121_v3_30k"
WARM_CHECKPOINT = ROOT / "checkpoints/cnn_v3_30k_best.pt"
OUTPUT = ROOT / "results/phase_c_local"
LEDGER = OUTPUT / "phase_c_local_ledger.json"
RUNS = (
    ("warm", "az_v2_warm_seed20261121", 20261121, WARM_CHECKPOINT),
    ("scratch", "az_v2_scratch_seed20261122", 20261122, None),
)
SEGMENT_MINUTES = 15


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_ledger(threads: int) -> dict:
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        if ledger.get("cpu_threads") != threads:
            raise ValueError("CPU thread count changed; preserve the registered local config")
        pending = ledger.get("pending_segment")
        if pending:
            pending["recovered_at"] = now()
            pending["accounted_seconds"] = pending["reserved_seconds"]
            pending["state"] = "interrupted_reservation_charged_conservatively"
            ledger.setdefault("segments", []).append(pending)
            ledger["pending_segment"] = None
            write_json_atomic(LEDGER, ledger)
        return ledger
    return {
        "schema_version": 1,
        "phase": "C V2 local execution",
        "created_at": now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_threads": threads,
        "phase_cap": None,
        "run_ids": [run_id for _, run_id, _, _ in RUNS],
        "segments": [],
        "pending_segment": None,
    }


def system_sample() -> dict:
    sample = {"timestamp": now()}
    try:
        import psutil
        sample["system_cpu_percent"] = psutil.cpu_percent(interval=None)
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if result.returncode == 0:
            sample["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return sample


def run_one(mode: str, threads: int, segment_minutes: float) -> None:
    selected = next((entry for entry in RUNS if entry[0] == mode), None)
    if selected is None:
        raise ValueError(f"unknown run mode: {mode}")
    _, run_id, seed, init_checkpoint = selected
    run_dir = OUTPUT / run_id
    latest = run_dir / "checkpoint_latest.pt"
    manifest_path = run_dir / "manifest.json"
    log_dir = OUTPUT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not DATASET.is_dir():
        raise FileNotFoundError(f"dataset package missing: {DATASET}")
    dataset_manifest = DATASET / "manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    if mode == "warm" and (init_checkpoint is None or not init_checkpoint.is_file()):
        raise FileNotFoundError(f"validation-selected warm checkpoint missing: {init_checkpoint}")
    if run_dir.exists() and not latest.is_file():
        raise RuntimeError(f"run directory exists without a resumable checkpoint: {run_dir}")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != run_id or previous.get("seed") != seed:
            raise RuntimeError(f"run identity mismatch in {manifest_path}")
        if previous.get("status") == "complete":
            print(f"{run_id}: already complete at iteration {previous.get('iteration')}", flush=True)
            return

    threads_env = str(threads)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = threads_env
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"

    command = [
        sys.executable, "-m", "connect4.selfplay",
        "--dataset", str(DATASET), "--output-root", str(OUTPUT),
        "--run-id", run_id, "--seed", str(seed), "--device", "cuda",
        "--cpu-threads", str(threads), "--iterations", "40",
        "--games-per-iteration", "32", "--parallel-games", "16",
        "--simulations", "400", "--training-steps", "200",
        "--batch-size", "128", "--snapshot-interval", "5",
        "--checkpoint-interval-seconds", "120",
        "--max-minutes", str(segment_minutes),
    ]
    if init_checkpoint:
        command += ["--init-checkpoint", str(init_checkpoint)]

    ledger = load_ledger(threads)
    segment_index = 1 + sum(seg.get("run_id") == run_id for seg in ledger["segments"])
    while True:
        resume = latest.is_file()
        if resume:
            command_segment = command + ["--resume"]
        else:
            command_segment = command.copy()
        log_path = log_dir / f"{run_id}_segment_{segment_index:04d}.log"
        reservation = {
            "run_id": run_id, "segment": segment_index,
            "reserved_seconds": int(segment_minutes * 60),
            "started_at": now(), "state": "reserved",
            "command": command_segment,
        }
        ledger["pending_segment"] = reservation
        write_json_atomic(LEDGER, ledger)
        print(f"Starting {run_id} segment {segment_index}; resume={resume}; "
              f"threads={threads}; log={log_path.relative_to(ROOT)}", flush=True)
        started = time.monotonic()
        process = None
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.Popen(
                    command_segment, cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    time.sleep(30)
                    sample = system_sample()
                    reservation.setdefault("resource_samples", []).append(sample)
                    write_json_atomic(LEDGER, ledger)
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                    print(f"{run_id}: elapsed={(time.monotonic()-started)/60:.1f}m; "
                          f"status={manifest.get('status','starting')}; "
                          f"iteration={manifest.get('iteration','?')}; "
                          f"games={manifest.get('games_completed','?')}; "
                          f"resources={sample}", flush=True)
                code = process.returncode
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reservation["state"] = "interrupted_reservation_charged_conservatively"
            reservation["recovered_at"] = now()
            ledger["segments"].append(reservation)
            ledger["pending_segment"] = None
            write_json_atomic(LEDGER, ledger)
            raise

        elapsed = time.monotonic() - started
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        reservation.update({
            "ended_at": now(), "elapsed_seconds": elapsed,
            "accounted_seconds": elapsed, "exit_code": code,
            "status": manifest.get("status", "missing_manifest"),
            "iteration": manifest.get("iteration"),
            "games_completed": manifest.get("games_completed"),
            "log": log_path.relative_to(ROOT).as_posix(),
        })
        ledger["segments"].append(reservation)
        ledger["pending_segment"] = None
        write_json_atomic(LEDGER, ledger)
        segment_index += 1
        if code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"{run_id} exited {code}; inspect {log_path}\n{tail}")
        if manifest.get("status") == "complete":
            if manifest.get("iteration") != 40:
                raise RuntimeError(f"{run_id} reports complete at unexpected iteration")
            print(f"Completed {run_id}: iteration 40, {manifest.get('games_completed')} games", flush=True)
            break
        if manifest.get("status") != "paused" or not latest.is_file():
            raise RuntimeError(f"{run_id} did not complete or pause safely; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("warm", "scratch", "all"), default="all")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--segment-minutes", type=float, default=SEGMENT_MINUTES)
    args = parser.parse_args()
    if args.threads < 1 or args.segment_minutes <= 0:
        parser.error("threads and segment-minutes must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected Python environment")
    torch.set_num_threads(args.threads)
    print(json.dumps({
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cpu_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
    }, indent=2), flush=True)
    modes = ("warm", "scratch") if args.run == "all" else (args.run,)
    for mode in modes:
        run_one(mode, args.threads, args.segment_minutes)


if __name__ == "__main__":
    main()
