from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
               for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ingest(source: Path, dataset: Path, destination_root: Path,
           log_path: Path = Path("docs/LOG.md")) -> Path:
    manifest_path = source / "manifest.json"
    required = {
        "schema_version", "run_id", "phase", "dataset_manifest_sha256",
        "frozen_test_sha256", "test_overlap_count", "configuration", "seed",
        "source_code_sha256", "python", "torch", "cuda", "gpu_name", "cpu_count",
        "ram_bytes", "start_time", "end_time", "solver_commit", "status",
        "completed_steps", "files_sha256",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["status"] != "complete" or manifest["phase"] not in {
        "cnn_supervised_solver_labels", "alphazero_selfplay", "rl_policy_gradient"
    }:
        raise ValueError("only completed CNN, RL or AlphaZero self-play runs can be ingested")
    data_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if sha256(dataset / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest checksum differs from run manifest")
    for name, expected in data_manifest["files_sha256"].items():
        path = dataset / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"input dataset checksum mismatch: {name}")
    if manifest["frozen_test_sha256"] != data_manifest["frozen_test_sha256"]:
        raise ValueError("frozen test dataset differs")
    if manifest["test_overlap_count"] != 0:
        raise ValueError("run reports nonzero frozen test overlap")
    if manifest["source_code_sha256"] != source_tree_sha256(Path.cwd()):
        raise ValueError("source code tree differs from the Colab run; restore its exact source before ingest")
    config = manifest["configuration"]
    if config.get("smoke", False):
        raise ValueError("smoke artifacts are pipeline checks, not eligible checkpoints")
    if config.get("dataset_manifest_sha256") != manifest["dataset_manifest_sha256"]:
        raise ValueError("configuration dataset hash mismatch")
    if config.get("dataset_id") != data_manifest["run_id"]:
        raise ValueError("configuration dataset id mismatch")
    for name, expected in manifest["files_sha256"].items():
        file_path = source / name
        if not file_path.is_file() or sha256(file_path) != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    latest = source / "checkpoint_latest.pt"
    candidate = source / ("checkpoint_best.pt" if manifest["phase"] == "cnn_supervised_solver_labels"
                          else "checkpoint_policy.pt")
    if not latest.is_file() or not candidate.is_file():
        raise ValueError("latest and candidate policy checkpoints are required")

    import torch
    from connect4.model import ConnectFourCNN

    checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
    model = ConnectFourCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    arrays = np.load(dataset / "train_validation.npz")
    sample = torch.as_tensor(arrays["features"][:min(2, len(arrays["features"]))], dtype=torch.float32)
    with torch.inference_mode():
        policy, value = model(sample)
    if tuple(policy.shape) != (len(sample), 7) or tuple(value.shape) != (len(sample),):
        raise ValueError("checkpoint inference returned invalid output shapes")
    if not torch.isfinite(policy).all() or not torch.isfinite(value).all():
        raise ValueError("checkpoint inference returned non-finite values")

    frozen = json.loads((dataset / "frozen_test_hashes.json").read_text(encoding="utf-8"))
    test_hashes = set(frozen["state_sha256_mirror_canonical"])
    if manifest["phase"] == "cnn_supervised_solver_labels":
        records = json.loads((dataset / "hashes.json").read_text(encoding="utf-8"))["records"]
        trained_keys = {row["state_sha256_mirror_canonical"] for row in records}
    elif manifest["phase"] == "alphazero_selfplay":
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        trained_keys = set(resume["training_state_hashes"])
        trained_keys.update(row["state_sha256_mirror_canonical"] for row in resume["replay"])
        trained_keys.update(row["state_sha256_mirror_canonical"]
                            for game in resume["active"] for row in game["rows"])
    else:
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        trained_keys = set(resume["training_state_hashes"])
    overlap = test_hashes & trained_keys
    if overlap:
        raise ValueError(f"local frozen test leakage check failed ({len(overlap)} states)")

    run_id = manifest["run_id"]
    destination = destination_root / run_id
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    shutil.copytree(source, destination)
    entry = {
        "date": manifest.get("end_time", manifest["start_time"])[:10],
        "event": "colab_run_ingested", "run_id": run_id,
        "artifact": str(destination), "seed": manifest["seed"],
        "steps": manifest["completed_steps"],
        "frozen_test_sha256": manifest["frozen_test_sha256"],
        "local_leakage_check": "passed", "inference_check": "passed",
    }
    log = log_path
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        output.write(f"\n### {entry['date']} — Colab artifact ingested\n\n")
        output.write("- " + json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps({"ingested": str(destination), **entry}, ensure_ascii=False, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and ingest a returned Colab training run")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=Path("results/colab"))
    parser.add_argument("--log-path", type=Path, default=Path("docs/LOG.md"))
    args = parser.parse_args()
    ingest(args.artifact, args.dataset, args.destination_root, args.log_path)


if __name__ == "__main__":
    main()
