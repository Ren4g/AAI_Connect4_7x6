"""Verify and ingest returned Colab ZIP artifacts against their exact source bundle.

This avoids comparing a historical run with today's modified checkout while
preserving strict source-tree, file-integrity, inference, and leakage checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_names(archive: zipfile.ZipFile) -> list[str]:
    names = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"symbolic links are not accepted in ZIPs: {info.filename!r}")
        if info.filename:
            names.append(info.filename)
    return names


def source_hash_in_zip(archive: zipfile.ZipFile) -> str:
    digest = hashlib.sha256()
    files = []
    for name in safe_names(archive):
        path = PurePosixPath(name)
        if path.suffix != ".py" or any(part in {
            ".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"
        } for part in path.parts):
            continue
        files.append(name)
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(archive.read(name))
    return digest.hexdigest()


def extract_checked(archive_path: Path, destination: Path,
                    required_root: str | None = None) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"corrupt member {bad_member!r} in {archive_path}")
        names = safe_names(archive)
        for name in names:
            info = archive.getinfo(name)
            path = PurePosixPath(name)
            if required_root and path.parts and path.parts[0] != required_root:
                raise ValueError(f"unexpected top-level path {name!r}; expected {required_root!r}")
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            extracted.append(target)
    return extracted


def preflight_runs(archives: list[Path], datasets: Path,
                   source_tree_hash: str) -> list[tuple[Path, dict, Path]]:
    dataset_by_id = {}
    for manifest_path in datasets.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_by_id[manifest["run_id"]] = (manifest_path.parent, manifest)
        except (KeyError, json.JSONDecodeError):
            continue

    checked = []
    seen = set()
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            names = safe_names(archive)
            candidates = [n for n in names if PurePosixPath(n).name == "manifest.json"]
            if len(candidates) != 1:
                raise ValueError(f"{archive_path.name}: expected exactly one manifest.json")
            run_manifest = json.loads(archive.read(candidates[0]))
            run_id = run_manifest["run_id"]
            if run_id in seen:
                raise ValueError(f"duplicate run_id in input archives: {run_id}")
            seen.add(run_id)
            if run_manifest.get("status") != "complete" or run_manifest.get("phase") != "cnn_supervised_solver_labels":
                raise ValueError(f"{run_id}: not a completed supervised CNN run")
            if run_manifest.get("source_code_sha256") != source_tree_hash:
                raise ValueError(f"{run_id}: source tree hash differs from supplied source bundle")
            dataset_id = run_manifest.get("configuration", {}).get("dataset_id")
            if dataset_id not in dataset_by_id:
                raise ValueError(f"{run_id}: missing local dataset package {dataset_id!r}")
            dataset_path, dataset_manifest = dataset_by_id[dataset_id]
            if sha256(dataset_path / "manifest.json") != run_manifest["dataset_manifest_sha256"]:
                raise ValueError(f"{run_id}: dataset manifest checksum mismatch")
            for name, expected in dataset_manifest.get("files_sha256", {}).items():
                file_path = dataset_path / name
                if not file_path.is_file() or sha256(file_path) != expected:
                    raise ValueError(f"{run_id}: dataset file checksum mismatch: {name}")
            if dataset_manifest.get("frozen_test_sha256") != run_manifest.get("frozen_test_sha256"):
                raise ValueError(f"{run_id}: frozen test SHA differs")
            if run_manifest.get("test_overlap_count") != 0:
                raise ValueError(f"{run_id}: run declares frozen-test overlap")
            for name, expected in run_manifest["files_sha256"].items():
                member = f"{run_id}/{name}"
                if member not in names or hashlib.sha256(archive.read(member)).hexdigest() != expected:
                    raise ValueError(f"{run_id}: missing or corrupt run artifact {name}")
            for name in ("checkpoint_latest.pt", "checkpoint_best.pt"):
                if f"{run_id}/{name}" not in names:
                    raise ValueError(f"{run_id}: required checkpoint missing: {name}")
            checked.append((archive_path, run_manifest, dataset_path))
    return checked


def ingest_all(archives: list[Path], source_bundle: Path,
               destination_root: Path, dataset_root: Path,
               log_path: Path) -> list[str]:
    bundle_manifest_path = source_bundle.with_suffix(".manifest.json")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if sha256(source_bundle) != bundle_manifest.get("zip_sha256"):
        raise ValueError("source bundle ZIP checksum differs from its manifest")

    with zipfile.ZipFile(source_bundle) as bundle:
        tree_hash = source_hash_in_zip(bundle)
        if tree_hash != bundle_manifest.get("source_tree_sha256"):
            raise ValueError("Python source tree inside bundle differs from its manifest")
        # Fully validate every run before the first artifact is copied or logged.
        checked = preflight_runs(archives, dataset_root, tree_hash)

    ingested = []
    with tempfile.TemporaryDirectory(prefix="connect4_ingest_") as temporary:
        temp_root = Path(temporary)
        source_root = temp_root / "source"
        extract_checked(source_bundle, source_root)
        from_bundle = subprocess.run(
            [sys.executable, str(source_root / "scripts" / "ingest_colab_run.py"),
             "--help"], cwd=source_root, capture_output=True, text=True,
        )
        if from_bundle.returncode:
            raise RuntimeError("could not run ingester from source bundle: " + from_bundle.stderr)

        for index, (archive_path, manifest, dataset_path) in enumerate(checked):
            run_id = manifest["run_id"]
            destination = destination_root / run_id
            if destination.exists():
                existing = destination / "manifest.json"
                if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != manifest:
                    raise FileExistsError(f"refusing to overwrite different artifact at {destination}")
                ingested.append(f"already-ingested:{run_id}")
                continue

            artifact_root = temp_root / f"artifact_{index}"
            extract_checked(archive_path, artifact_root, required_root=run_id)
            artifact = artifact_root / run_id
            result = subprocess.run(
                [sys.executable, str(source_root / "scripts" / "ingest_colab_run.py"),
                 str(artifact), "--dataset", str(dataset_path),
                 "--destination-root", str(destination_root), "--log-path", str(log_path)],
                cwd=source_root, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"ingest failed for {run_id}:\n{result.stdout}\n{result.stderr}")
            print(result.stdout.strip())
            ingested.append(run_id)
    return ingested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Directory containing returned run ZIP files")
    parser.add_argument("--dataset-root", type=Path, default=Path("colab/data"))
    parser.add_argument("--destination-root", type=Path, default=Path("results/colab"))
    parser.add_argument("--log-path", type=Path, default=Path("docs/LOG.md"))
    parser.add_argument("--pattern", default="cnn_v3_30k*.zip")
    args = parser.parse_args()
    archives = sorted(p.resolve() for p in args.artifact_dir.glob(args.pattern))
    if not archives:
        raise SystemExit(f"no archives matched {args.pattern!r} in {args.artifact_dir}")
    result = ingest_all(archives, args.source_bundle.resolve(),
                        args.destination_root.resolve(), args.dataset_root.resolve(),
                        args.log_path.resolve())
    print(json.dumps({"archive_count": len(archives), "ingested": result}, indent=2))


if __name__ == "__main__":
    main()
