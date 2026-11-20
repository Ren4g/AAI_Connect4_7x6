"""Create a minimal ZIP source snapshot and SHA256 sidecar for Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", ".venv-gpu", "__pycache__", ".pytest_cache"}
INCLUDED_ROOT_FILES = {"requirements-colab.txt", "README.md"}


def tree_hash(root: Path, source_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    manifest_path = output.with_name(output.stem + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("source snapshots are immutable; choose a new output name")
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix == ".py" or relative.as_posix() in INCLUDED_ROOT_FILES:
            files.append(path)
    # Include small Colab notebooks, criteria and setup notes, but never datasets/artifacts.
    for folder in (root / "colab", root / "docs"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if path.is_file() and not any(part in EXCLUDED_PARTS for part in relative.parts):
                if path.suffix == ".ipynb" or (folder.name == "docs" and path.suffix == ".md"):
                    files.append(path)
    files = sorted(set(files))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip() or None
    record = {
        "schema_version": 1, "created_at_local": datetime.now().astimezone().isoformat(),
        "source_commit": commit,
        "source_tree_sha256": tree_hash(root, [path for path in files if path.suffix == ".py"]),
        "zip_sha256": digest, "file_count": len(files),
        "files": [path.relative_to(root).as_posix() for path in files],
    }
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"zip": str(output), "manifest": str(manifest_path), **record}, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path,
                        default=Path("colab/source_bundle_20261120_v2.zip"))
    args = parser.parse_args()
    package(args.root, args.output)


if __name__ == "__main__":
    main()
