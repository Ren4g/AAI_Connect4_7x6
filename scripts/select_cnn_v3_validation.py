"""Select the Phase C warm-start CNN using Phase B validation only."""
from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_key(manifest: dict) -> tuple[str, float, str]:
    cfg = manifest["configuration"]
    if "_ab3" in manifest["run_id"]:
        family = "ab3_teacher"
    elif cfg["policy_target"] == "hard_best":
        family = "hard_policy"
    else:
        family = "pons_soft"
    return family, float(cfg["data_fraction"]), cfg["dataset_id"]


def display_config(key: tuple[str, float, str]) -> str:
    family, fraction, _ = key
    if family == "ab3_teacher":
        return "AB3 teacher · 100%"
    if family == "hard_policy":
        return "Pons hard · 100%"
    return f"Pons soft · {fraction:.0%}"


def main() -> None:
    run_dirs = sorted((ROOT / "results/colab").glob("cnn_v3_30k*/"))
    manifests = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") != "cnn_supervised_solver_labels":
            continue
        if manifest.get("status") != "complete" or manifest.get("test_overlap_count") != 0:
            raise ValueError(f"run is incomplete or reports test overlap: {manifest['run_id']}")
        if manifest.get("best_validation_metrics", {}).get("positions") != 4530:
            raise ValueError(f"unexpected validation size: {manifest['run_id']}")
        checkpoint = run_dir / "checkpoint_best.pt"
        if not checkpoint.is_file() or sha256(checkpoint) != manifest["files_sha256"]["checkpoint_best.pt"]:
            raise ValueError(f"best checkpoint checksum mismatch: {manifest['run_id']}")
        manifests.append(manifest)

    groups: dict[tuple[str, float, str], list[dict]] = {}
    for manifest in manifests:
        groups.setdefault(config_key(manifest), []).append(manifest)
    if len(manifests) != 15 or len(groups) != 5 or any(len(group) != 3 for group in groups.values()):
        raise ValueError(f"expected 15 runs in five three-seed groups; got {len(manifests)} runs and group sizes {[len(g) for g in groups.values()]}")

    summaries = []
    for key, group in groups.items():
        group = sorted(group, key=lambda row: row["seed"])
        top1 = [row["best_validation_metrics"]["exact_optimal_top1"] for row in group]
        losses = [row["best_validation_loss"] for row in group]
        summaries.append({
            "configuration": display_config(key), "family": key[0],
            "data_fraction": key[1], "dataset_id": key[2],
            "seeds": [row["seed"] for row in group],
            "validation_positions_per_seed": [row["best_validation_metrics"]["positions"] for row in group],
            "exact_optimal_top1_by_seed": top1,
            "exact_optimal_top1_mean": statistics.mean(top1),
            "exact_optimal_top1_sd": statistics.stdev(top1),
            "best_validation_loss_by_seed": losses,
            "mean_validation_loss": statistics.mean(losses),
        })
    # Registered primary metric for comparing data/label configurations is exact-optimal.
    # A single run's checkpoint remains the one selected by that run's validation loss.
    winner_summary = max(summaries, key=lambda row: (row["exact_optimal_top1_mean"], -row["mean_validation_loss"]))
    winner_key = next(key for key in groups if display_config(key) == winner_summary["configuration"])
    winner_group = groups[winner_key]
    selected = min(winner_group, key=lambda row: (row["best_validation_loss"], -row["best_validation_metrics"]["exact_optimal_top1"]))

    checkpoint_source = ROOT / "results/colab" / selected["run_id"] / "checkpoint_best.pt"
    checkpoint_target = ROOT / "checkpoints/cnn_v3_30k_best.pt"
    sidecar = ROOT / "checkpoints/cnn_v3_30k_best.manifest.json"
    if checkpoint_target.exists() or sidecar.exists():
        raise FileExistsError("refusing to replace an existing Phase C initialization checkpoint or its sidecar")
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_source, checkpoint_target)
    checkpoint_hash = sha256(checkpoint_target)

    output = {
        "created_local": datetime.now().astimezone().isoformat(),
        "selection_data": "validation only; frozen solver test not read",
        "selection_rule": "choose configuration with highest mean validation exact_optimal_top1 over its three seeds; each checkpoint is selected by that run's minimum validation loss",
        "configuration_summaries": sorted(summaries, key=lambda row: row["exact_optimal_top1_mean"], reverse=True),
        "selected_configuration": winner_summary["configuration"],
        "selected_run_id": selected["run_id"],
        "selected_seed": selected["seed"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_loss": selected["best_validation_loss"],
        "selected_validation_exact_optimal_top1": selected["best_validation_metrics"]["exact_optimal_top1"],
        "selected_validation_positions": selected["best_validation_metrics"]["positions"],
        "checkpoint_path": str(checkpoint_target.relative_to(ROOT).as_posix()),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest_sha256": selected["dataset_manifest_sha256"],
        "source_code_sha256": selected["source_code_sha256"],
        "source_code_commit": selected.get("source_code_commit"),
        "frozen_test_sha256": selected["frozen_test_sha256"],
        "run_manifest_path": f"results/colab/{selected['run_id']}/manifest.json",
    }
    output_path = ROOT / "results/cnn_v3_validation_selection_20261122.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    sidecar.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# CNN v3 validation selection (2026-11-22)", "",
        "Configuration comparison uses validation exact-optimal top-1, averaged over the three registered seeds. Checkpoint within a run is the validation-loss-selected `checkpoint_best.pt`. The frozen solver test was not used in this selection.", "",
        "| Cấu hình | Seed 1/2/3 exact-optimal | Mean ± SD | Mean validation loss |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda entry: entry["exact_optimal_top1_mean"], reverse=True):
        vals = " / ".join(f"{100*x:.2f}%" for x in row["exact_optimal_top1_by_seed"])
        lines.append(f"| {row['configuration']} | {vals} | {100*row['exact_optimal_top1_mean']:.2f}% ± {100*row['exact_optimal_top1_sd']:.2f} pp | {row['mean_validation_loss']:.4f} |")
    lines += [
        "", f"Selected configuration: **{winner_summary['configuration']}**.",
        f"Warm-start checkpoint: `{selected['run_id']}/checkpoint_best.pt` (seed {selected['seed']}, epoch {selected['best_epoch']}); validation exact-optimal {100*selected['best_validation_metrics']['exact_optimal_top1']:.2f}% on {selected['best_validation_metrics']['positions']} positions, validation loss {selected['best_validation_loss']:.4f}.",
        f"Checkpoint copied to `checkpoints/cnn_v3_30k_best.pt`; SHA256 `{checkpoint_hash}`.",
        "The three seeds support a descriptive validation comparison only; do not treat their SD as a cluster-bootstrap CI.",
    ]
    report_path = ROOT / "docs/CNN_V3_SELECTION.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_path = ROOT / "docs/LOG.md"
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n### 2026-11-22 — Ingest CNN v3 và khóa checkpoint Phase C theo validation\n\n")
        log.write("- 15 archive CNN v3 được kiểm tra và ingest bằng source bundle v11 đúng source-tree SHA; artifact checksum, dataset checksum, inference CPU và leakage test local đều đạt. Hash của Colab run không khớp working tree mới vì mã nguồn đã thay đổi sau bundle; không bỏ qua kiểm tra, thay vào đó chạy ingester từ snapshot v11.\n")
        log.write(f"- So sánh cấu hình bằng validation exact-optimal top-1, ba seed/cấu hình: chọn {winner_summary['configuration']} ({100*winner_summary['exact_optimal_top1_mean']:.2f}% ± {100*winner_summary['exact_optimal_top1_sd']:.2f} điểm phần trăm).\n")
        log.write(f"- Checkpoint warm-start chọn theo validation loss trong cấu hình đã khóa: {selected['run_id']}, epoch {selected['best_epoch']}, loss {selected['best_validation_loss']:.4f}, SHA256 `{checkpoint_hash}`. Frozen test chưa được dùng để chọn.\n")

    print(json.dumps(output, indent=2))
    print(f"Wrote {report_path.relative_to(ROOT)} and {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
