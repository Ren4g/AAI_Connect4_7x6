"""Render selected CNN v3 validation/test results from their JSON sources."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "results/solver_eval_cnn_v3_soft_3seeds_20261122.json"
VALIDATION_FILE = ROOT / "results/cnn_v3_validation_selection_20261122.json"
BASELINE_FILE = ROOT / "results/solver_eval_cnn_v2_3seeds.json"
OUT = ROOT / "docs/CNN_V3_RESULTS.md"


def percent(value: float) -> str:
    return f"{100*value:.1f}%"


def ci(interval: list[float]) -> str:
    return f"[{100*interval[0]:.1f}; {100*interval[1]:.1f}]%"


def table(headers: list[str], rows: list[tuple]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def main() -> None:
    result = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    selection = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    if result["partial_diagnostic_only"] or result["positions"] != 1500:
        raise ValueError("expected the full, non-smoke frozen-test evaluation")
    if result["frozen_positions_sha256"] != selection["frozen_test_sha256"]:
        raise ValueError("frozen test hash differs from the validation-selected checkpoint provenance")
    if result["solver_commit"] != baseline["solver_commit"] or result["frozen_positions_sha256"] != baseline["frozen_positions_sha256"]:
        raise ValueError("new and baseline evaluations do not share the frozen test and solver revision")

    metrics = result["metrics"]
    seed_ids = (20261121, 20261122, 20261123)
    keys = [f"CNN-CNNv3_s{seed}-policy" for seed in seed_ids]
    raw_keys = [f"CNN-CNNv3_s{seed}-raw" for seed in seed_ids]
    test_rows = []
    for seed, raw_key, filtered_key in zip(seed_ids, raw_keys, keys):
        for mode, key in (("Mạng thuần, không safety filter", raw_key), ("Có safety filter", filtered_key)):
            item = metrics[key]["all"]
            paired = result["paired_difference_vs_ab3"][key]
            test_rows.append((str(seed), mode, item["positions"], percent(item["optimal"]["rate"]), ci(item["optimal"]["cluster_bootstrap_95"]), percent(item["wdl_preserving"]["rate"]), ci(item["wdl_preserving"]["cluster_bootstrap_95"]), ci(paired["optimal"])))

    aggregate_rows = []
    old_metrics = baseline["metrics"]
    for mode, names in (("CNN v3 raw", raw_keys), ("CNN v3 filtered", keys)):
        rates = [metrics[name]["all"]["optimal"]["rate"] for name in names]
        wdl = [metrics[name]["all"]["wdl_preserving"]["rate"] for name in names]
        aggregate_rows.append((mode, percent(statistics.mean(rates)), f"{100*statistics.stdev(rates):.2f} pp", percent(statistics.mean(wdl)), f"{100*statistics.stdev(wdl):.2f} pp"))
    # The v2 baseline used three different seeds; keep the within-run cohort
    # rather than incorrectly assuming the v3 seed IDs were reused.
    old_seed_ids = (20261120, 20261121, 20261122)
    old_keys = [f"CNN-CNN_v2_s{seed}-policy" for seed in old_seed_ids]
    old_raw = [f"CNN-CNN_v2_s{seed}-raw" for seed in old_seed_ids]
    for mode, names in (("CNN v2 raw baseline", old_raw), ("CNN v2 filtered baseline", old_keys)):
        rates = [old_metrics[name]["all"]["optimal"]["rate"] for name in names]
        wdl = [old_metrics[name]["all"]["wdl_preserving"]["rate"] for name in names]
        aggregate_rows.append((mode, percent(statistics.mean(rates)), f"{100*statistics.stdev(rates):.2f} pp", percent(statistics.mean(wdl)), f"{100*statistics.stdev(wdl):.2f} pp"))

    ab3 = metrics["AlphaBeta-depth3"]["all"]
    chosen = selection["selected_run_id"]
    text = f'''# CNN v3: tổng hợp validation và frozen-test

## Lựa chọn theo validation

{selection['selected_configuration']} đạt mean exact-optimal top-1 {percent(selection['configuration_summaries'][0]['exact_optimal_top1_mean'])} ± {100*selection['configuration_summaries'][0]['exact_optimal_top1_sd']:.2f} điểm phần trăm trên 3 seed. Checkpoint trong run được chọn bằng validation loss; warm-start là `{chosen}` với validation exact-optimal {percent(selection['selected_validation_exact_optimal_top1'])}, loss {selection['selected_validation_loss']:.4f}. Frozen test không được dùng để chọn.

## Frozen-test

Thử nghiệm dùng {result['positions']} vị trí; SHA256 `{result['frozen_positions_sha256']}`, solver commit `{result['solver_commit']}`, cluster bootstrap {result['bootstrap']['replicates']:,} lần theo source game. Cột “hiệu so với AB3” là CI paired của agent trừ AlphaBeta depth 3 trên cùng các vị trí; cận dưới/ trên chứa 0 thì chưa phân biệt được.

{table(['Seed', 'Chế độ', 'N', 'Exact', 'CI 95% exact', 'Giữ WDL', 'CI 95% WDL', 'Hiệu exact − AB3, CI 95%'], test_rows)}

**AlphaBeta depth 3:** {percent(ab3['optimal']['rate'])} exact, CI {ci(ab3['optimal']['cluster_bootstrap_95'])}; giữ WDL {percent(ab3['wdl_preserving']['rate'])}, CI {ci(ab3['wdl_preserving']['cluster_bootstrap_95'])}.

## Trung bình và độ lệch chuẩn giữa seed

SD ở bảng này là độ lệch chuẩn mẫu giữa ba seed; nó khác với CI bootstrap theo source game ở bảng trên.

{table(['Nhóm', 'Exact mean', 'Exact SD', 'WDL mean', 'WDL SD'], aggregate_rows)}

## Diễn giải

- Kết quả học tập được báo cho cấu hình Pons-soft 100%; các cấu hình hard, AlphaBeta-3 teacher, 10% và 30% vẫn được giữ trong validation artifact nhưng không được chọn làm warm start.
- Filter và mạng thuần được báo riêng. Không quy tác dụng của safety filter cho CNN.
- Kết quả này là đánh giá một lần trên test đóng băng đã dùng cho baseline trong dự án, không phải test độc lập mới. Tuyệt đối không dùng lại để chọn cấu hình hoặc checkpoint.
- CI hiệu CNN−AB3 quyết định việc đạt cổng; giá trị mean đơn lẻ không đủ để kết luận.

## Nguồn

- Frozen-test: `{TEST_FILE.relative_to(ROOT).as_posix()}`
- Validation selection: `{VALIDATION_FILE.relative_to(ROOT).as_posix()}`
- Baseline CNN v2: `{BASELINE_FILE.relative_to(ROOT).as_posix()}`
- Ingested manifests and checkpoints: `results/colab/cnn_v3_30k_*/`
'''
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
