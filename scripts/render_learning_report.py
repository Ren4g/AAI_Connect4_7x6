"""Generate a reproducible summary of clean Colab checkpoints and solver evaluations."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci(interval: list[float]) -> str:
    return f"[{100 * interval[0]:.1f}; {100 * interval[1]:.1f}]"


def decimal_ci(interval: list[float]) -> str:
    return f"[{interval[0]:.3f}; {interval[1]:.3f}]"


def metric_line(name: str, metrics: dict) -> str:
    exact = metrics["all"]["optimal"]
    wdl = metrics["all"]["wdl_preserving"]
    return (f"| {name} | {exact['successes']}/{metrics['all']['positions']} | "
            f"{percent(exact['rate'])} {ci(exact['cluster_bootstrap_95'])} | "
            f"{percent(wdl['rate'])} {ci(wdl['cluster_bootstrap_95'])} |")


def render(args: argparse.Namespace) -> str:
    cnn = read_json(args.cnn_eval)
    learning = read_json(args.learning_eval)
    value = read_json(args.value_eval)
    arena = read_json(args.arena_eval)
    cnn_runs = [read_json(path) for path in args.cnn_manifests]
    az_run, rl_run = read_json(args.az_manifest), read_json(args.rl_manifest)
    cnn_validation = []
    for manifest_path, run in zip(args.cnn_manifests, cnn_runs):
        log_path = manifest_path.parent / "train_log.jsonl"
        epochs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        best = min(epochs, key=lambda row: row["validation"]["loss"])
        cnn_validation.append((best["validation"]["loss"], run["seed"], run["run_id"],
                               best["validation"]["exact_optimal_top1"]))
    selected_cnn = min(cnn_validation)
    value_cnn = next((run for run in cnn_runs
                      if run["files_sha256"].get("checkpoint_best.pt") == value["checkpoint_sha256"]), None)
    value_seed = value_cnn["seed"] if value_cnn else "unknown"

    lines = [
        "# Kết quả huấn luyện và đánh giá CNN, RL, AlphaZero",
        "",
        "> Bản này được sinh từ các manifest và JSON kết quả; chạy lại `python scripts/render_learning_report.py` để tái tạo.",
        "",
        "## Thiết kế đánh giá",
        "",
        (f"Các phép đo dùng cùng tập test Pons đóng băng gồm {learning['positions']:,} vị trí, "
         f"SHA256 `{learning['frozen_positions_sha256']}`, cùng solver `{learning['solver_commit']}`. "
         f"Khoảng tin cậy 95% bootstrap theo ván nguồn, {learning['bootstrap']['replicates']:,} lần lặp; "
         "chênh lệch với AlphaBeta depth 3 ghép cặp trên cùng vị trí/ván nguồn. "
         "Thời gian suy luận đo local và không dùng để so sánh tài nguyên Colab."),
        "",
        "## Run Colab đã ingest",
        "",
        "| Nhánh | Run | Bước/iteration | Resume | GPU | Cấu hình / ghi chú |",
        "|---|---|---:|---:|---|---|",
    ]
    for run in cnn_runs:
        cfg = run["configuration"]
        lines.append(
            f"| CNN supervised | `{run['run_id']}` | {run['completed_steps']} bước | "
            f"không | {run['gpu_name']} | {cfg.get('epochs')} epoch; seed {run['seed']} |"
        )
    lines.append(
        f"| AlphaZero self-play | `{az_run['run_id']}` | {az_run.get('iteration')} iteration / "
        f"{az_run['completed_steps']:,} update, {az_run.get('games_completed', 'n/a')} game | "
        f"có ({az_run.get('resume_count', 0)} lần resume) | "
        f"{az_run['gpu_name']} | {az_run['configuration']['games_per_iteration']} games/iteration, "
        f"{az_run['configuration']['simulations']} simulations/nước; khởi tạo từ CNN seed 20261121 |"
    )
    lines.append(
        f"| Policy gradient RL | `{rl_run['run_id']}` | {rl_run['completed_steps']:,} episode | "
        f"không | {rl_run['gpu_name']} | policy-gradient đấu Random; seed {rl_run['seed']} |"
    )

    lines += [
        "",
        "## Năng lực chọn nước theo solver",
        "",
        "Nước tối ưu là chọn một nước có điểm exact cao nhất theo solver; giữ W/D/L là chọn nước cùng dấu với kết quả tốt nhất. Các CI trong bảng là phần trăm.",
        "",
        "| Agent | Đúng tối ưu / n | Tỷ lệ nước tối ưu, CI 95% | Giữ W/D/L, CI 95% |",
        "|---|---:|---:|---:|",
    ]
    for name in ("Random", "Heuristic", "AlphaBeta-depth3", "AlphaBeta-depth4",
                 "AlphaBeta-depth5", "AlphaBeta-depth6", "MCTS-500"):
        lines.append(metric_line(name, learning["metrics"][name]))
    for name in cnn["agents"]:
        if name.endswith("-policy"):
            lines.append(metric_line(name, cnn["metrics"][name]))
    lines.append(metric_line("CNN-RL-20261120-policy", learning["metrics"]["CNN-RL-20261120-policy"]))
    lines.append(metric_line("PUCT-AZ-20261120 (500 simulations)", learning["metrics"]["PUCT-AZ-20261120"]))

    cnn_names = [name for name in cnn["agents"] if name.endswith("-policy")]
    cnn_rates = [cnn["metrics"][name]["all"]["optimal"]["rate"] for name in cnn_names]
    lines += [
        "",
        (f"Ba seed CNN đạt trung bình mô tả {percent(statistics.mean(cnn_rates))} "
         f"(độ lệch chuẩn giữa seed {100 * statistics.stdev(cnn_rates):.1f} điểm phần trăm); "
         "đây là độ biến thiên giữa ba seed, không phải CI bootstrap gộp."),
        (f"Checkpoint CNN dùng trong round-robin là `{selected_cnn[2]}` (seed {selected_cnn[1]}), "
         f"được chọn theo validation loss thấp nhất giữa các checkpoint-best; "
         f"exact-optimal top-1 ở epoch được lưu là {percent(selected_cnn[3])}."),
        "",
        "### Theo giai đoạn",
        "",
        "| Agent | Khai cuộc (n=500) | Trung cuộc (n=500) | Tàn cuộc (n=500) |",
        "|---|---:|---:|---:|",
    ]
    focus = cnn_names + ["CNN-RL-20261120-policy", "PUCT-AZ-20261120"]
    for name in focus:
        source = cnn if name in cnn["metrics"] else learning
        parts = []
        for stage in ("opening", "middle", "endgame"):
            stat = source["metrics"][name][f"stage_{stage}"]["optimal"]
            parts.append(f"{percent(stat['rate'])} {ci(stat['cluster_bootstrap_95'])}")
        lines.append(f"| {name} | " + " | ".join(parts) + " |")

    lines += [
        "",
        "## Cổng CNN và so sánh ghép cặp",
        "",
        "Tiêu chí tối thiểu đặt trước yêu cầu cận dưới CI 95% của hiệu `CNN − AlphaBeta depth 3` không âm trên tổng thể. Giá trị dưới đây là điểm phần trăm; cổng chỉ đạt nếu khoảng không nằm dưới 0.",
        "",
        "| Checkpoint | Hiệu tỷ lệ nước tối ưu, CI ghép cặp 95% | Kết luận cổng |",
        "|---|---:|---|",
    ]
    gate_results = []
    for name in cnn_names:
        interval = cnn["paired_difference_vs_ab3"][name]["optimal"]
        passed = interval[0] >= 0
        gate_results.append(passed)
        lines.append(f"| {name} | {ci(interval)} điểm % | {'đạt' if passed else 'chưa đạt'} |")
    for name, key in (("RL-Policy (safety-filtered)", "CNN-RL-20261120-policy"),
                      ("PUCT-AZ-20261120, 500 simulations", "PUCT-AZ-20261120")):
        interval = learning["paired_difference_vs_ab3"][key]["optimal"]
        lines.append(f"| {name} | {ci(interval)} điểm % | agent tham chiếu, chưa qua cổng CNN |")

    overall = value["metrics"]["all"]
    lines += [
        "",
        "## Arena chọn checkpoint self-play",
        "",
        "| Candidate | Incumbent | W-D-L candidate | Score | CI bootstrap 95% | Opening | Kết luận |",
        "|---|---|---:|---:|---:|---:|---|",
        (f"| {arena['candidate']['label']} | {arena['incumbent']['label']} | "
         f"{arena['summary']['wins']}-{arena['summary']['draws']}-{arena['summary']['losses']} | "
         f"{arena['summary']['score']:.3f} | {ci(arena['summary']['score_bootstrap_95'])} | "
         f"{arena['summary']['unique_openings']} | "
         f"{'được promote' if arena['promoted'] else 'không promote'} |"),
        "",
        "## Value head của CNN checkpoint dùng trong round-robin",
        "",
        (f"Checkpoint seed {value_seed}: MSE {overall['value_mse']:.3f} "
         f"(CI 95% {decimal_ci(overall['cluster_bootstrap_95']['value_mse_95'])}); W/D/L accuracy "
         f"{percent(overall['wdl_accuracy'])} (CI {ci(overall['cluster_bootstrap_95']['wdl_accuracy_95'])}) "
         f"so với baseline lớp đa số {percent(overall['majority_baseline_accuracy'])}. "
         "Ở tàn cuộc, accuracy là "
         f"{percent(value['metrics']['endgame']['wdl_accuracy'])} "
         f"(CI {ci(value['metrics']['endgame']['cluster_bootstrap_95']['wdl_accuracy_95'])})."),
        "",
        "## Diễn giải và giới hạn",
        "",
        "- Ba CNN supervised đều thấp hơn AlphaBeta depth 3 theo hiệu ghép cặp; cổng tối thiểu chưa đạt. Mục tiêu 85% nước tối ưu cũng chưa đạt.",
        f"- AZ self-play đã hoàn tất {az_run.get('iteration')} iteration và {az_run.get('games_completed', 'n/a')} ván tự đấu, "
        f"nhưng PUCT với {learning['metrics']['PUCT-AZ-20261120']['local_cost']['mean_simulations_per_move']:.0f} mô phỏng/nước trên test đạt "
        f"{percent(learning['metrics']['PUCT-AZ-20261120']['all']['optimal']['rate'])} nước tối ưu; "
        "khoảng hiệu ghép cặp với AB3 vẫn âm. Đây là đánh giá một checkpoint/seed, không đủ kết luận xu hướng huấn luyện.",
        f"- RL chạy đủ {rl_run['completed_steps']:,} episode; solver đánh giá policy sau safety filter thấp hơn baseline tìm kiếm. Kết quả huấn luyện ghi nhận hiệu suất trong lúc học so với Random, không thay cho round-robin độc lập.",
        ("- Value head CNN có W/D/L accuracy điểm ước lượng trên baseline lớp đa số, nhưng CI 95% chứa baseline; ở tàn cuộc điểm thấp hơn baseline. Chưa có bằng chứng chắc rằng nó dùng độc lập thay rollout được."),
        (f"- Arena AZ iteration {arena['summary']['games']} ván / {arena['summary']['unique_openings']} opening so với PUCT khởi tạo đạt score {arena['summary']['score']:.3f} "
         f"(CI bootstrap 95% {ci(arena['summary']['score_bootstrap_95'])} điểm %); cận dưới không vượt 50%, nên checkpoint không được promote."),
        "- Test có các vị trí lấy theo ván nguồn; CI bootstrap theo cụm xử lý phụ thuộc trong cùng ván. Kết quả chỉ áp dụng cho tập test và cấu hình đã ghi, không khái quát tự động sang mọi khai cuộc.",
        "",
        "## Nguồn dữ liệu",
        "",
        f"- Solver evaluation CNN: `{args.cnn_eval.as_posix()}`.",
        f"- Solver evaluation AZ/RL và đối chứng: `{args.learning_eval.as_posix()}`.",
        f"- Value head: `{args.value_eval.as_posix()}`.",
        f"- Checkpoint arena: `{args.arena_eval.as_posix()}`.",
        f"- Đường cong depth/chất lượng: `{args.depth_svg.as_posix()}`.",
        f"- Đường cong huấn luyện chẩn đoán (không phải strength test): `{args.training_svg.as_posix()}`.",
        "- Checkpoint chỉ được dùng sau ingest; manifest, checksum và kiểm tra test leakage nằm trong `results/colab/`.",
        "",
    ]
    if not any(gate_results):
        lines.insert(lines.index("## Diễn giải và giới hạn"),
                     "**Kết luận cổng:** chưa có checkpoint CNN vượt cổng tối thiểu.\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-eval", type=Path, default=Path("results/solver_eval_cnn_v2_3seeds.json"))
    parser.add_argument("--learning-eval", type=Path, default=Path("results/solver_eval_az_rl_20261121_seed20261121.json"))
    parser.add_argument("--value-eval", type=Path, default=Path("results/value_head/value_cnn_seed20261121_20261121.json"))
    parser.add_argument("--arena-eval", type=Path, default=Path("benchmarks/arena/arena_az_vs_init_100openings_seed20261121.json"))
    parser.add_argument("--cnn-manifests", nargs="*", type=Path, default=[
        Path("results/colab/cnn_seed20261120_colab1/manifest.json"),
        Path("results/colab/cnn_seed20261121_colab1/manifest.json"),
        Path("results/colab/cnn_seed20261122_colab1/manifest.json"),
    ])
    parser.add_argument("--az-manifest", type=Path, default=Path("results/colab/az_seed20261120_run1/manifest.json"))
    parser.add_argument("--rl-manifest", type=Path, default=Path("results/colab/rl_seed20261120_run1/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/LEARNING_RESULTS.md"))
    parser.add_argument("--depth-svg", type=Path, default=Path("docs/solver_depth_curve_learning.svg"))
    parser.add_argument("--training-svg", type=Path, default=Path("docs/training_process_trace.svg"))
    args = parser.parse_args()
    if args.output.exists() or args.depth_svg.exists() or args.training_svg.exists():
        raise FileExistsError("Refusing to overwrite a generated report or figure; choose new output paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.depth_svg.parent.mkdir(parents=True, exist_ok=True)
    args.training_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args), encoding="utf-8")
    from render_solver_report import render_svg
    args.depth_svg.write_text(render_svg(read_json(args.learning_eval)), encoding="utf-8")
    from render_training_traces import read_jsonl, render as render_training
    rl_log = read_jsonl(args.rl_manifest.parent / "train_log.jsonl")
    az_log = read_jsonl(args.az_manifest.parent / "train_log.jsonl")
    args.training_svg.write_text(render_training(rl_log, az_log), encoding="utf-8")
    print(f"Wrote {args.output}, {args.depth_svg} and {args.training_svg}")


if __name__ == "__main__":
    main()
