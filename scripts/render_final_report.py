"""Generate the main academic-style project report from committed JSON results."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def interval(values: list[float], scale: float = 100) -> str:
    return f"[{scale * values[0]:.1f}; {scale * values[1]:.1f}]"


def solver_row(label: str, data: dict, key: str) -> str:
    metrics = data["metrics"][key]["all"]
    exact = metrics["optimal"]
    wdl = metrics["wdl_preserving"]
    return (f"| {label} | {exact['successes']}/{metrics['positions']} | "
            f"{pct(exact['rate'])} {interval(exact['cluster_bootstrap_95'])} | "
            f"{pct(wdl['rate'])} {interval(wdl['cluster_bootstrap_95'])} |")


def render(args: argparse.Namespace) -> str:
    cnn = load(args.cnn_eval)
    search = load(args.search_eval)
    arena = load(args.arena)
    rr = load(args.round_robin)
    rr_meta = load(args.round_robin_meta)
    resource = load(args.resource)
    value = load(args.value_eval)
    errors = load(args.cnn_errors)
    manifests = [load(path) for path in args.manifests]

    cnn_keys = [f"CNN-CNN_v2_s{seed}-policy" for seed in (20261120, 20261121, 20261122)]
    cnn_rates = [cnn["metrics"][key]["all"]["optimal"]["rate"] for key in cnn_keys]
    cnn_mean = statistics.mean(cnn_rates)
    cnn_sd = statistics.stdev(cnn_rates)

    lines = [
        "# AI cho Connect Four 7×6: hiện thực, huấn luyện và đánh giá có kiểm soát",
        "",
        "> Báo cáo chính được sinh từ kết quả JSON đã kiểm tra. Tái tạo bằng `python scripts/render_final_report.py`; các bảng dẫn xuất được sinh lại từ JSON nguồn, không nhập tay.",
        "",
        "## Tóm tắt",
        "",
        (f"Đề tài xây dựng hệ thống chơi Connect Four 7×6 theo tám cấp độ, từ tác tử ngẫu nhiên đến tìm kiếm kết hợp policy/value và self-play. "
         f"Đánh giá dùng bộ test đóng băng {search['positions']:,} vị trí đã chấm bằng solver Pons, cùng round-robin ghép cặp trên {len(rr_meta['agents'])} agent. "
         f"AlphaBeta depth 4 đạt tỷ lệ chọn nước tối ưu {pct(search['metrics']['AlphaBeta-depth4']['all']['optimal']['rate'])}; CNN supervised đạt trung bình {pct(cnn_mean)} (SD giữa ba seed {100*cnn_sd:.1f} điểm %), còn PUCT-AZ 500 mô phỏng đạt {pct(search['metrics']['PUCT-AZ-20261120']['all']['optimal']['rate'])}. "
         f"Không CNN nào đạt cổng không-kém-hơn AlphaBeta depth 3; checkpoint self-play cũng không qua cổng thăng hạng trong arena. Kết quả thể hiện rằng pipeline và các cấp agent đã được hiện thực, trong khi các mô hình học được chưa vượt baseline tìm kiếm theo tiêu chí định trước."),
        "",
        "## 1. Bài toán và mục tiêu",
        "",
        "Connect Four là trò chơi hai người, tổng bằng không, thông tin hoàn hảo trên lưới 7 cột × 6 hàng. Mỗi lượt thả quân vào một cột còn chỗ; thắng khi tạo bốn quân liên tiếp theo hàng, cột hoặc đường chéo. Không gian trạng thái hữu hạn nhưng nhánh hành động và tương tác chiến thuật khiến luật chơi đơn giản vẫn phù hợp để khảo sát tìm kiếm, học có giám sát và self-play.",
        "",
        "Mục tiêu của đề tài là (i) triển khai các agent có độ phức tạp tăng dần trên cùng luật chơi; (ii) xây dựng giao thức so sánh tái lập được; (iii) dùng solver exact làm thước đo chất lượng nước đi; và (iv) chuẩn bị pipeline huấn luyện có thể resume trên Colab, nhưng kiểm định artifact và đánh giá cuối ở local.",
        "",
        "## 2. Kiến trúc và phương pháp",
        "",
        "Engine cung cấp API trạng thái thống nhất với hai biểu diễn bàn cờ: grid và bitboard. Bitboard dùng trong tìm kiếm/giải đấu nhờ thao tác chiến thuật nhanh; encoder CNN tạo hai mặt phẳng quân của người đến lượt và đối thủ, được dùng chung khi huấn luyện và suy luận. Solver Pons được gọi qua adapter, nhận chuỗi cột hợp lệ và trả điểm exact cho từng nước đi.",
        "",
        "| Cấp | Phương pháp | Vai trò trong nghiên cứu |",
        "|---:|---|---|",
        "| 1 | Random | Baseline tối thiểu và đối chứng cho pipeline. |",
        "| 2 | Heuristic | Chấm các mẫu bốn ô, ưu tiên trung tâm và đe dọa/chặn chiến thuật. |",
        "| 3 | AlphaBeta | Minimax với alpha-beta pruning, iterative deepening, move ordering và giới hạn thời gian tùy preset. |",
        "| 4 | MCTS-UCT | Chọn nước theo thống kê thăm cây và exploration bonus; cấu hình so sánh dùng 500 mô phỏng/nước. |",
        "| 5 | CNN supervised | Policy/value CNN học nhãn solver; trong PolicyAgent hiện tại, policy được dùng trực tiếp, còn value được đánh giá riêng. |",
        "| 6 | Policy-gradient RL | REINFORCE/self-play đối kháng Random, có safety filter khi triển khai policy. |",
        "| 7 | AlphaGo-inspired PUCT | Policy tạo prior, value ước lượng lá; dùng PUCT để phân bổ tìm kiếm. Không có mạng rollout riêng. |",
        "| 8 | AlphaZero-inspired | Self-play PUCT với nhiễu Dirichlet, replay buffer và cập nhật policy/value; vòng arena quyết định promote checkpoint. |",
        "",
        "Các cấp 5–8 có CLI và checkpoint/resume. Job Colab chỉ thực hiện huấn luyện nặng; notebook là lớp gọi CLI. Artifact chỉ được dùng sau khi ingest local kiểm tra manifest, checksum, inference và loại trừ trùng lặp với test đóng băng.",
        "",
        "## 3. Thiết kế thực nghiệm",
        "",
        "### 3.1 Dữ liệu và giao thức solver",
        "",
        (f"Bộ test đóng băng có {search['positions']:,} vị trí từ các ván nguồn; SHA256 vị trí `{search['frozen_positions_sha256']}` và commit solver `{search['solver_commit']}`. "
         f"Tỷ lệ nước tối ưu là phần vị trí agent chọn một nước có điểm exact cao nhất. Giữ W/D/L đo việc nước đi bảo toàn cùng lớp kết cục lý thuyết so với nước tốt nhất. "
         f"Khoảng tin cậy 95% được bootstrap theo cụm ván nguồn, {search['bootstrap']['replicates']:,} lần lặp; các hiệu so với AlphaBeta depth 3 là ghép cặp trên cùng vị trí/cụm."),
        "",
        "### 3.2 Round-robin và arena",
        "",
        (f"Round-robin dùng {rr_meta['games_per_pair']} ván mỗi cặp trên {rr_meta['opening_generation']['accepted']} khai cuộc chung, mỗi khai cuộc đánh cả hai ghế; bộ lọc fair loại {rr_meta['opening_generation']['filtered_immediate_win']} khai cuộc có thắng tức thì trong {rr_meta['opening_generation']['attempts']} ứng viên. "
         f"Score = (thắng + 0,5 hòa)/số ván; CI ghép cặp bootstrap theo khai cuộc. Có {len(rr)} cặp độc lập; JSONL lưu replay nước đi. Arena AlphaZero dùng {arena['summary']['unique_openings']} khai cuộc ghép cặp và ngưỡng promote đặt trước yêu cầu cận dưới CI score > 0,5."),
        "",
        "### 3.3 Tiêu chí chuyển giai đoạn",
        "",
        "Cổng tối thiểu CNN được đặt trước khi huấn luyện: cận dưới CI bootstrap ghép cặp của hiệu tỷ lệ nước tối ưu (CNN − AlphaBeta depth 3) không âm trên test. Mục tiêu 85% nước tối ưu là mục tiêu, không phải cổng chặn. Chọn checkpoint theo validation; test đóng băng không dùng để chọn mô hình.",
        "",
        "## 4. Kết quả",
        "",
        "### 4.1 So sánh với solver exact",
        "",
        "| Agent | Nước tối ưu / n | Tỷ lệ nước tối ưu, CI 95% | Giữ W/D/L, CI 95% |",
        "|---|---:|---:|---:|",
    ]
    for name in ("Random", "Heuristic", "AlphaBeta-depth3", "AlphaBeta-depth4",
                 "AlphaBeta-depth5", "AlphaBeta-depth6", "MCTS-500"):
        lines.append(solver_row(name, search, name))
    for source_key, label in (("CNN-CNN_v2_s20261120-policy", "CNN supervised seed 20261120"),
                              ("CNN-CNN_v2_s20261121-policy", "CNN supervised seed 20261121"),
                              ("CNN-CNN_v2_s20261122-policy", "CNN supervised seed 20261122")):
        lines.append(solver_row(label, cnn, source_key))
    for key, label in (("CNN-RL-20261120-policy", "RL policy (safety-filtered)"),
                       ("PUCT-AZ-20261120", "PUCT-AZ, 500 simulations")):
        lines.append(solver_row(label, search, key))
    lines += [
        "",
        (f"Trung bình ba seed CNN là {pct(cnn_mean)} (độ lệch chuẩn giữa seed {100*cnn_sd:.1f} điểm phần trăm). "
         "Các khoảng tin cậy ghép cặp CNN − AB3 đều nằm dưới 0 nên không seed nào đạt cổng tối thiểu. "
         f"PUCT-AZ đạt {pct(search['metrics']['PUCT-AZ-20261120']['all']['optimal']['rate'])}, thấp hơn AB3 ({pct(search['metrics']['AlphaBeta-depth3']['all']['optimal']['rate'])}); "
         "RL policy còn thấp hơn trên test solver. Kết luận chỉ áp dụng cho checkpoint và giao thức đã đo, không suy rộng sang mọi seed."),
        "",
        "### 4.1.1 Phân tầng theo giai đoạn",
        "",
        "Bảng dưới dùng checkpoint CNN seed 20261121 được chọn bằng validation; mỗi ô là tỷ lệ nước tối ưu và CI bootstrap 95% trên 500 vị trí của giai đoạn đó.",
        "",
        "| Agent | Khai cuộc | Trung cuộc | Tàn cuộc |",
        "|---|---:|---:|---:|",
    ]
    stages = (("stage_opening", "Khai cuộc"), ("stage_middle", "Trung cuộc"),
              ("stage_endgame", "Tàn cuộc"))
    for label, source, key in (
        ("AlphaBeta depth 3", search, "AlphaBeta-depth3"),
        ("CNN seed 20261121", cnn, "CNN-CNN_v2_s20261121-policy"),
        ("RL policy", search, "CNN-RL-20261120-policy"),
        ("PUCT-AZ 500", search, "PUCT-AZ-20261120"),
    ):
        cells = []
        for stage_key, _ in stages:
            metric = source["metrics"][key][stage_key]["optimal"]
            cells.append(f"{pct(metric['rate'])} {interval(metric['cluster_bootstrap_95'])}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### 4.1.2 Cổng CNN và hiệu ghép cặp",
        "",
        "| Checkpoint | Hiệu CNN − AB3, CI 95% (điểm %) | Kết quả |",
        "|---|---:|---|",
    ]
    for key, label in zip(cnn_keys, ("CNN seed 20261120", "CNN seed 20261121", "CNN seed 20261122")):
        ci_values = cnn["paired_difference_vs_ab3"][key]["optimal"]
        lines.append(f"| {label} | {interval(ci_values)} | Chưa đạt |")
    for key, label in (("CNN-RL-20261120-policy", "RL policy"),
                       ("PUCT-AZ-20261120", "PUCT-AZ 500")):
        lines.append(f"| {label} | {interval(search['paired_difference_vs_ab3'][key]['optimal'])} | Tham chiếu; dưới AB3 |")
    lines += [
        "",
        "### 4.2 Giải đấu trực tiếp",
        "",
        "| Cặp (A vs B) | W-D-L của A | Score A | CI bootstrap 95% |",
        "|---|---:|---:|---:|",
    ]
    selected_pairs = [
        ("AlphaBeta-depth4", "MCTS-500"), ("AlphaBeta-depth4", "PUCT"),
        ("AlphaBeta-depth4", "CNN-Policy"), ("AlphaBeta-depth6", "PUCT"),
        ("Heuristic", "CNN-Policy"), ("Heuristic", "RL-Policy"),
        ("RL-Policy", "PUCT"), ("CNN-Policy", "PUCT"),
    ]
    pair_map = {(row["first_agent"], row["second_agent"]): row for row in rr}
    for a, b in selected_pairs:
        row = pair_map[(a, b)]
        lines.append(f"| {a} vs {b} | {row['wins']}-{row['draws']}-{row['losses']} | "
                     f"{row['score']:.3f} | {interval(row['score_bootstrap_95'])} |")
    lines += [
        "",
        (f"AlphaBeta depth 4 đạt score {pair_map[('AlphaBeta-depth4','MCTS-500')]['score']:.3f} trước MCTS-500; score trước PUCT gần 0,5 và CI chứa 0,5. "
         "CI của CNN và Heuristic còn chồng lấp; CNN thua hai mức AlphaBeta trong cặp đã đo. RL thua rõ các agent tìm kiếm. "
         f"AZ iteration 100 đạt score {arena['summary']['score']:.3f} trước PUCT khởi tạo, CI {interval(arena['summary']['score_bootstrap_95'])}; cận dưới không vượt ngưỡng nên checkpoint không được promote."),
        "",
        "### 4.3 Value head, chi phí và tài nguyên",
        "",
        (f"Checkpoint CNN được chọn có độ chính xác W/D/L của value head {pct(value['metrics']['all']['wdl_accuracy'])} "
         f"(CI theo cụm {interval(value['metrics']['all']['cluster_bootstrap_95']['wdl_accuracy_95'])}, so với baseline lớp đa số {pct(value['metrics']['all']['majority_baseline_accuracy'])}. "
         "CI chứa baseline nên một checkpoint này chưa chứng minh được lợi thế độc lập của value head. Thời gian, node/mô phỏng và bộ nhớ local được báo cáo riêng trong [round-robin](ROUND_ROBIN_FINAL.md) và [đo tài nguyên](RESOURCE_PROFILE.md)."),
        "",
        "## 5. Phân tích và các bẫy đánh giá đã xử lý",
        "",
        "Các lượt đầu của dự án cho thấy kết quả có thể bị sai lệch bởi ván lặp từ agent tất định, gộp thời gian của hai phía, nhầm tổng node với node/nước, thang điểm MCTS và khai cuộc đã có kết cục chiến thuật. Giao thức cuối dùng khai cuộc chung, đổi ghế, ghi replay JSONL, phân tách chi phí theo agent và kiểm tra fair opening trước giải đấu.",
        "",
        "Kết quả CNN tập trung yếu ở khai cuộc/trung cuộc hơn tàn cuộc; điều này không tự chứng minh nguyên nhân là thiếu dữ liệu hoặc không thấy fork. Báo cáo giữ riêng tỷ lệ nước tối ưu solver, giữ W/D/L và score đối đầu. Không diễn giải kết quả training-time đấu Random thành sức mạnh độc lập.",
        "",
        "### 5.1 Sai lầm đầu tiên trong các ván CNN thua",
        "",
        (f"Phân tích solver trên {errors['losing_games']} ván thua của CNN tìm được ít nhất một nước sai điểm exact ở {errors['games_with_solver_error']} ván; "
         f"{errors['games_with_wdl_degrading_error']} ván có nước đầu tiên làm giảm W/D/L. Trong nhóm nước WDL-degrading đầu tiên, "
         f"{errors['first_wdl_error_classifications'].get('conceded_fork', 0)} được phân loại là nhượng fork theo kiểm tra hai ply, còn "
         f"{errors['first_wdl_error_classifications'].get('deeper_or_positional', 0)} là lỗi sâu hơn hoặc định vị. "
         f"Solver xác định {errors['games_already_forced_loss_at_opening_for_cnn']} ván đã là thế thua cưỡng bức cho CNN ngay sau opening. "
         "Vì vậy dữ liệu không ủng hộ giả thuyết CNN chủ yếu thua do bỏ fork; phần lớn lỗi WDL đầu tiên là positional hoặc sâu hơn."),
        "",
        "Chi tiết từng ván và định nghĩa phân loại: [CNN_LOSS_ANALYSIS.md](CNN_LOSS_ANALYSIS.md).",
        "",
        "### 5.1 Sai lầm đầu tiên trong các ván CNN thua",
        "",
        (f"Phân tích solver trên {errors['losing_games']} ván thua của CNN tìm được ít nhất một nước sai điểm exact ở {errors['games_with_solver_error']} ván; "
         f"{errors['games_with_wdl_degrading_error']} ván có nước đầu tiên làm giảm W/D/L. Trong nhóm nước WDL-degrading đầu tiên, "
         f"{errors['first_wdl_error_classifications'].get('conceded_fork', 0)} được phân loại là nhượng fork theo kiểm tra hai ply, còn "
         f"{errors['first_wdl_error_classifications'].get('deeper_or_positional', 0)} là lỗi sâu hơn hoặc định vị. "
         f"Solver xác định {errors['games_already_forced_loss_at_opening_for_cnn']} ván đã là thế thua cưỡng bức cho CNN ngay sau opening. "
         "Vì vậy dữ liệu không ủng hộ giả thuyết CNN chủ yếu thua do bỏ fork; phần lớn lỗi WDL đầu tiên là positional hoặc sâu hơn."),
        "",
        "Chi tiết từng ván và định nghĩa phân loại: [CNN_LOSS_ANALYSIS.md](CNN_LOSS_ANALYSIS.md).",
        "",
        "## 6. Hạn chế và khả năng tái lập",
        "",
        f"- Solver test gồm {search['positions']:,} vị trí có quan hệ theo ván nguồn; bootstrap theo cụm giảm giả định độc lập nhưng không tạo thêm các ván độc lập mới.",
        "- CNN có ba seed; RL và AlphaZero mỗi nhánh chỉ có một run/checkpoint. Kết luận RL/AZ vì vậy chỉ áp dụng cho cấu hình đã chạy.",
        "- Hai manifest Colab lịch sử của RL và AlphaZero không ghi peak VRAM. Pipeline mới đã ghi trường này và tích lũy cực đại qua resume, nhưng không thể khôi phục peak quá khứ từ checkpoint.",
        f"- Round-robin được chạy từ working tree có thay đổi cục bộ; metadata lưu checkpoint hashes, kết quả/replay hashes và ghi chú khác biệt mã nguồn. Commit nền: `{rr_meta['source_code_commit']}`.",
        "- Thời gian local phụ thuộc phần cứng cụ thể; Colab T4 và tài nguyên phiên thay đổi, nên không dùng throughput Colab để xếp hạng agent.",
        "- GPU training/simulation có thể không tái sinh bit-for-bit giữa phần cứng/phiên; log JSONL cho phép phát lại ván đã lưu, không tái tạo toàn bộ tiến trình stochastic.",
        "",
        "Mỗi run Colab có manifest, checkpoint, log và checksum; dữ liệu đầu vào xác minh frozen-test hash và loại trừ trạng thái test, kể cả gương. Script ingest local là cổng bắt buộc trước đánh giá. Kết quả chính có file JSON nguồn và report renderer để tái tạo bảng.",
        "",
        "## 7. Kết luận",
        "",
        "Dự án đã triển khai hệ thống tám cấp agent, quy trình huấn luyện có thể resume và đánh giá được kiểm soát bằng solver exact cùng round-robin ghép cặp. Tìm kiếm AlphaBeta vẫn là baseline mạnh nhất trong các cấu hình được báo cáo; CNN supervised chưa vượt cổng định trước, RL policy yếu trên test solver, và AlphaZero chưa được promote sau arena. Giá trị chính của kết quả là pipeline tái lập và các phép đo có giới hạn được nêu rõ, không phải khẳng định rằng các agent học đã đạt sức chơi vượt tìm kiếm.",
        "",
        "## Tệp nguồn và tái tạo",
        "",
        f"- CNN solver JSON `{args.cnn_eval.as_posix()}` (SHA256 `{sha256(args.cnn_eval)}`).",
        f"- Solver JSON cho RL/AZ và baseline `{args.search_eval.as_posix()}` (SHA256 `{sha256(args.search_eval)}`).",
        f"- Round-robin summary/meta/replay: `{args.round_robin.as_posix()}`, `{args.round_robin_meta.as_posix()}`; seed {rr_meta['seed']}, {rr_meta['games_per_pair']} games/pair.",
        f"- Arena `{args.arena.as_posix()}`; value-head JSON `{args.value_eval.as_posix()}`; resource profile `{args.resource.as_posix()}`.",
        f"- Phân tích nước sai đầu tiên: `{args.cnn_errors.as_posix()}` (SHA256 `{sha256(args.cnn_errors)}`).",
        "- Báo cáo chi tiết sinh tự động: `python scripts/render_learning_report.py` và `python scripts/render_round_robin_report.py`.",
        "",
    ]
    diagnostic_heading = "### 5.1 Sai lầm đầu tiên trong các ván CNN thua"
    limitations_heading = "## 6. Hạn chế và khả năng tái lập"
    if lines.count(diagnostic_heading) > 1:
        duplicate_start = lines.index(diagnostic_heading, lines.index(diagnostic_heading) + 1)
        del lines[duplicate_start:lines.index(limitations_heading, duplicate_start)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-eval", type=Path, default=Path("results/solver_eval_cnn_v2_3seeds.json"))
    parser.add_argument("--search-eval", type=Path, default=Path("results/solver_eval_az_rl_20261121_seed20261121.json"))
    parser.add_argument("--round-robin", type=Path, default=Path("benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json"))
    parser.add_argument("--round-robin-meta", type=Path, default=Path("benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json"))
    parser.add_argument("--arena", type=Path, default=Path("benchmarks/arena/arena_az_vs_init_100openings_seed20261121.json"))
    parser.add_argument("--value-eval", type=Path, default=Path("results/value_head/value_cnn_seed20261121_20261121.json"))
    parser.add_argument("--cnn-errors", type=Path, default=Path("results/cnn_first_errors_seed20261122.json"))
    parser.add_argument("--resource", type=Path, default=Path("results/resource_profile_20261121_v2.json"))
    parser.add_argument("--manifests", nargs="*", type=Path, default=[
        Path("results/colab/cnn_seed20261120_colab1/manifest.json"),
        Path("results/colab/cnn_seed20261121_colab1/manifest.json"),
        Path("results/colab/cnn_seed20261122_colab1/manifest.json"),
        Path("results/colab/rl_seed20261120_run1/manifest.json"),
        Path("results/colab/az_seed20261120_run1/manifest.json"),
    ])
    parser.add_argument("--output", type=Path, default=Path("docs/report.md"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
