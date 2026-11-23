"""Regenerate Phase C v2 analysis from immutable JSON artifacts (UTF-8)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results/phase_c_local/validation_curve_20261123.json"
SOLVER = ROOT / "results/solver_eval_phase_c_local_20261123.json"
RR = ROOT / "results/round_robin_phase_c_20261123.json"
RR_META = ROOT / "results/round_robin_phase_c_20261123.meta.json"
RUNS = ROOT / "results/phase_c_local"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def ci(values: list[float]) -> str:
    return f"[{100*values[0]:.2f}; {100*values[1]:.2f}]%"


def signed_pp(x: float) -> str:
    return f"{100*x:+.2f} đpt"


def main() -> None:
    phase, solver, rr, meta = map(load, (PHASE, SOLVER, RR, RR_META))
    completed = phase["completed"]
    assert len(completed) == 18
    assert solver["positions"] == 1500
    assert len(rr) == 28
    assert all(r["wins"] + r["draws"] + r["losses"] == r["games"] for r in rr)
    assert all(abs(r["score"] - (r["wins"] + .5*r["draws"])/r["games"]) < 1e-12 for r in rr)

    lines = [
        "# Kết quả Phase C v2: self-play PUCT", "",
        "Ngày chốt kết quả: 23/11/2026. Nội dung này được sinh từ JSON nguồn; kết quả cũ được giữ nguyên.", "",
        "## Thiết kế và kiểm soát", "",
        "Hai run hoàn tất 40 vòng, mỗi vòng 32 ván self-play và 200 bước cập nhật, batch 128. PUCT dùng 400 mô phỏng/nước, 16 ván chạy song song. Warm-start khởi tạo từ CNN v3; scratch khởi tạo ngẫu nhiên. Huấn luyện chạy local trên Windows 11 với RTX 3050 Laptop/CUDA; thời gian không so sánh với Colab.", "",
        "Checkpoint được chọn bằng validation: warm vòng 0 và scratch vòng 40, theo exact-optimal của PUCT có safety filter. Tập validation có 4.530 vị trí; sau khi loại trùng replay còn 4.490 vị trí warm và 4.497 scratch. CI là bootstrap 10.000 lần theo cụm ván nguồn. Đây là một seed mỗi nhánh; kết quả học là thăm dò.", "",
        "## Đường cong validation", "",
        "| Run | Vòng | PUCT có filter: exact (95% CI) | Hiệu ghép cặp so với vòng 0 (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for mode, label in (("warm", "Warm"), ("scratch", "Scratch")):
        for iteration in range(0, 41, 5):
            item = completed[f"{mode}_iter{iteration:02d}"]
            metric = item["metrics"]["puct_filtered"]
            if iteration == 0:
                delta = "—"
            else:
                paired = item["paired_delta_vs_iteration_0"]["puct_filtered"]
                delta = f"{signed_pp(paired['delta'])} [{signed_pp(paired['cluster_bootstrap_95'][0])}; {signed_pp(paired['cluster_bootstrap_95'][1])}]"
            lines.append(f"| {label} | {iteration} | {pct(metric['exact_optimal_rate'])} {ci(metric['cluster_bootstrap_95'])} | {delta} |")

    lines += ["", "Warm-start suy giảm sau self-play; scratch tăng từ vòng 0 đến 40 nhưng hiệu ghép cặp có CI chứa 0, nên chưa phân biệt được cải thiện. Không suy diễn nguyên nhân từ đường cong này.", "",
              "Bộ JSON cũng chứa policy-only/PUCT, có và không safety filter cho mỗi snapshot. Loss policy/value riêng và entropy target không được ghi; do đó không có đường cong cho các đại lượng này.", "",
              "## Đánh giá test đóng băng", "",
              "Chỉ sau khi chọn checkpoint trên validation, đánh giá dùng đủ 1.500 vị trí đóng băng. Exact là tỷ lệ chọn nước thuộc tập nước tối ưu Pons; WDL là tỷ lệ bảo toàn kết cục thắng/hòa/thua. CI bootstrap theo cụm ván nguồn.", "",
              "| Agent/cấu hình | Exact (95% CI) | WDL (95% CI) |",
              "|---|---:|---:|"]
    selected = [
        ("CNN-CNNv3-policy", "CNN v3 policy, có filter"),
        ("PUCT-PUCTwarm_iter00-raw", "PUCT-400 warm vòng 0, thô"),
        ("PUCT-PUCTwarm_iter00-filtered", "PUCT-400 warm vòng 0, có filter"),
        ("PUCT-PUCTscratch_iter40-raw", "PUCT-400 scratch vòng 40, thô"),
        ("PUCT-PUCTscratch_iter40-filtered", "PUCT-400 scratch vòng 40, có filter"),
    ]
    for key, label in selected:
        m = solver["metrics"][key]["all"]
        lines.append(f"| {label} | {pct(m['optimal']['rate'])} {ci(m['optimal']['cluster_bootstrap_95'])} | {pct(m['wdl_preserving']['rate'])} {ci(m['wdl_preserving']['cluster_bootstrap_95'])} |")

    ab3 = solver["metrics"]["AlphaBeta-depth3"]["all"]
    warm = solver["metrics"]["PUCT-PUCTwarm_iter00-filtered"]["all"]
    scratch = solver["metrics"]["PUCT-PUCTscratch_iter40-filtered"]["all"]
    warm_ci = solver["paired_difference_vs_ab3"]["PUCT-PUCTwarm_iter00-filtered"]
    scratch_ci = solver["paired_difference_vs_ab3"]["PUCT-PUCTscratch_iter40-filtered"]
    lines += ["", f"Warm vòng 0 so với AlphaBeta depth 3: exact {signed_pp(warm['optimal']['rate']-ab3['optimal']['rate'])}, CI ghép cặp [{signed_pp(warm_ci['optimal'][0])}; {signed_pp(warm_ci['optimal'][1])}]; WDL {signed_pp(warm['wdl_preserving']['rate']-ab3['wdl_preserving']['rate'])}, CI [{signed_pp(warm_ci['wdl_preserving'][0])}; {signed_pp(warm_ci['wdl_preserving'][1])}]. Cả hai CI không chứa 0.",
               f"Scratch vòng 40 so với AlphaBeta depth 3: exact {signed_pp(scratch['optimal']['rate']-ab3['optimal']['rate'])}, CI ghép cặp [{signed_pp(scratch_ci['optimal'][0])}; {signed_pp(scratch_ci['optimal'][1])}]; WDL {signed_pp(scratch['wdl_preserving']['rate']-ab3['wdl_preserving']['rate'])}, CI [{signed_pp(scratch_ci['wdl_preserving'][0])}; {signed_pp(scratch_ci['wdl_preserving'][1])}]. Các CI chứa 0.", "",
               "## Round-robin sau Phase C", "",
               f"Tám agent, {meta['games_per_pair']} ván/cặp, {meta['opening_plies']} ply mở đầu, ghép cặp theo khai cuộc và đổi ghế. Bộ lọc công bằng thử {meta['opening_generation']['attempts']} khai cuộc; giữ {meta['opening_generation']['accepted']}, loại {meta['opening_generation']['filtered_immediate_win']} khai cuộc có thắng tức thì và {meta['opening_generation']['filtered_no_safe_action']} khai cuộc không có nước an toàn. Engine bitboard; seed {meta['seed']}; không giới hạn thời gian. Score = (thắng + 0,5 hòa)/số ván; CI bootstrap theo paired opening.", "",
               "| Agent A | Agent B | W-D-L A | Score A | CI 95% |",
               "|---|---|---:|---:|---:|"]
    for r in rr:
        lines.append(f"| {r['first_agent']} | {r['second_agent']} | {r['wins']}-{r['draws']}-{r['losses']} | {r['score']:.3f} | [{r['score_bootstrap_95'][0]:.3f}; {r['score_bootstrap_95'][1]:.3f}] |")

    lines += ["", "## Cấu hình học và giới hạn", ""]
    for run_id in ("az_v2_warm_seed20261121", "az_v2_scratch_seed20261122"):
        run_dir = RUNS / run_id
        manifest = load(run_dir / "manifest.json")
        logs = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        avg_plies = sum(x["mean_game_plies"] for x in logs) / len(logs)
        draws = sum(x["selfplay_draws"] for x in logs)
        lines.append(f"- `{run_id}`: buffer cuối {logs[-1]['replay_positions']:,}; trung bình {avg_plies:.2f} ply/ván; {draws}/1.280 hòa; {manifest['resume_count']} lần resume.")
    lines += ["", "Replay cap 100.000. Mỗi vòng sampling có hoàn lại `200 × 128 = 25.600` vị trí. Mirror augmentation được áp dụng ngẫu nhiên 50% trong batch, không nhân đôi buffer. Adam, learning rate 0,001 cố định, không scheduler, weight decay mặc định 0; loss = policy cross-entropy + 0,5 × MSE value. Policy target chuẩn hóa visit count; temperature 1 trong 12 ply đầu rồi 0. Value target nhìn từ người đến lượt: thắng +1, thua −1, hòa 0. Self-play không có safety filter và không đấu với checkpoint tốt nhất.", "",
               "Train log lưu loss tổng, độ dài ván và số hòa từng vòng; không lưu policy/value loss riêng hay entropy policy. Validation không tính WDL/value accuracy/calibration theo snapshot. Chưa chạy PUCT-100/1600, MCTS rollout đối chứng, arena snapshot hoặc ablation LR/update. Các kiểm tra batched-vs-single inference và test dấu end-to-end vẫn còn thiếu.", "",
               "## File nguồn", "",
               f"- Validation: `{PHASE.relative_to(ROOT).as_posix()}`.",
               f"- Test solver: `{SOLVER.relative_to(ROOT).as_posix()}`.",
               f"- Round-robin: `{RR.relative_to(ROOT).as_posix()}`; metadata `{RR_META.relative_to(ROOT).as_posix()}`.", ""]
    (ROOT / "docs/PHASE_C_V2_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    rr_lines = ["# Round-robin cuối sau Phase C v2", "", f"Nguồn: `{RR.relative_to(ROOT).as_posix()}`; seed {meta['seed']}; CI 95% bootstrap 10.000 lần theo paired opening.", "",
                "Giao thức: 100 ván mỗi cặp; 50 khai cuộc 6 ply dùng chung cho hai ghế; engine bitboard; bộ lọc fair; không có ngân sách thời gian. Trong 61 lần thử khai cuộc, 50 được giữ; 10 bị loại do thắng tức thì khả dụng, một lần bị loại do trùng trạng thái.", "",
                "| Agent A | Agent B | W-D-L A | Score A | CI 95% |", "|---|---|---:|---:|---:|"]
    for r in rr:
        rr_lines.append(f"| {r['first_agent']} | {r['second_agent']} | {r['wins']}-{r['draws']}-{r['losses']} | {r['score']:.3f} | [{r['score_bootstrap_95'][0]:.3f}; {r['score_bootstrap_95'][1]:.3f}] |")
    rr_lines += ["", "Các so sánh trọng tâm:", "", "| So sánh | Score A | CI 95% | Diễn giải |", "|---|---:|---:|---|"]
    index = {(r["first_agent"], r["second_agent"]): r for r in rr}
    for a, b in (("AlphaBeta-depth4", "MCTS-500"), ("AlphaBeta-depth4", "PUCT"), ("AlphaBeta-depth6", "PUCT"), ("CNN-Policy", "PUCT"), ("PUCT", "RL-Policy")):
        r = index.get((a,b))
        reverse = False
        if r is None:
            r = index[(b,a)]; reverse = True
        score = 1-r["score"] if reverse else r["score"]
        lo, hi = r["score_bootstrap_95"]
        if reverse: lo, hi = 1-hi, 1-lo
        verdict = "CI bevat 0,5: chưa phân biệt được" if lo <= .5 <= hi else ("score trên 0,5" if score > .5 else "score dưới 0,5")
        rr_lines.append(f"| {a} – {b} | {score:.3f} | [{lo:.3f}; {hi:.3f}] | {verdict} |")
    rr_lines.append("")
    (ROOT / "docs/ROUND_ROBIN_PHASE_C_20261123.md").write_text("\n".join(rr_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
