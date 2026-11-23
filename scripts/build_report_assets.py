"""Build report tables and plots from frozen result JSON files.

Run from repository root: python scripts/build_report_assets.py
The Markdown is the editable source. PDF/DOCX export is handled separately when
the local office renderer is available.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def percent(number: float) -> str:
    return f"{100 * number:.1f}%"


def ci(interval: list[float]) -> str:
    return f"[{100 * interval[0]:.1f}; {100 * interval[1]:.1f}]%"


def table(headers: list[str], rows: list[tuple]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


solver = load("results/solver_eval_cnn_v2_3seeds.json")
learning = load("results/solver_eval_az_rl_20261121_seed20261121.json")
puct_cnn = load("results/solver_eval_level7_puct500.json")
round_robin = load("benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json")
rr_meta = load("benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json")
value_data = load("results/value_head/value_cnn_seed20261121_20261121.json")
error_data = load("results/cnn_first_errors_seed20261122.json")
budget_data = load("benchmarks/head_to_head/benchmark_ab_mcts_standard_100_paired_seed20261126.json")[0]
ablation_data = load("benchmarks/ablation/ablation_vs_heuristic_3seeds_paired.json")
rl_manifest = load("results/colab/rl_seed20261120_run1/manifest.json")
az_manifest = load("results/colab/az_seed20261120_run1/manifest.json")
resource_data = load("results/resource_profile_20261121_v2.json")

metrics = solver["metrics"]
az_metrics = learning["metrics"]
puct_metrics = puct_cnn["metrics"]

# Figure 1: exact solver choice rate and cluster-bootstrap intervals.
rows = [
    ("Random", metrics["Random"]), ("Heuristic", metrics["Heuristic"]),
    ("AB depth 3", metrics["AlphaBeta-depth3"]), ("AB depth 4", metrics["AlphaBeta-depth4"]),
    ("AB depth 5", metrics["AlphaBeta-depth5"]), ("AB depth 6", metrics["AlphaBeta-depth6"]),
    ("MCTS-500", metrics["MCTS-500"]),
]
labels = [x[0] for x in rows]
stats = [x[1]["all"]["optimal"] for x in rows]
cnn_seeds = [metrics[f"CNN-CNN_v2_s{s}-policy"]["all"]["optimal"] for s in ("20261120", "20261121", "20261122")]
labels += ["CNN seed 20", "CNN seed 21", "CNN seed 22"]
stats += cnn_seeds
for name, source, key in (("RL policy", az_metrics, "CNN-RL-20261120-policy"), ("PUCT-AZ-500", az_metrics, "PUCT-AZ-20261120"), ("PUCT-CNN-500", puct_metrics, "PUCT-PUCT_CNN_s20261121")):
    labels.append(name)
    stats.append(source[key]["all"]["optimal"])
values = np.array([s["rate"] for s in stats])
low = np.array([s["cluster_bootstrap_95"][0] for s in stats])
high = np.array([s["cluster_bootstrap_95"][1] for s in stats])
fig, ax = plt.subplots(figsize=(9.6, 5.8))
y = np.arange(len(labels))
ax.barh(y, 100 * values, color="#477b9e")
ax.errorbar(100 * values, y, xerr=[100 * (values - low), 100 * (high - values)], fmt="none", color="#202830", capsize=3)
ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="Nước tối ưu exact (%)", title="Solver evaluation trên 1.500 vị trí đóng băng")
ax.invert_yaxis(); ax.grid(axis="x", alpha=.25); fig.tight_layout()
fig.savefig(ASSETS / "solver_exact.png", dpi=240); plt.close(fig)

# Figure 2: stage results for representative agents.
stages = ["stage_opening", "stage_middle", "stage_endgame"]
stage_names = ["Khai cuộc", "Trung cuộc", "Tàn cuộc"]
stage_agents = [
    ("Heuristic", metrics["Heuristic"]), ("AB depth 3", metrics["AlphaBeta-depth3"]),
    ("AB depth 4", metrics["AlphaBeta-depth4"]), ("MCTS-500", metrics["MCTS-500"]),
    ("CNN mean", None), ("RL", az_metrics["CNN-RL-20261120-policy"]), ("PUCT-AZ", az_metrics["PUCT-AZ-20261120"]),
]
fig, ax = plt.subplots(figsize=(9.2, 5.4)); x = np.arange(3); width = .11
for i, (name, source) in enumerate(stage_agents):
    vals = ([np.mean([metrics[f"CNN-CNN_v2_s{s}-policy"][k]["optimal"]["rate"] for s in ("20261120", "20261121", "20261122")]) for k in stages]
            if source is None else [source[k]["optimal"]["rate"] for k in stages])
    ax.bar(x + (i - 3) * width, np.array(vals) * 100, width, label=name)
ax.set(xticks=x, xticklabels=stage_names, ylim=(0, 100), ylabel="Nước tối ưu exact (%)", title="Tỷ lệ nước tối ưu theo giai đoạn")
ax.grid(axis="y", alpha=.25); ax.legend(ncol=4, fontsize=8); fig.tight_layout()
fig.savefig(ASSETS / "solver_by_stage.png", dpi=240); plt.close(fig)

# Figure 3: score matrix, symmetrically derived from paired-match JSON.
agents = list(dict.fromkeys(a for r in round_robin for a in (r["first_agent"], r["second_agent"])))
matrix = np.full((len(agents), len(agents)), np.nan); np.fill_diagonal(matrix, .5)
for r in round_robin:
    i, j = agents.index(r["first_agent"]), agents.index(r["second_agent"])
    matrix[i, j] = r["score"]; matrix[j, i] = 1 - r["score"]
assert np.allclose(matrix + matrix.T, 1.0, equal_nan=True)
fig, ax = plt.subplots(figsize=(9, 7)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
ax.set(xticks=range(len(agents)), xticklabels=agents, yticks=range(len(agents)), yticklabels=agents, title="Score theo hàng gặp đối thủ theo cột")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for i in range(len(agents)):
    for j in range(len(agents)):
        ax.text(j, i, "—" if i == j else f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(image, ax=ax, label="score"); fig.tight_layout(); fig.savefig(ASSETS / "round_robin_score.png", dpi=240); plt.close(fig)

# Small architecture overview, authored as vector-like primitives in matplotlib.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis("off")
boxes = [(.03,.62,.20,.22,"Game engine\nGrid / Bitboard"),(.29,.78,.24,.16,"Agents 1–4\nRandom · Heuristic · Search"),(.29,.49,.24,.18,"Agents 5–6\nCNN · Policy Gradient"),(.29,.19,.24,.18,"Agents 7–8\nPUCT · Self-play"),(.62,.62,.18,.22,"Match runner\nOpenings · seats · logs"),(.84,.62,.14,.22,"Reports\nW-D-L · cost"),(.62,.19,.18,.22,"Pons solver\nExact values"),(.84,.19,.14,.22,"Frozen test\nCI · stages")]
for x0,y0,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.012",facecolor="#eaf1f8",edgecolor="#42627f",lw=1.1))
    ax.text(x0+w/2,y0+h/2,label,ha="center",va="center",fontsize=9)
def arrow(start,end): ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=11,color="#50677a",lw=1.1))
for target in [(.29,.86),(.29,.58),(.29,.28)]: arrow((.23,.72),target)
arrow((.53,.85),(.62,.73)); arrow((.53,.58),(.62,.69)); arrow((.53,.28),(.62,.29)); arrow((.80,.72),(.84,.72)); arrow((.80,.29),(.84,.29)); arrow((.71,.41),(.71,.62))
ax.set(xlim=(0,1),ylim=(0,1)); fig.tight_layout(); fig.savefig(ASSETS / "architecture.png", dpi=240, bbox_inches="tight"); plt.close(fig)

solver_rows = []
for label, key, source in [(name, key, metrics) for name, key in [("Random", "Random"), ("Heuristic", "Heuristic"), ("AB-3", "AlphaBeta-depth3"), ("AB-4", "AlphaBeta-depth4"), ("AB-5", "AlphaBeta-depth5"), ("AB-6", "AlphaBeta-depth6"), ("MCTS-500", "MCTS-500")]]:
    item = source[key]["all"]
    solver_rows.append((label, item["positions"], percent(item["optimal"]["rate"]), ci(item["optimal"]["cluster_bootstrap_95"]), percent(item["wdl_preserving"]["rate"]), ci(item["wdl_preserving"]["cluster_bootstrap_95"])))
for seed in ("20261120", "20261121", "20261122"):
    item = metrics[f"CNN-CNN_v2_s{seed}-policy"]["all"]
    solver_rows.append((f"CNN {seed[-2:]}", item["positions"], percent(item["optimal"]["rate"]), ci(item["optimal"]["cluster_bootstrap_95"]), percent(item["wdl_preserving"]["rate"]), ci(item["wdl_preserving"]["cluster_bootstrap_95"])))
for label, source, key in [("RL policy", az_metrics, "CNN-RL-20261120-policy"), ("PUCT-AZ-500", az_metrics, "PUCT-AZ-20261120"), ("PUCT-CNN-500", puct_metrics, "PUCT-PUCT_CNN_s20261121")]:
    item = source[key]["all"]
    solver_rows.append((label, item["positions"], percent(item["optimal"]["rate"]), ci(item["optimal"]["cluster_bootstrap_95"]), percent(item["wdl_preserving"]["rate"]), ci(item["wdl_preserving"]["cluster_bootstrap_95"])))

rr_index = {(r["first_agent"], r["second_agent"]): r for r in round_robin}
for result in round_robin:
    assert result["wins"] + result["draws"] + result["losses"] == result["games"], result
    expected = (result["wins"] + 0.5 * result["draws"]) / result["games"]
    assert abs(result["score"] - expected) < 1e-12, result
assert solver["positions"] == 1500 and learning["positions"] == 1500 and puct_cnn["positions"] == 1500
def oriented(a, b):
    if (a, b) in rr_index:
        return rr_index[(a, b)], False
    return rr_index[(b, a)], True
selected_pairs = [("AlphaBeta-depth4", "MCTS-500"), ("AlphaBeta-depth4", "PUCT"), ("CNN-Policy", "Heuristic"), ("AlphaBeta-depth4", "AlphaBeta-depth6"), ("CNN-Policy", "PUCT"), ("RL-Policy", "Random")]
rr_rows = []
for a, b in selected_pairs:
    r, reverse = oriented(a, b); score = 1-r["score"] if reverse else r["score"]
    lo0, hi0 = r["score_bootstrap_95"]; lo, hi = (1-hi0, 1-lo0) if reverse else (lo0, hi0)
    wins, losses = (r["losses"], r["wins"]) if reverse else (r["wins"], r["losses"])
    text = "Chưa phân biệt được với 0,5" if lo <= .5 <= hi else ("score > 0,5" if score > .5 else "score < 0,5")
    rr_rows.append((f"{a} – {b}", f"{wins}-{r['draws']}-{losses}", f"{score:.3f}", f"[{lo:.3f}; {hi:.3f}]", text))

cnn_values = [metrics[f"CNN-CNN_v2_s{s}-policy"]["all"]["optimal"]["rate"] for s in ("20261120", "20261121", "20261122")]
cnn_mean, cnn_sd = float(np.mean(cnn_values)), float(np.std(cnn_values, ddof=1))
cnn_paired = solver["paired_difference_vs_ab3"]
cnn_paired_rows = []
for seed in ("20261120", "20261121", "20261122"):
    interval = cnn_paired[f"CNN-CNN_v2_s{seed}-policy"]["optimal"]
    cnn_paired_rows.append((seed, f"[{100*interval[0]:.1f}; {100*interval[1]:.1f}] percentage points", "Cận trên < 0" if interval[1] < 0 else "CI chứa 0"))
ab4 = metrics["AlphaBeta-depth4"]["all"]["optimal"]
ab3 = metrics["AlphaBeta-depth3"]["all"]["optimal"]
rl = az_metrics["CNN-RL-20261120-policy"]["all"]["optimal"]
az = az_metrics["PUCT-AZ-20261120"]["all"]["optimal"]
p7 = puct_metrics["PUCT-PUCT_CNN_s20261121"]["all"]["optimal"]
cnn_open = np.mean([metrics[f"CNN-CNN_v2_s{s}-policy"]["stage_opening"]["optimal"]["rate"] for s in ("20261120", "20261121", "20261122")])
cnn_end = np.mean([metrics[f"CNN-CNN_v2_s{s}-policy"]["stage_endgame"]["optimal"]["rate"] for s in ("20261120", "20261121", "20261122")])
value_all = value_data["metrics"]["all"]
value_acc = value_all["wdl_accuracy"]
value_base = value_all["majority_baseline_accuracy"]
value_ci = value_all["cluster_bootstrap_95"]["wdl_accuracy_95"]
value_rows = []
for stage_name, key in (("Tổng thể", "all"), ("Khai cuộc", "opening"), ("Trung cuộc", "middle"), ("Tàn cuộc", "endgame")):
    entry = value_data["metrics"][key]
    interval = entry["cluster_bootstrap_95"]["wdl_accuracy_95"]
    value_rows.append((stage_name, entry["positions"], percent(entry["wdl_accuracy"]), percent(entry["majority_baseline_accuracy"]), f"[{100*interval[0]:.1f}; {100*interval[1]:.1f}]%", f"{entry['value_mse']:.3f}"))
resource_by_agent = {r["agent"]: r for r in resource_data["local_agents"]}
resource_rows = [(name, f"{resource_by_agent[name]['peak_rss_mib']:.1f}", f"{resource_by_agent[name]['peak_increment_mib']:.2f}") for name in ("AlphaBeta-depth4", "MCTS-500", "CNN-Policy", "PUCT-500")]
budget_lo, budget_hi = budget_data["score_bootstrap_95"]
ablation_nodes = {}
for item in ablation_data:
    ablation_nodes.setdefault(item["configuration"], []).append(item["agent_metrics"]["nodes_per_move"])
pruned_nodes = float(np.mean(ablation_nodes["ab-pruning-ordering"]))
unpruned_nodes = float(np.mean(ablation_nodes["ab-no-pruning"]))
prune_ratio = unpruned_nodes / pruned_nodes

report = f'''# Xây dựng và đánh giá Game AI đa cấp độ: Connect Four 7×6

**Báo cáo đề tài cuối kỳ – Trí tuệ nhân tạo nâng cao**
Nhóm: [CẦN BỔ SUNG: tên nhóm/mã nhóm]
Thành viên: [CẦN BỔ SUNG: họ tên, mã sinh viên, tỷ lệ đóng góp]
Giảng viên: [CẦN BỔ SUNG: họ tên giảng viên]
Ngày chốt dữ liệu: 23/11/2026

## Tóm tắt

Đề tài xây dựng hệ thống Connect Four 7×6 gồm các tác tử từ Random, Heuristic và tìm kiếm đối kháng đến MCTS, CNN policy/value, policy-gradient và self-play theo hướng AlphaZero. Hệ thống có hai engine lưới/bitboard, giao diện tác tử, bộ điều phối giải đấu và adapter gọi solver Pons để chấm vị trí chính xác. Trên tập test đóng băng 1.500 vị trí, AlphaBeta độ sâu 4 chọn nước tối ưu exact ở {percent(ab4['rate'])} (CI bootstrap theo cụm 95% {ci(ab4['cluster_bootstrap_95'])}); ba seed CNN đạt trung bình {percent(cnn_mean)} (độ lệch chuẩn giữa seed {100*cnn_sd:.1f} điểm phần trăm). PUCT-CNN cấp 7 đạt {percent(p7['rate'])}, còn PUCT-AZ cấp 8 đạt {percent(az['rate'])}. Round-robin ghép cặp cho thấy một số matchup như AlphaBeta-4 với MCTS-500 chưa phân biệt được theo score do CI chứa 0,5. CNN chưa đạt cổng đã đặt so với AlphaBeta-3; RL chỉ được huấn luyện đấu Random; AlphaZero-inspired chỉ có một run và thiếu checkpoint trung gian để kết luận self-play cải thiện. Phạm vi báo cáo dừng ở kết quả đã hoàn tất; các cải tiến tiếp theo được tạm hoãn do giới hạn GPU Colab.

**Từ khóa:** Connect Four; Alpha-Beta; MCTS; CNN; reinforcement learning; self-play; PUCT; solver.

## Mục lục

1. [Giới thiệu](#1-giới-thiệu) · 2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết-và-công-trình-liên-quan) · 3. [Phương pháp](#3-phương-pháp-và-thiết-kế-hệ-thống) · 4. [Kết quả](#4-thực-nghiệm-kết-quả-và-thảo-luận) · 5. [Kết luận](#5-kết-luận-và-hướng-phát-triển)

## Danh mục hình và bảng

Hình 1 kiến trúc hệ thống; Hình 2 nước tối ưu exact; Hình 3 theo giai đoạn; Hình 4 round-robin. Bảng 1 kết quả solver; Bảng 2 hiệu ghép cặp; Bảng 3 matchup; Bảng 4 value head; Bảng 5 thành phần cấp 7–8. Bảng RSS ở mục 4.5 là phép đo tài nguyên.

## Từ viết tắt

| Thuật ngữ | Diễn giải |
|---|---|
| AB | Alpha-Beta |
| CNN | Convolutional Neural Network |
| MCTS | Monte Carlo Tree Search |
| PUCT | UCT có policy prior |
| RL | Reinforcement Learning |
| WDL | Win/Draw/Loss; ở solver là giữ nguyên kết cục lý thuyết |
| CI | Khoảng tin cậy |

# 1. Giới thiệu

Connect Four là trò chơi hai người trên bàn 7 cột × 6 hàng. Mỗi lượt thả một quân vào cột; quân rơi xuống ô trống thấp nhất. Bốn quân liên tiếp theo ngang, dọc hoặc chéo tạo chiến thắng; bàn đầy mà không có chuỗi bốn là hòa. Đây là trò chơi thông tin đầy đủ, tất định, tổng bằng không và luân phiên lượt. Tromp ghi nhận khoảng 4,53 nghìn tỷ trạng thái hợp lệ; Allis chứng minh người đi trước thắng về lý thuyết trên bàn chuẩn nếu chơi tối ưu và khai cuộc ở cột giữa [1, 2]. Kết quả lý thuyết không đồng nghĩa các agent trong dự án tự chứng minh được chiến lược thắng.

Mục tiêu là xây dựng thang tác tử có độ phức tạp tăng dần, so sánh chất lượng nước đi, kết quả đối đầu và chi phí suy luận. Phạm vi chỉ gồm Connect Four 7×6. Câu hỏi nghiên cứu: (RQ1) các tác tử khác nhau thế nào theo solver và đối đầu; (RQ2) pruning/order ảnh hưởng chi phí ra sao; (RQ3) CNN có đạt giáo viên AlphaBeta-3 không; (RQ4) PUCT thêm gì cho policy; (RQ5) quy mô self-play/RL đã chạy có cho bằng chứng cải thiện; (RQ6) các kiểm soát đánh giá xử lý được sai lệch nào.

# 2. Cơ sở lý thuyết và công trình liên quan

## 2.1 Tìm kiếm đối kháng

Minimax giả định mỗi bên chọn hành động tối ưu theo mục tiêu đối nghịch. Alpha-Beta cắt nhánh không thể ảnh hưởng quyết định bằng các cận alpha/beta; lợi ích phụ thuộc thứ tự duyệt nước. Iterative deepening tìm lần lượt độ sâu tăng dần để duy trì phương án tốt nhất khi có giới hạn thời gian. Độ sâu cố định không tương đương ngân sách thời gian giữa các phương pháp.

## 2.2 MCTS và UCT

MCTS gồm chọn nút, mở rộng, mô phỏng và lan truyền kết quả. UCT cân bằng khai thác nhánh có giá trị cao với khám phá nhánh ít thăm [3]. Thang thưởng và góc nhìn giá trị phải nhất quán khi đổi lượt. MCTS trong dự án dùng rollout ngẫu nhiên nên nhạy theo seed và số simulation.

## 2.3 Policy/value CNN, RL và self-play

Bàn cờ được mã hóa thành hai mặt phẳng nhị phân: quân người đến lượt và quân đối thủ. CNN xuất policy trên bảy cột và value WDL. Học có giám sát bắt chước nhãn chuyên gia, do đó có thể thừa hưởng giới hạn của nguồn nhãn. Policy-gradient tối ưu policy theo phần thưởng episode; entropy khuyến khích khám phá. PUCT đưa policy prior vào tìm kiếm. AlphaGo/AlphaZero gốc dùng pipeline lớn hơn đáng kể; hai cấp tương ứng trong dự án chỉ là inspired, không phải tái hiện đầy đủ [4, 5].

## 2.4 Solver

Pons Connect Four Solver là chương trình C++ ngoài, gọi qua adapter Python và dùng làm nguồn điểm minimax; không phải đối thủ trong bảng xếp hạng. Mã nguồn phân phối theo AGPL-3.0 [6]. Kết quả agent được gọi là sức mạnh thực nghiệm; chỉ solver cung cấp đáp án exact cho vị trí.

# 3. Phương pháp và thiết kế hệ thống

## 3.1 Mô hình game và engine

State gồm bàn cờ, người đi, kết cục; action là cột hợp lệ. Transition đặt quân theo trọng lực, kiểm tra thắng/hòa và đổi lượt khi ván chưa kết thúc. Engine grid dễ kiểm tra; engine bitboard dùng bit để kiểm tra nhanh. Test parity đối chiếu diễn tiến giữa hai biểu diễn, cùng test nước sai, thắng, hòa và terminal.

## 3.2 Kiến trúc

![Hình 1. Kiến trúc tổng quát của hệ thống.](assets/architecture.png)

*Hình 1. Kiến trúc hệ thống và luồng đánh giá.*

Agent tương tác với engine qua giao diện chung. Bộ điều phối quản lý lượt, openings, ghế và log. Pons được gọi riêng để chấm vị trí. Pipeline Colab tạo checkpoint và manifest; ingest local kiểm tra gói trước khi dùng. Mọi phép đo thời gian agent lấy trên máy local; Colab chỉ là môi trường huấn luyện.

## 3.3 Tám cấp tác tử

| Cấp | Phương pháp và hiện trạng |
|---|---|
| 1 | Random: chọn nước hợp lệ ngẫu nhiên, baseline. |
| 2 | Heuristic: luật chiến thuật và điểm cửa sổ bốn ô. |
| 3 | Alpha-Beta: minimax, pruning, ordering và iterative deepening ở cấu hình giới hạn thời gian. |
| 4 | MCTS-UCT với rollout ngẫu nhiên và ngân sách mô phỏng. |
| 5 | CNN supervised: ba seed từ nhãn solver; policy dùng khi chơi thuần, value được kiểm tra và dùng tại PUCT. |
| 6 | Policy-gradient 10.000 episode, chỉ đấu Random khi huấn luyện. |
| 7 | AlphaGo-inspired: policy prior + value leaf trong PUCT; đánh giá 500 simulations. Chưa có vòng RL policy đầy đủ như AlphaGo. |
| 8 | AlphaZero-inspired: self-play PUCT, Dirichlet, temperature, replay, policy/value update, resume; Phase C v2 có hai run local 40 vòng, một warm-start và một scratch. Self-play chưa cải thiện rõ trên validation. |

Tactical filter thắng ngay/chặn ngay có thể đóng góp đáng kể cho agent mạng. Kết quả cần được hiểu là hiệu năng của toàn cấu hình; tập kết quả chính chưa có bảng đối xứng có/không filter cho mọi agent.

## 3.4 Dữ liệu và huấn luyện

Ba run CNN dùng bộ nhãn solver v2: batch 64, tối đa 30 epoch, learning rate 0,001, value loss weight 0,5 và mirror augmentation. Chia train/validation theo ván nguồn; manifest ghi không trùng frozen test. Mỗi epoch xấp xỉ 21 batch; tổng log là 609/630 bước tùy early stopping. Môi trường là Tesla T4, Python 3.13.16, PyTorch 2.5.1+cu124; seed 20261120–20261122.

RL policy-gradient chạy {rl_manifest['configuration']['episodes']:,} episode với Random, entropy weight {rl_manifest['configuration']['entropy_weight']}. AZ chạy {az_manifest['configuration']['iterations']} iterations × {az_manifest['configuration']['games_per_iteration']} ván, {az_manifest['configuration']['parallel_games']} ván song song, {az_manifest['configuration']['simulations']} simulations/nước, {az_manifest['configuration']['training_steps']} update steps/iteration, batch {az_manifest['configuration']['batch_size']}, replay cap {az_manifest['configuration']['replay_capacity']:,}, khởi tạo từ CNN seed 20261121. Manifest ghi {az_manifest['games_completed']:,} self-play games, {az_manifest['completed_steps']:,} update steps và {az_manifest['resume_count']} lần resume. Chỉ lưu checkpoint cuối, không có snapshot iteration để tái tạo learning curve độc lập.

## 3.5 Thiết kế đánh giá

Frozen solver test gồm 1.500 vị trí, 500 mỗi giai đoạn: ply 7–10, 11–24, 25–41. SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`; solver commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Metric chính là tỷ lệ nước tối ưu exact và tỷ lệ giữ WDL. CI 95% dùng cluster bootstrap theo ván nguồn.

Round-robin có {len(agents)} agent, {rr_meta['games_per_pair']} ván mỗi cặp, {rr_meta['opening_generation']['accepted']} opening ghép cặp, đổi ghế. Trong {rr_meta['opening_generation']['attempts']} opening thử, bộ lọc giữ {rr_meta['opening_generation']['accepted']} và loại {rr_meta['opening_generation']['attempts']-rr_meta['opening_generation']['accepted']}. Score = (thắng + 0,5 × hòa)/số ván; CI bootstrap theo opening ghép cặp. Đây là đấu theo cấu hình cố định, không phải cùng ngân sách thời gian. Benchmark 0,5 s/nước được báo riêng.

## 3.6 Bẫy đánh giá và xử lý

| Vấn đề | Xử lý và giới hạn |
|---|---|
| Ván lặp do agent tất định | Dùng chung opening, đổi ghế, ghi số opening hiệu dụng. |
| Thống kê hai bên bị gộp | JSON cuối lưu metric riêng từng agent; log cũ không so sánh trực tiếp. |
| Góc nhìn/thang điểm MCTS | Có test terminal/perspective; kết quả lấy từ bản đã sửa. |
| Khai cuộc có thắng tức thì | Lọc và ghi số lượng bị loại. |
| Rò rỉ test | Manifest các run CNN/RL/AZ ghi overlap 0; thử nghiệm đời sau có overlap được loại khỏi báo cáo. |
| Chọn ngưỡng hậu nghiệm | Dùng CRITERIA; phân tích sau được ghi nhãn thăm dò. |
| Phụ thuộc giữa vị trí | Bootstrap theo ván nguồn hoặc opening ghép cặp. |

# 4. Thực nghiệm, kết quả và thảo luận

## 4.1 Môi trường

Mã đánh giá ghi commit `5423c266894ee910d1b833d95aefe23a85f41613`; solver commit ở mục 3.5. Huấn luyện Colab ghi Tesla T4, 2 CPU cores, khoảng 12,7 GiB RAM, CUDA 12.4 và PyTorch 2.5.1+cu124. Thời gian suy luận đo local, không dùng runtime Colab làm so sánh agent.

## 4.2 Đánh giá solver

Hình 2 và Bảng 1 tóm tắt mức khớp nước tối ưu exact và giữ WDL. CNN mỗi seed được để riêng; không lấy CI của một seed làm CI của trung bình ba seed.

![Hình 2. Nước tối ưu exact trên tập test solver.](assets/solver_exact.png)

*Hình 2. Tỷ lệ chọn nước tối ưu exact và khoảng tin cậy bootstrap theo cụm.*

**Bảng 1. Kết quả solver trên tập test đóng băng.**
{table(["Agent", "N", "Exact", "CI 95%", "Giữ WDL", "CI 95%"], solver_rows)}

AlphaBeta-4 đạt {percent(ab4['rate'])} exact (CI {ci(ab4['cluster_bootstrap_95'])}). AlphaBeta-5 cao hơn trên mẫu này, trong khi depth 6 thấp hơn depth 5; không suy rộng xu hướng này ngoài vị trí đã lấy mẫu. MCTS-500 đạt {percent(az_metrics['MCTS-500']['all']['optimal']['rate'])} trong lượt solver tương ứng. Ba CNN đạt {percent(min(cnn_values))}–{percent(max(cnn_values))} exact, thấp hơn AlphaBeta-3 ({percent(ab3['rate'])}). Bảng ghép cặp dưới đây cho thấy CI của (CNN − AB3) đều nằm dưới 0; vì vậy cả ba seed không qua cổng tối thiểu đã đặt. PUCT-CNN cấp 7 và PUCT-AZ cấp 8 là hai checkpoint khác nhau.

**Bảng 2. Hiệu ghép cặp của từng seed CNN so với AlphaBeta-3.**
{table(["Seed CNN", "Hiệu exact CNN − AB3, CI 95%", "Diễn giải"], cnn_paired_rows)}

## 4.3 Kết quả theo giai đoạn

![Hình 3. Nước tối ưu exact theo giai đoạn.](assets/solver_by_stage.png)

*Hình 3. Tỷ lệ chọn nước tối ưu exact theo giai đoạn.*

CNN trung bình đạt {percent(cnn_open)} ở khai cuộc và {percent(cnn_end)} ở tàn cuộc. Mức thấp ở đầu ván phù hợp với chẩn đoán rằng lỗi không chỉ là bỏ lỡ fork đơn lẻ. Các vị trí phụ thuộc ván nguồn; CI dùng cluster bootstrap.

## 4.4 Đối đầu trực tiếp

![Hình 4. Ma trận score round-robin; hàng là agent được tính score.](assets/round_robin_score.png)

*Hình 4. Score của agent theo hàng khi gặp đối thủ theo cột.*

**Bảng 3. Các matchup round-robin được thảo luận.**
{table(["Cặp", "W-D-L", "Score", "CI 95% theo opening", "Diễn giải"], rr_rows)}

AB4–MCTS-500 score là {rr_index[("AlphaBeta-depth4", "MCTS-500")]['score']:.3f} (CI {rr_index[("AlphaBeta-depth4", "MCTS-500")]['score_bootstrap_95']}); AB4–PUCT cũng có CI chứa 0,5. Chưa phân biệt được hai cặp này. CNN–Heuristic có score {oriented("CNN-Policy", "Heuristic")[0]['score'] if not oriented("CNN-Policy", "Heuristic")[1] else 1-oriented("CNN-Policy", "Heuristic")[0]['score']:.3f} và CI chứa 0,5. RL policy thua đa số ván trước Random; reward lúc huấn luyện không thay phép đánh giá độc lập.

## 4.5 Ngân sách thời gian và ablation

Benchmark riêng AB-time-0,5s và MCTS-time-0,5s gồm {budget_data['games']} ván/{budget_data['unique_openings']} opening: AlphaBeta score {budget_data['score']:.3f}, CI paired-opening [{budget_lo:.3f}; {budget_hi:.3f}], nên chưa phân biệt với 0,5. AB trung bình {budget_data['per_agent'][budget_data['first_agent']]['average_move_seconds']:.3f} s/nước, MCTS {budget_data['per_agent'][budget_data['second_agent']]['average_move_seconds']:.3f} s/nước. Đây là cấu hình riêng, không nhập vào round-robin cố định.

Ablation AlphaBeta qua ba seed, mỗi cấu hình chỉ 10 ván/seed, cho trung bình {pruned_nodes:.0f} node/nước với pruning+ordering so với {unpruned_nodes:.0f} khi tắt pruning, tức giảm {prune_ratio:.1f} lần theo số node. Cỡ mẫu thấp nên đây là bằng chứng về chi phí tìm kiếm, không đủ để khẳng định sức chơi tốt hơn. Quét số simulation của MCTS cũng được xem là thăm dò.

Đo RSS local dưới đây lấy từ một ván cho mỗi tiến trình, sampling interval 100 ms; đây là đỉnh quan sát được, không phải giới hạn bộ nhớ cực đại. VRAM huấn luyện Colab được báo riêng từ manifest khi có, không dùng để so sánh agent.

{table(["Agent", "Peak RSS (MiB)", "RSS tăng so với nền (MiB)"], resource_rows)}

## 4.6 Chẩn đoán lỗi và value head

Trong {error_data['losing_games']} ván thua CNN được phân tích, {error_data['games_with_solver_error']} có ít nhất một nước dưới exact-optimal; {error_data['games_with_wdl_degrading_error']} có nước đầu tiên làm giảm WDL, {error_data['games_without_exact_score_error']} không có lỗi exact; {error_data['first_wdl_error_classifications']['conceded_fork']} lỗi đầu giảm WDL được phân loại fork hai ply và {error_data['first_wdl_error_classifications']['deeper_or_positional']} là lỗi sâu/định vị. Có {error_data['games_already_forced_loss_at_opening_for_cnn']} ván đã ở thế thua cưỡng bức sau opening. Đây là thống kê hậu nghiệm trên ván thua được log, không đại diện toàn bộ ván.

**Bảng 4. Độ chính xác và sai số value head theo giai đoạn.**
{table(["Giai đoạn", "N", "Accuracy WDL", "Baseline đa số", "CI 95% cluster", "MSE"], value_rows)}

Trên toàn tập, CI accuracy {ci(value_ci)} chứa baseline {percent(value_base)}; do đó chưa có bằng chứng value head vượt baseline. MSE được báo riêng, không quy đổi thành sức chơi.

## 4.7 Thảo luận và đe dọa hợp lệ

Solver exact và thắng thua đối kháng trả lời hai câu hỏi khác nhau. Tỷ lệ nước tối ưu phản ánh độ khớp quyết định trên tập test; score phản ánh đối đầu trên opening được chọn. CNN thuần nhanh hơn tìm kiếm nhưng còn khoảng cách với AB3. PUCT-CNN có exact cao hơn CNN thuần trong các file hiện có, song thiếu đối chứng cùng checkpoint policy-only có CI paired tương ứng để quy kết nhân quả. AZ chỉ một run, không có snapshot trung gian; RL chỉ học đấu Random. Không kết luận hai hướng này đã cải thiện sức chơi.

Giới hạn gồm một game, 1.500 vị trí solver, 50 opening hiệu dụng trong round-robin, phụ thuộc giữa các vị trí cùng ván, đóng góp của tactical filter, số seed hạn chế cho RL/AZ và tài nguyên Colab biến động. Pons dùng như thước đo ngoài, không phải agent dự thi.

# 5. Kết luận và hướng phát triển

| Câu hỏi | Kết luận | Mức chắc chắn |
|---|---|---|
| RQ1: so sánh agent | Agent tìm kiếm đạt exact cao hơn CNN/RL trong tập này; một số matchup có CI score chứa 0,5. | Solver có CI theo ván; matchup chưa phân biệt ở một số cặp. |
| RQ2: pruning/order | Số node giảm rõ trong ablation nhỏ. | Gợi ý về chi phí; sức chơi chưa đủ mẫu. |
| RQ3: CNN và giáo viên AB3 | Ba seed CNN thấp hơn AB3 và không qua cổng đặt trước. | Kết quả trên frozen test; hiệu ghép cặp là cổng định trước. |
| RQ4: PUCT và policy | PUCT-CNN có exact cao, nhưng thiếu ablation cùng checkpoint đủ để quy tác dụng. | Gợi ý, chưa chứng minh nhân quả. |
| RQ5: RL/self-play | Không có bằng chứng đủ mạnh về cải thiện; RL chỉ học với Random, AZ thiếu learning curve test. | Chưa kiểm chứng cải thiện. |
| RQ6: lợi thế đi trước | Đổi ghế đã áp dụng; chưa có phân tích riêng đầy đủ theo ghế cho mọi cặp. | Chưa kết luận định lượng. |

Hệ thống đã có tám mức tác tử ở mức triển khai khác nhau, benchmark đối kháng, solver evaluation và pipeline Colab/ingest. Kết quả cho thấy tìm kiếm cổ điển là mốc mạnh; CNN chưa qua cổng AB3; RL và AZ vẫn là thử nghiệm quy mô hạn chế. Nhóm tạm dừng trước khi chuẩn bị các lượt cải thiện tiếp theo do quota GPU Colab. Không có kết quả CNN v3 hoặc AZ v2 nào được đưa vào đây.

Ưu tiên tiếp theo: mở rộng solver labels không rò rỉ; huấn luyện nhiều seed; lưu checkpoint AZ định kỳ; ablation filter và policy-only/PUCT; tăng opening độc lập và báo cáo lợi thế ghế. Việc này là kế hoạch, chưa phải kết quả.

# Tài liệu tham khảo

[1] V. Allis, *A Knowledge-based Approach of Connect-Four: The Game is Solved: White Wins*, M.Sc. thesis, Vrije Universiteit Amsterdam, 1988, IR-163.
[2] J. Tromp, “John’s Connect Four Playground,” https://tromp.github.io/c4/c4, truy cập 21/11/2026.
[3] L. Kocsis and C. Szepesvári, “Bandit Based Monte-Carlo Planning,” *ECML 2006*, LNCS 4212, pp. 282–293, 2006, https://doi.org/10.1007/11871842_29.
[4] D. Silver et al., “Mastering the game of Go with deep neural networks and tree search,” *Nature*, 529, 484–489, 2016, doi:10.1038/nature16961, https://www.nature.com/articles/nature16961.
[5] D. Silver et al., “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” *Science*, 362(6419), 1140–1144, 2018, doi:10.1126/science.aar6404, https://doi.org/10.1126/science.aar6404.
[6] P. Pons, *Connect Four Solver*, AGPL-3.0, https://github.com/PascalPons/connect4.
[7] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018, https://mitpress.mit.edu/9780262039246/reinforcement-learning/.
[8] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2021/2022, https://www.pearson.com/en-us/subject-catalog/p/artificial-intelligence-a-modern-approach/P200000003500/9780137505135.

# Phụ lục A. Thành phần cấp 7–8

**Bảng 5. Thành phần cấp 7–8 đã hiện thực và còn thiếu.**
{table(["Cấp", "Đã hiện thực", "Chưa hiện thực/chưa chứng minh"], [("7 – AlphaGo-inspired", "Policy prior, value leaf, PUCT 500 simulations.", "Không có policy RL đầy đủ; chưa nhiều seed."), ("8 – AlphaZero-inspired", "CNN init, self-play PUCT, Dirichlet, temperature, replay, policy/value update, resume.", "Một run; 100 simulations/nước; thiếu snapshot/đối chứng scratch và learning curve solver.")])}

# Phụ lục B. Tái lập và chỗ cần bổ sung

Tái tạo bảng/hình: `python scripts/build_report_assets.py`. Bộ test: `python -m unittest discover -s tests`. Các file nguồn được lập tại `docs/report_sources.md`; vấn đề mở và phần hành chính thiếu nằm trong `docs/report_open_issues.md`. Build Word/PDF cần được kiểm tra bằng công cụ văn phòng có sẵn trước khi nộp.
'''

# Append the completed local Phase C results to the baseline report without
# replacing historical files. The tables are rendered from immutable JSON.
phase_c = load("results/phase_c_local/validation_curve_20261123.json")
phase_solver = load("results/solver_eval_phase_c_local_20261123.json")
phase_rr = load("results/round_robin_phase_c_20261123.json")
phase_meta = load("results/round_robin_phase_c_20261123.meta.json")
phase_rows = []
for mode, label in (("warm", "Warm"), ("scratch", "Scratch")):
    for iteration in range(0, 41, 5):
        item = phase_c["completed"][f"{mode}_iter{iteration:02d}"]
        metric = item["metrics"]["puct_filtered"]
        if iteration == 0:
            delta = "—"
        else:
            paired = item["paired_delta_vs_iteration_0"]["puct_filtered"]
            delta = f"{100*paired['delta']:+.2f} pp [{100*paired['cluster_bootstrap_95'][0]:+.2f}; {100*paired['cluster_bootstrap_95'][1]:+.2f}]"
        phase_rows.append((label, iteration, f"{100*metric['exact_optimal_rate']:.2f}%", ci(metric["cluster_bootstrap_95"]), delta))
phase_selected = [
    ("CNN v3 policy + filter", "CNN-CNNv3-policy"),
    ("PUCT-400 warm, iter 0 + filter", "PUCT-PUCTwarm_iter00-filtered"),
    ("PUCT-400 scratch, iter 40 + filter", "PUCT-PUCTscratch_iter40-filtered"),
]
phase_test_rows = []
for label, key in phase_selected:
    metric = phase_solver["metrics"][key]["all"]
    phase_test_rows.append((label, phase_solver["metrics"][key]["all"]["positions"], percent(metric["optimal"]["rate"]), ci(metric["optimal"]["cluster_bootstrap_95"]), percent(metric["wdl_preserving"]["rate"]), ci(metric["wdl_preserving"]["cluster_bootstrap_95"])))
phase_rr_rows = [(r["first_agent"], r["second_agent"], f"{r['wins']}-{r['draws']}-{r['losses']}", f"{r['score']:.3f}", f"[{r['score_bootstrap_95'][0]:.3f}; {r['score_bootstrap_95'][1]:.3f}]") for r in phase_rr]
phase_section = f'''\n## 4.4 Phase C v2: self-play PUCT\n\nHai run local hoàn tất 40 vòng, 400 simulations/nước, 32 ván và 200 cập nhật mỗi vòng, 16 ván song song, batch 128. Warm-start được khởi tạo từ CNN v3; scratch từ trọng số ngẫu nhiên. Checkpoint được chọn theo validation: warm iteration 0, scratch iteration 40. Mỗi nhánh có một seed nên kết luận về phương pháp học còn thăm dò.\n\nĐường cong dưới dùng tỷ lệ nước tối ưu exact của PUCT có safety filter. CI bootstrap theo cụm ván nguồn; hiệu là ghép cặp so với iteration 0 của cùng run.\n\n{table(["Run", "Iteration", "Exact", "CI 95%", "Delta vs 0, pp (CI)"] , phase_rows)}\n\nSelf-play làm giảm exact ở nhánh warm; nhánh scratch tăng 1,00 điểm phần trăm ở iteration 40 nhưng CI ghép cặp chứa 0. Trên frozen test 1.500 vị trí, các cấu hình được chọn đạt:\n\n{table(["Agent", "n", "Exact", "CI 95%", "WDL", "CI 95%"], phase_test_rows)}\n\nWarm iteration 0 so với AlphaBeta depth 3 có exact +4,93 pp (CI ghép cặp [{100*phase_solver['paired_difference_vs_ab3']['PUCT-PUCTwarm_iter00-filtered']['optimal'][0]:.2f}; {100*phase_solver['paired_difference_vs_ab3']['PUCT-PUCTwarm_iter00-filtered']['optimal'][1]:.2f}] pp) và WDL +3,47 pp (CI [{100*phase_solver['paired_difference_vs_ab3']['PUCT-PUCTwarm_iter00-filtered']['wdl_preserving'][0]:.2f}; {100*phase_solver['paired_difference_vs_ab3']['PUCT-PUCTwarm_iter00-filtered']['wdl_preserving'][1]:.2f}] pp). Scratch chưa phân biệt được với depth 3.\n\nRound-robin mới gồm tám agent, 100 ván/cặp, 50 opening chung đã qua fair filter, seed {phase_meta['seed']}; CI bootstrap theo paired opening.\n\n{table(["Agent A", "Agent B", "W-D-L A", "Score A", "CI 95%"], phase_rr_rows)}\n\nChi tiết manifest, stage metrics, chi phí, train log và các giới hạn chẩn đoán nằm trong `docs/PHASE_C_V2_RESULTS.md` và `docs/ROUND_ROBIN_PHASE_C_20261123.md`. Nguồn JSON: `results/phase_c_local/validation_curve_20261123.json`, `results/solver_eval_phase_c_local_20261123.json`, `results/round_robin_phase_c_20261123.json`.\n'''
report = report.replace("AlphaZero-inspired chỉ có một run và thiếu checkpoint trung gian để kết luận self-play cải thiện.", "Phase C v2 có hai run local; warm-start suy giảm trên validation, còn mức tăng của scratch chưa phân biệt được thống kê.")
report = report.replace("Phạm vi báo cáo dừng ở kết quả đã hoàn tất; các cải tiến tiếp theo được tạm hoãn do giới hạn GPU Colab.", "Phase C v2 và đánh giá round-robin cuối đã chạy local; các ablation mở rộng vẫn chưa thực hiện.")
report = report.replace("## 5. Phân tích và các bẫy đánh giá đã xử lý", phase_section + "\n## 5. Phân tích và các bẫy đánh giá đã xử lý")
report = report.replace("RL và AZ mỗi nhánh chỉ có một run/checkpoint.", "RL có một run; Phase C AZ v2 có hai run nhưng chỉ một seed cho mỗi nhánh.")
report = report.replace("AZ thiếu learning curve test.", "Phase C AZ v2 có đường cong validation; warm-start giảm, scratch chưa cho thấy cải thiện có ý nghĩa.")
report = report.replace("Nhóm tạm dừng trước khi chuẩn bị các lượt cải thiện tiếp theo do quota GPU Colab. Không có kết quả CNN v3 hoặc AZ v2 nào được đưa vào đây.", "Phase C v2 hoàn tất trên local sau khi quota Colab hạn chế. PUCT warm iteration 0 là cấu hình học mạnh nhất được chọn theo validation; scratch chưa thể hiện tăng trưởng có ý nghĩa. Kết quả CNN v3 và AZ v2 được trình bày cùng nguồn JSON.")
report = report.replace("Một run; 100 simulations/nước; thiếu snapshot/đối chứng scratch và learning curve solver.", "Hai run 40 vòng, warm và scratch, 400 simulations/nước; warm giảm trên validation, scratch tăng không có ý nghĩa thống kê. Chưa có nhiều seed, PUCT simulation sweep 100/1600 hay MCTS rollout đối chứng.")

phase_section = phase_section.replace("## 4.4 Phase C v2", "## 4.8 Phase C v2")
report = report.replace("# 5. Kết luận và hướng phát triển", phase_section + "\n# 5. Kết luận và hướng phát triển")
report = report.replace("AZ chỉ một run, không có snapshot trung gian; RL chỉ học đấu Random. Không kết luận hai hướng này đã cải thiện sức chơi.", "Phase C v2 có learning curve validation hai nhánh: warm giảm và scratch chưa cải thiện có ý nghĩa; RL chỉ học đấu Random. Kết quả self-play chưa chứng minh cải thiện phương pháp.")

report = chr(10).join(line.rstrip() for line in report.splitlines()) + chr(10)
(OUT / "report.md").write_text(report, encoding="utf-8")

claims = f'''# Truy vết số liệu trong báo cáo

File này được ghi cùng mỗi lần chạy `python scripts/build_report_assets.py`. Số và bảng trong báo cáo được đối chiếu với các nguồn sau:

| Nội dung được nêu | Nguồn chính |
|---|---|
| Bảng exact/WDL, AlphaBeta depth 1–6, ba CNN và CI | `results/solver_eval_cnn_v2_3seeds.json` → `metrics.*.all`; CNN dùng các entry `CNN-CNN_v2_s*-policy`. |
| RL, PUCT-AZ, MCTS lượt đánh giá tương ứng | `results/solver_eval_az_rl_20261121_seed20261121.json` → `metrics.*`. |
| PUCT-CNN cấp 7 | `results/solver_eval_level7_puct500.json` → `metrics.PUCT-PUCT_CNN_s20261121`. |
| Giai đoạn khai/mid/endgame | Ba file solver trên → `metrics.*.stage_opening/stage_middle/stage_endgame`. |
| Round-robin, W-D-L, score, CI, node/time theo bên | `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json`. |
| 56/50/6 opening | `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json` → `opening_generation`. |
| Benchmark thời gian 0,5 giây/nước | `benchmarks/head_to_head/benchmark_ab_mcts_standard_100_paired_seed20261126.json`. |
| Ablation node/nước | `benchmarks/ablation/ablation_vs_heuristic_3seeds_paired.json`. |
| CNN error diagnosis | `results/cnn_first_errors_seed20261122.json`. |
| Value accuracy, baseline và CI | `results/value_head/value_cnn_seed20261121_20261121.json` → `metrics.all`. |
| Value accuracy/MSE theo giai đoạn | Cùng JSON → `metrics.all/opening/middle/endgame`. |
| Peak RSS trên máy local | `results/resource_profile_20261121_v2.json` → `local_agents`; một ván/agent, sampling interval ghi trong JSON. |
| Run parameters, Colab environment | `results/colab/cnn_seed20261120_colab1/manifest.json`, `results/colab/rl_seed20261120_run1/manifest.json`, `results/colab/az_seed20261120_run1/manifest.json`. |
| Hash tập test và phiên bản solver | Các JSON solver và manifests; frozen SHA `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`, solver commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. |

Các nguồn chính thức/thăm dò và lý do loại trừ được liệt kê ở `docs/report_sources.md`.
'''
claims += "\n| Phase C validation curve and paired deltas | `results/phase_c_local/validation_curve_20261123.json` → `completed.*.metrics`, `paired_delta_vs_iteration_0`. |\n| Phase C frozen-test exact/WDL and paired AB3 deltas | `results/solver_eval_phase_c_local_20261123.json` → `metrics.*`, `paired_difference_vs_ab3`. |\n| Phase C round-robin W-D-L, score and CI | `results/round_robin_phase_c_20261123.json`; configuration/opening diagnostics from matching `.meta.json`. |\n"
claims = chr(10).join(line.rstrip() for line in claims.splitlines()) + chr(10)
(OUT / "claims_trace.md").write_text(claims, encoding="utf-8")

rubric = '''# Đối chiếu rubric đề bài

Các trọng số dưới đây theo mô tả trong yêu cầu người dùng; cần kiểm tra lại PDF gốc trước khi nộp vì tệp PDF hiện không trích xuất được.

| Tiêu chí (mô tả làm việc) | Trọng số | Vị trí báo cáo | Trạng thái/bằng chứng thiếu |
|---|---:|---|---|
| Bối cảnh, mục tiêu, câu hỏi nghiên cứu | 10 | Chương 1 | Có; xác minh cách diễn đạt theo PDF. |
| Cơ sở lý thuyết/công trình liên quan | 15 | Chương 2, tài liệu tham khảo | Có; đối chiếu yêu cầu trích dẫn cụ thể trong PDF. |
| Phương pháp, kiến trúc, các cấp tác tử | 20 | Chương 3, Phụ lục A | Có; làm rõ phần inspired và giới hạn. |
| Thiết kế thực nghiệm/đánh giá | 15 | Chương 3.5–3.6 | Có; một số chỉ số ablation còn thăm dò. |
| Kết quả và phân tích | 15 | Chương 4, hình/bảng sinh từ JSON | Có; RL/AZ ít seed, không có learning curve AZ. |
| Kết luận, hạn chế, hướng phát triển | 10 | Chương 5 | Có; không tuyên bố vượt bằng chứng. |
| Tái lập, kiểm thử, phụ lục | 10 | Phụ lục B, report_sources, claims_trace | Một số thông tin máy và rubric cần bổ sung. |
| Trình bày, hành chính, đóng góp nhóm | 5 | Bìa và định dạng | Thiếu thông tin thành viên/lớp/giảng viên; cần điền trước nộp. |

Tổng trọng số dự kiến: 100 điểm. Tên và mô tả chính thức của từng tiêu chí chưa được xác minh trực tiếp từ PDF.
'''
rubric = chr(10).join(line.rstrip() for line in rubric.splitlines()) + chr(10)
(OUT / "rubric_check.md").write_text(rubric, encoding="utf-8")

# Export HTML/PDF from the same Markdown. Mistune is part of the report tooling.
import mistune
html_body = mistune.create_markdown(plugins=["table", "strikethrough", "task_lists"])(report)
html_body = html_body.replace('src="assets/', f'src="{ASSETS.as_posix()}/')
html = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>
@page{size:A4;margin:22mm 20mm 20mm 25mm;@top-right{content:"Connect Four 7×6 · Báo cáo";color:#777;font-size:8pt}@bottom-center{content:"Trang " counter(page);color:#777;font-size:9pt}}
body{font:10.5pt/1.45 Arial,"DejaVu Sans",sans-serif;color:#20242a}h1{font-size:20pt;color:#163a5b;page-break-before:always}h1:first-child{page-break-before:avoid}h2{font-size:15pt;color:#235b7c;margin-top:20pt}h3{font-size:12pt;color:#285c73}table{border-collapse:collapse;width:100%;font-size:8pt;margin:10pt 0;page-break-inside:avoid}th,td{border:1px solid #8c98a3;padding:4px 5px}th{background:#e7eff5}img{max-width:100%;height:auto;display:block;margin:10pt auto}code{font-family:Consolas,monospace;font-size:9pt}pre{white-space:pre-wrap;background:#f4f5f6;padding:8px}li{margin:3pt 0}
</style></head><body>''' + html_body + "</body></html>"
(OUT / "report.html").write_text(html, encoding="utf-8")

# Build a minimal standards-compliant DOCX, preserving headings, tables and figures.
def xml_run(text: str, bold: bool = False) -> str:
    prop = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:r>{prop}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

def xml_paragraph(text: str, style: str = "Normal") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{xml_run(text)}</w:p>'

body_parts = []
image_files = []
image_relations = []
table_rows = []
for line in report.splitlines():
    if line.startswith("| "):
        if re.match(r"^\|[ :|\-]+\|$", line):
            continue
        table_rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        continue
    if table_rows:
        cols = max(map(len, table_rows))
        table_xml = '<w:tbl><w:tblPr><w:tblBorders>' + ''.join(f'<w:{edge} w:val="single" w:sz="4"/>' for edge in ("top","left","bottom","right","insideH","insideV")) + '</w:tblBorders></w:tblPr>'
        for row_i, row in enumerate(table_rows):
            table_xml += '<w:tr>'
            for cell in row + [""] * (cols-len(row)):
                table_xml += '<w:tc><w:tcPr><w:tcW w:w="1800" w:type="dxa"/></w:tcPr>' + xml_paragraph(cell, "TableText") + '</w:tc>'
            table_xml += '</w:tr>'
        body_parts.append(table_xml + '</w:tbl>')
        table_rows = []
    stripped = line.strip()
    if not stripped or stripped == "---":
        continue
    image_match = re.match(r"!\[[^]]*\]\(([^)]+)\)", stripped)
    if image_match:
        image_path = ROOT / "docs" / "report" / image_match.group(1)
        if image_path.exists():
            idx = len(image_files) + 1; rid = f"rIdImage{idx}"; name = f"image{idx}.png"
            image_files.append((name, image_path.read_bytes()))
            image_relations.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>')
            drawing = f'''<w:p><w:r><w:drawing><wp:inline><wp:extent cx="5700000" cy="3000000"/><wp:docPr id="{idx}" name="{name}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{idx}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="5700000" cy="3000000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
            body_parts.append(drawing)
        continue
    if stripped.startswith("# "):
        body_parts.append(xml_paragraph(stripped[2:], "Heading1"))
    elif stripped.startswith("## "):
        body_parts.append(xml_paragraph(stripped[3:], "Heading2"))
    elif stripped.startswith("### "):
        body_parts.append(xml_paragraph(stripped[4:], "Heading3"))
    elif stripped.startswith("|"):
        continue
    elif stripped.startswith("```"):
        continue
    elif stripped.startswith("[") or stripped.startswith("- ") or re.match(r"^\d+\. ", stripped):
        body_parts.append(xml_paragraph(re.sub(r"\*\*|`|\[|\]\([^)]*\)", "", stripped).lstrip("- "), "Normal"))
    else:
        body_parts.append(xml_paragraph(re.sub(r"\*\*|`", "", stripped)))
if table_rows:
    for row in table_rows:
        body_parts.append(xml_paragraph(" | ".join(row)))

styles = '''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="300"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'''
for level, size in ((1,34),(2,28),(3,24)):
    styles += f'<w:style w:type="paragraph" w:styleId="Heading{level}"><w:name w:val="Heading {level}"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="{level-1}"/></w:pPr><w:rPr><w:b/><w:color w:val="235B7C"/><w:sz w:val="{size}"/></w:rPr></w:style>'
styles += '<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:rPr><w:sz w:val="16"/></w:rPr></w:style></w:styles>'
document = f'''<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{''.join(body_parts)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="1134"/><w:headerReference w:type="default" r:id="rIdHeader"/><w:footerReference w:type="default" r:id="rIdFooter"/></w:sectPr></w:body></w:document>'''
ctypes = '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
root_rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
doc_rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdHeader" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer.xml"/>' + ''.join(image_relations) + '</Relationships>'
header = '''<?xml version="1.0" encoding="UTF-8"?><w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:color w:val="777777"/><w:sz w:val="16"/></w:rPr><w:t>Connect Four 7×6 · Báo cáo</w:t></w:r></w:p></w:hdr>'''
footer = '''<?xml version="1.0" encoding="UTF-8"?><w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'''
with zipfile.ZipFile(OUT / "report.docx", "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("[Content_Types].xml", ctypes); archive.writestr("_rels/.rels", root_rels)
    archive.writestr("word/document.xml", document); archive.writestr("word/styles.xml", styles)
    archive.writestr("word/_rels/document.xml.rels", doc_rels); archive.writestr("word/header.xml", header); archive.writestr("word/footer.xml", footer)
    for name, data in image_files: archive.writestr("word/media/" + name, data)

edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if Path(edge).exists():
    pdf = OUT / "report.pdf"
    result = subprocess.run([edge, "--headless", "--disable-gpu", "--no-sandbox", "--disable-extensions", f"--print-to-pdf={pdf}", (OUT / "report.html").resolve().as_uri()], capture_output=True, text=True, timeout=120)
    if result.returncode or not pdf.exists():
        raise RuntimeError("PDF export failed: " + result.stderr[-1000:])
else:
    raise RuntimeError("Edge headless was not found; Markdown, HTML and DOCX were generated, but PDF export is unavailable.")

print(f"Generated report.md, report.html, report.docx, report.pdf and {len(list(ASSETS.glob('*.png')))} PNG figures at 240 dpi.")
