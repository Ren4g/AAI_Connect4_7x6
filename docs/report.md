# AI cho Connect Four 7×6: hiện thực, huấn luyện và đánh giá có kiểm soát

> **Bản này đã được thay thế.** Báo cáo cập nhật, sinh từ kết quả chính thức Phase C v2, nằm tại [docs/report/report.md](report/report.md); bản xuất là [Word](report/report.docx) và [PDF](report/report.pdf). Giữ nội dung bên dưới làm snapshot lịch sử, không dùng làm báo cáo cuối.

> Báo cáo chính được sinh từ kết quả JSON đã kiểm tra. Tái tạo bằng `python scripts/render_final_report.py`; các bảng dẫn xuất được sinh lại từ JSON nguồn, không nhập tay.

## Tóm tắt

Đề tài xây dựng hệ thống chơi Connect Four 7×6 theo tám cấp độ, từ tác tử ngẫu nhiên đến tìm kiếm kết hợp policy/value và self-play. Đánh giá dùng bộ test đóng băng 1,500 vị trí đã chấm bằng solver Pons, cùng round-robin ghép cặp trên 8 agent. AlphaBeta depth 4 đạt tỷ lệ chọn nước tối ưu 80.5%; CNN supervised đạt trung bình 70.9% (SD giữa ba seed 0.3 điểm %), còn PUCT-AZ 500 mô phỏng đạt 72.1%. Không CNN nào đạt cổng không-kém-hơn AlphaBeta depth 3; checkpoint self-play cũng không qua cổng thăng hạng trong arena. Kết quả thể hiện rằng pipeline và các cấp agent đã được hiện thực, trong khi các mô hình học được chưa vượt baseline tìm kiếm theo tiêu chí định trước.

## 1. Bài toán và mục tiêu

Connect Four là trò chơi hai người, tổng bằng không, thông tin hoàn hảo trên lưới 7 cột × 6 hàng. Mỗi lượt thả quân vào một cột còn chỗ; thắng khi tạo bốn quân liên tiếp theo hàng, cột hoặc đường chéo. Không gian trạng thái hữu hạn nhưng nhánh hành động và tương tác chiến thuật khiến luật chơi đơn giản vẫn phù hợp để khảo sát tìm kiếm, học có giám sát và self-play.

Mục tiêu của đề tài là (i) triển khai các agent có độ phức tạp tăng dần trên cùng luật chơi; (ii) xây dựng giao thức so sánh tái lập được; (iii) dùng solver exact làm thước đo chất lượng nước đi; và (iv) chuẩn bị pipeline huấn luyện có thể resume trên Colab, nhưng kiểm định artifact và đánh giá cuối ở local.

## 2. Kiến trúc và phương pháp

Engine cung cấp API trạng thái thống nhất với hai biểu diễn bàn cờ: grid và bitboard. Bitboard dùng trong tìm kiếm/giải đấu nhờ thao tác chiến thuật nhanh; encoder CNN tạo hai mặt phẳng quân của người đến lượt và đối thủ, được dùng chung khi huấn luyện và suy luận. Solver Pons được gọi qua adapter, nhận chuỗi cột hợp lệ và trả điểm exact cho từng nước đi.

| Cấp | Phương pháp | Vai trò trong nghiên cứu |
|---:|---|---|
| 1 | Random | Baseline tối thiểu và đối chứng cho pipeline. |
| 2 | Heuristic | Chấm các mẫu bốn ô, ưu tiên trung tâm và đe dọa/chặn chiến thuật. |
| 3 | AlphaBeta | Minimax với alpha-beta pruning, iterative deepening, move ordering và giới hạn thời gian tùy preset. |
| 4 | MCTS-UCT | Chọn nước theo thống kê thăm cây và exploration bonus; cấu hình so sánh dùng 500 mô phỏng/nước. |
| 5 | CNN supervised | Policy/value CNN học nhãn solver; trong PolicyAgent hiện tại, policy được dùng trực tiếp, còn value được đánh giá riêng. |
| 6 | Policy-gradient RL | REINFORCE/self-play đối kháng Random, có safety filter khi triển khai policy. |
| 7 | AlphaGo-inspired PUCT | Policy tạo prior, value ước lượng lá; dùng PUCT để phân bổ tìm kiếm. Không có mạng rollout riêng. |
| 8 | AlphaZero-inspired | Self-play PUCT với nhiễu Dirichlet, replay buffer và cập nhật policy/value; vòng arena quyết định promote checkpoint. |

Các cấp 5–8 có CLI và checkpoint/resume. Job Colab chỉ thực hiện huấn luyện nặng; notebook là lớp gọi CLI. Artifact chỉ được dùng sau khi ingest local kiểm tra manifest, checksum, inference và loại trừ trùng lặp với test đóng băng.

## 3. Thiết kế thực nghiệm

### 3.1 Dữ liệu và giao thức solver

Bộ test đóng băng có 1,500 vị trí từ các ván nguồn; SHA256 vị trí `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01` và commit solver `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Tỷ lệ nước tối ưu là phần vị trí agent chọn một nước có điểm exact cao nhất. Giữ W/D/L đo việc nước đi bảo toàn cùng lớp kết cục lý thuyết so với nước tốt nhất. Khoảng tin cậy 95% được bootstrap theo cụm ván nguồn, 10,000 lần lặp; các hiệu so với AlphaBeta depth 3 là ghép cặp trên cùng vị trí/cụm.

### 3.2 Round-robin và arena

Round-robin dùng 100 ván mỗi cặp trên 50 khai cuộc chung, mỗi khai cuộc đánh cả hai ghế; bộ lọc fair loại 6 khai cuộc có thắng tức thì trong 56 ứng viên. Score = (thắng + 0,5 hòa)/số ván; CI ghép cặp bootstrap theo khai cuộc. Có 28 cặp độc lập; JSONL lưu replay nước đi. Arena AlphaZero dùng 100 khai cuộc ghép cặp và ngưỡng promote đặt trước yêu cầu cận dưới CI score > 0,5.

### 3.3 Tiêu chí chuyển giai đoạn

Cổng tối thiểu CNN được đặt trước khi huấn luyện: cận dưới CI bootstrap ghép cặp của hiệu tỷ lệ nước tối ưu (CNN − AlphaBeta depth 3) không âm trên test. Mục tiêu 85% nước tối ưu là mục tiêu, không phải cổng chặn. Chọn checkpoint theo validation; test đóng băng không dùng để chọn mô hình.

## 4. Kết quả

### 4.1 So sánh với solver exact

| Agent | Nước tối ưu / n | Tỷ lệ nước tối ưu, CI 95% | Giữ W/D/L, CI 95% |
|---|---:|---:|---:|
| Random | 538/1500 | 35.9% [33.4; 38.3] | 71.1% [68.7; 73.4] |
| Heuristic | 1103/1500 | 73.5% [71.4; 75.7] | 88.3% [86.6; 89.8] |
| AlphaBeta-depth3 | 1158/1500 | 77.2% [75.0; 79.4] | 90.5% [89.0; 92.0] |
| AlphaBeta-depth4 | 1207/1500 | 80.5% [78.5; 82.4] | 91.7% [90.3; 93.1] |
| AlphaBeta-depth5 | 1246/1500 | 83.1% [81.2; 84.9] | 93.3% [92.0; 94.5] |
| AlphaBeta-depth6 | 1222/1500 | 81.5% [79.5; 83.4] | 91.5% [90.0; 92.9] |
| MCTS-500 | 1145/1500 | 76.3% [74.2; 78.4] | 91.5% [90.1; 92.9] |
| CNN supervised seed 20261120 | 1060/1500 | 70.7% [68.3; 72.9] | 87.8% [86.1; 89.4] |
| CNN supervised seed 20261121 | 1064/1500 | 70.9% [68.6; 73.2] | 88.0% [86.4; 89.6] |
| CNN supervised seed 20261122 | 1068/1500 | 71.2% [68.8; 73.6] | 89.1% [87.5; 90.7] |
| RL policy (safety-filtered) | 853/1500 | 56.9% [54.3; 59.4] | 79.7% [77.7; 81.6] |
| PUCT-AZ, 500 simulations | 1081/1500 | 72.1% [69.7; 74.3] | 91.7% [90.1; 93.1] |

Trung bình ba seed CNN là 70.9% (độ lệch chuẩn giữa seed 0.3 điểm phần trăm). Các khoảng tin cậy ghép cặp CNN − AB3 đều nằm dưới 0 nên không seed nào đạt cổng tối thiểu. PUCT-AZ đạt 72.1%, thấp hơn AB3 (77.2%); RL policy còn thấp hơn trên test solver. Kết luận chỉ áp dụng cho checkpoint và giao thức đã đo, không suy rộng sang mọi seed.

### 4.1.1 Phân tầng theo giai đoạn

Bảng dưới dùng checkpoint CNN seed 20261121 được chọn bằng validation; mỗi ô là tỷ lệ nước tối ưu và CI bootstrap 95% trên 500 vị trí của giai đoạn đó.

| Agent | Khai cuộc | Trung cuộc | Tàn cuộc |
|---|---:|---:|---:|
| AlphaBeta depth 3 | 65.4% [61.2; 69.5] | 73.0% [68.9; 76.9] | 93.2% [91.0; 95.3] |
| CNN seed 20261121 | 63.6% [59.2; 68.0] | 64.0% [59.9; 68.1] | 85.2% [82.0; 88.3] |
| RL policy | 35.6% [31.3; 39.8] | 53.4% [49.0; 57.8] | 81.6% [78.0; 85.1] |
| PUCT-AZ 500 | 67.8% [63.8; 71.8] | 65.0% [60.8; 69.2] | 83.4% [79.9; 86.7] |

### 4.1.2 Cổng CNN và hiệu ghép cặp

| Checkpoint | Hiệu CNN − AB3, CI 95% (điểm %) | Kết quả |
|---|---:|---|
| CNN seed 20261120 | [-9.1; -4.0] | Chưa đạt |
| CNN seed 20261121 | [-8.8; -3.7] | Chưa đạt |
| CNN seed 20261122 | [-8.6; -3.4] | Chưa đạt |
| RL policy | [-22.9; -17.8] | Tham chiếu; dưới AB3 |
| PUCT-AZ 500 | [-7.8; -2.5] | Tham chiếu; dưới AB3 |

### 4.2 Giải đấu trực tiếp

| Cặp (A vs B) | W-D-L của A | Score A | CI bootstrap 95% |
|---|---:|---:|---:|
| AlphaBeta-depth4 vs MCTS-500 | 56-6-38 | 0.590 | [52.0; 66.5] |
| AlphaBeta-depth4 vs PUCT | 45-9-46 | 0.495 | [40.5; 58.5] |
| AlphaBeta-depth4 vs CNN-Policy | 74-7-19 | 0.775 | [69.0; 85.0] |
| AlphaBeta-depth6 vs PUCT | 57-6-37 | 0.600 | [50.0; 70.0] |
| Heuristic vs CNN-Policy | 49-11-40 | 0.545 | [46.5; 63.0] |
| Heuristic vs RL-Policy | 83-4-13 | 0.850 | [77.5; 91.5] |
| RL-Policy vs PUCT | 6-2-92 | 0.070 | [2.5; 13.0] |
| CNN-Policy vs PUCT | 16-15-69 | 0.235 | [16.5; 30.5] |

AlphaBeta depth 4 đạt score 0.590 trước MCTS-500; score trước PUCT gần 0,5 và CI chứa 0,5. CI của CNN và Heuristic còn chồng lấp; CNN thua hai mức AlphaBeta trong cặp đã đo. RL thua rõ các agent tìm kiếm. AZ iteration 100 đạt score 0.535 trước PUCT khởi tạo, CI [47.5; 59.2]; cận dưới không vượt ngưỡng nên checkpoint không được promote.

### 4.3 Value head, chi phí và tài nguyên

Checkpoint CNN được chọn có độ chính xác W/D/L của value head 53.5% (CI theo cụm [50.6; 56.5], so với baseline lớp đa số 52.1%. CI chứa baseline nên một checkpoint này chưa chứng minh được lợi thế độc lập của value head. Thời gian, node/mô phỏng và bộ nhớ local được báo cáo riêng trong [round-robin](ROUND_ROBIN_FINAL.md) và [đo tài nguyên](RESOURCE_PROFILE.md).

## 5. Phân tích và các bẫy đánh giá đã xử lý

Các lượt đầu của dự án cho thấy kết quả có thể bị sai lệch bởi ván lặp từ agent tất định, gộp thời gian của hai phía, nhầm tổng node với node/nước, thang điểm MCTS và khai cuộc đã có kết cục chiến thuật. Giao thức cuối dùng khai cuộc chung, đổi ghế, ghi replay JSONL, phân tách chi phí theo agent và kiểm tra fair opening trước giải đấu.

Kết quả CNN tập trung yếu ở khai cuộc/trung cuộc hơn tàn cuộc; điều này không tự chứng minh nguyên nhân là thiếu dữ liệu hoặc không thấy fork. Báo cáo giữ riêng tỷ lệ nước tối ưu solver, giữ W/D/L và score đối đầu. Không diễn giải kết quả training-time đấu Random thành sức mạnh độc lập.

### 5.1 Sai lầm đầu tiên trong các ván CNN thua

Phân tích solver trên 194 ván thua của CNN tìm được ít nhất một nước sai điểm exact ở 181 ván; 145 ván có nước đầu tiên làm giảm W/D/L. Trong nhóm nước WDL-degrading đầu tiên, 14 được phân loại là nhượng fork theo kiểm tra hai ply, còn 131 là lỗi sâu hơn hoặc định vị. Solver xác định 97 ván đã là thế thua cưỡng bức cho CNN ngay sau opening. Vì vậy dữ liệu không ủng hộ giả thuyết CNN chủ yếu thua do bỏ fork; phần lớn lỗi WDL đầu tiên là positional hoặc sâu hơn.

Chi tiết từng ván và định nghĩa phân loại: [CNN_LOSS_ANALYSIS.md](CNN_LOSS_ANALYSIS.md).

## 6. Hạn chế và khả năng tái lập

- Solver test gồm 1,500 vị trí có quan hệ theo ván nguồn; bootstrap theo cụm giảm giả định độc lập nhưng không tạo thêm các ván độc lập mới.
- CNN có ba seed; RL và AlphaZero mỗi nhánh chỉ có một run/checkpoint. Kết luận RL/AZ vì vậy chỉ áp dụng cho cấu hình đã chạy.
- Hai manifest Colab lịch sử của RL và AlphaZero không ghi peak VRAM. Pipeline mới đã ghi trường này và tích lũy cực đại qua resume, nhưng không thể khôi phục peak quá khứ từ checkpoint.
- Round-robin được chạy từ working tree có thay đổi cục bộ; metadata lưu checkpoint hashes, kết quả/replay hashes và ghi chú khác biệt mã nguồn. Commit nền: `5423c266894ee910d1b833d95aefe23a85f41613`.
- Thời gian local phụ thuộc phần cứng cụ thể; Colab T4 và tài nguyên phiên thay đổi, nên không dùng throughput Colab để xếp hạng agent.
- GPU training/simulation có thể không tái sinh bit-for-bit giữa phần cứng/phiên; log JSONL cho phép phát lại ván đã lưu, không tái tạo toàn bộ tiến trình stochastic.

Mỗi run Colab có manifest, checkpoint, log và checksum; dữ liệu đầu vào xác minh frozen-test hash và loại trừ trạng thái test, kể cả gương. Script ingest local là cổng bắt buộc trước đánh giá. Kết quả chính có file JSON nguồn và report renderer để tái tạo bảng.

## 7. Kết luận

Dự án đã triển khai hệ thống tám cấp agent, quy trình huấn luyện có thể resume và đánh giá được kiểm soát bằng solver exact cùng round-robin ghép cặp. Tìm kiếm AlphaBeta vẫn là baseline mạnh nhất trong các cấu hình được báo cáo; CNN supervised chưa vượt cổng định trước, RL policy yếu trên test solver, và AlphaZero chưa được promote sau arena. Giá trị chính của kết quả là pipeline tái lập và các phép đo có giới hạn được nêu rõ, không phải khẳng định rằng các agent học đã đạt sức chơi vượt tìm kiếm.

## Tệp nguồn và tái tạo

- CNN solver JSON `results/solver_eval_cnn_v2_3seeds.json` (SHA256 `c4936eabf0ef5e666fb9af37f3ebb503cb517ebab7af3613615327c4a83cbe86`).
- Solver JSON cho RL/AZ và baseline `results/solver_eval_az_rl_20261121_seed20261121.json` (SHA256 `c1505fcf0af907ff35c85f50ccb39f42eecf475479e849f95bce162cfd11a614`).
- Round-robin summary/meta/replay: `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json`, `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json`; seed 20261121, 100 games/pair.
- Arena `benchmarks/arena/arena_az_vs_init_100openings_seed20261121.json`; value-head JSON `results/value_head/value_cnn_seed20261121_20261121.json`; resource profile `results/resource_profile_20261121_v2.json`.
- Phân tích nước sai đầu tiên: `results/cnn_first_errors_seed20261122.json` (SHA256 `d0d801ebe5b002ad5c29e72b70a14aa3b4edde2f5b9067149cadafb3f40d4e00`).
- Báo cáo chi tiết sinh tự động: `python scripts/render_learning_report.py` và `python scripts/render_round_robin_report.py`.
