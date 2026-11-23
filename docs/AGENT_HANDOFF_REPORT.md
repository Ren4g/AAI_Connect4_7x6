# Bàn giao ngữ cảnh viết báo cáo Connect Four 7×6

**Cập nhật:** 22-11-2026
**Mục đích:** cung cấp cho một agent khác bối cảnh, số liệu đã kiểm chứng và các giới hạn cần tuân thủ khi hoàn thiện báo cáo học thuật. Tài liệu này tóm tắt tình trạng mới nhất; một số báo cáo cũ trong repo chưa được cập nhật sau khi CNN v3 được ingest.

## 1. Yêu cầu đề tài và phạm vi

Đề tài xây dựng và đánh giá hệ thống Game AI cho Connect Four 7×6. Hệ thống hướng tới tám cấp tác tử, từ luật ngẫu nhiên đến học có giám sát, reinforcement learning và tìm kiếm PUCT/self-play. Trọng tâm báo cáo là mô tả thiết kế thực tế, đánh giá có kiểm soát và nêu rõ những thành phần nào chỉ là phiên bản *inspired*, không tuyên bố tái hiện đầy đủ AlphaGo hoặc AlphaZero.

Chỉ sử dụng số liệu đã có trong repo, JSON/JSONL, manifest và log. Không tự tạo kết quả, không dùng kết quả Colab chưa được ingest, không chọn checkpoint bằng tập test. Thông tin nhóm, sinh viên, giảng viên và phân công chưa có nguồn trong repo; để chỗ `[CẦN BỔ SUNG: ...]`.

## 2. Trạng thái mới nhất — cần ưu tiên hơn tài liệu cũ

- Các tác tử truyền thống cấp 1–4 đã được hiện thực và đánh giá: Random, Heuristic, AlphaBeta, MCTS-UCT.
- CNN có giám sát v2 và các nhánh RL/AZ lịch sử đã được ingest, đánh giá bằng solver và có trong các kết quả round-robin cũ.
- Đã ingest đủ **15 run CNN v3**. Các archive được huấn luyện từ source bundle v11; ingest được thực hiện bằng đúng snapshot đó sau khi kiểm tra hash, checksum dữ liệu/artifact, inference và loại trừ rò rỉ test. Không nới lỏng kiểm tra commit.
- Chọn cấu hình CNN v3 bằng validation: **Pons-soft, 100% dữ liệu**, ba seed. Chọn checkpoint warm-start theo validation loss trong cấu hình này: seed `20261122`, checkpoint `checkpoints/cnn_v3_30k_best.pt`.
- CNN v3 đã được đánh giá trên tập solver test đóng băng. Báo cáo kết quả: `docs/CNN_V3_RESULTS.md`; JSON: `results/solver_eval_cnn_v3_soft_3seeds_20261122.json`.
- **Phase C (AlphaZero-inspired V2) chưa hoàn tất.** Cấu hình đã chuẩn bị trong `colab/03_selfplay_az.ipynb`, nhưng các run V2 warm-start và scratch chưa có kết quả hoàn chỉnh được ingest/đánh giá. Theo thông tin người dùng, Phase C chưa chạy xong.
- Do Phase C chưa hoàn tất, **chưa thể chạy round-robin cuối bao gồm CNN v3 và các checkpoint Phase C**. Round-robin trong repo là kết quả của một mốc trước, không được trình bày như bảng cuối có các mô hình mới.

## 3. Kiến trúc và phương pháp

| Cấp | Tác tử | Tình trạng / mô tả cần dùng |
|---:|---|---|
| 1 | Random | Baseline chọn nước hợp lệ ngẫu nhiên. |
| 2 | Heuristic | Đánh giá mẫu liên tiếp và đe dọa chiến thuật; có ưu tiên trung tâm. |
| 3 | AlphaBeta | Minimax với alpha-beta pruning, iterative deepening và sắp xếp nước đi; có cấu hình độ sâu/ngân sách. |
| 4 | MCTS-UCT | Monte Carlo Tree Search với exploration bonus; round-robin lịch sử dùng 500 simulations/nước. |
| 5 | CNN supervised | Policy/value CNN học nhãn Pons. Khi chơi policy-only, policy head chọn nước; phải tách kết quả mạng thuần và safety filter. Value head được đo riêng. |
| 6 | Policy-gradient RL | REINFORCE/actor-critic nhẹ, lịch sử huấn luyện chủ yếu đấu Random; chỉ một run/seed. |
| 7 | AlphaGo-inspired PUCT | Policy cung cấp prior và value đánh giá nút lá cho tìm kiếm PUCT. Đây là kiến trúc lấy cảm hứng, không có rollout network riêng hay vòng RL AlphaGo đầy đủ. |
| 8 | AlphaZero-inspired | Self-play PUCT, policy/value update, replay, Dirichlet noise và nhiệt độ. Run lịch sử đã hoàn thành; Phase C V2 warm-vs-scratch là thí nghiệm mới, chưa xong. |

Engine grid và bitboard cùng mô hình hóa bàn 7×6; Pons là solver ngoài dùng làm thước đo exact, không phải tác tử thi đấu trong bảng xếp hạng. Encoder CNN dùng hai mặt phẳng theo góc nhìn người đến lượt và đối thủ. Mã chính trong `connect4/`; mô tả lệnh và pipeline ở `README.md`, `docs/COLAB.md`.

## 4. Giao thức đánh giá và thuật ngữ

- Tập test solver đóng băng: **1.500 vị trí**, 500 mỗi giai đoạn (ply 7–10, 11–24, 25–41); SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`; Pons commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`.
- Tỷ lệ nước tối ưu exact: agent chọn một nước có điểm solver tối đa. Tỷ lệ giữ WDL: agent giữ nguyên dấu kết quả thắng/hòa/thua so với nước tốt nhất.
- Mọi kết quả đối đầu dùng **score = (thắng + 0,5 × hòa) / số ván**; không gọi score là win rate.
- Khoảng tin cậy solver là bootstrap 95% theo cụm ván nguồn; hiệu giữa hai agent là bootstrap ghép cặp trên cùng vị trí/cụm. Round-robin dùng opening ghép cặp và CI bootstrap theo opening. Không coi các vị trí cùng ván là quan sát độc lập.
- CI hiệu chứa 0 (hoặc CI score chứa 0,5) nghĩa là **chưa phân biệt được**.
- Thời gian suy luận/throughput chỉ so sánh khi đo trên cùng máy local. Không dùng tốc độ Colab giữa các runtime khác nhau làm so sánh agent.
- Safety filter chiến thuật có thể đóng góp nhiều; với agent mạng phải báo riêng raw policy và policy có filter khi có kết quả.

## 5. Kết quả solver đã có

### 5.1 Baseline trước CNN v3

Nguồn: `results/solver_eval_cnn_v2_3seeds.json`, được trình bày trong `docs/LEARNING_RESULTS.md` và `docs/report.md` hiện có.

| Agent | Nước tối ưu exact / 1.500 | CI 95% |
|---|---:|---:|
| Random | 35,9% | [33,4; 38,3]% |
| Heuristic | 73,5% | [71,4; 75,7]% |
| AlphaBeta depth 3 | 77,2% | [75,0; 79,4]% |
| AlphaBeta depth 4 | 80,5% | [78,5; 82,4]% |
| AlphaBeta depth 5 | 83,1% | [81,2; 84,9]% |
| AlphaBeta depth 6 | 81,5% | [79,5; 83,4]% |
| MCTS-500 | 76,3% | [74,2; 78,4]% |
| CNN v2, ba seed | 70,7%; 70,9%; 71,2% | CI từng seed trong `docs/LEARNING_RESULTS.md` |
| RL policy (filtered) | 56,9% | [54,3; 59,4]% |
| PUCT-AZ lịch sử, 500 simulations | 72,1% | [69,7; 74,3]% |

Các con số này là baseline của các run tương ứng, không gộp với lượt đánh giá khác có seed/configuration khác. CNN v2 đều thấp hơn AlphaBeta depth 3 theo hiệu ghép cặp; cổng tối thiểu đã định trước chưa đạt.

### 5.2 CNN v3 — validation và frozen test

Nguồn selection: `results/cnn_v3_validation_selection_20261122.json`; báo cáo: `docs/CNN_V3_SELECTION.md`. Pons-soft 100% có mean validation exact-optimal **80,09% ± 0,29 điểm phần trăm SD giữa ba seed**, 4.530 vị trí/seed. Đây là SD giữa seed, không phải CI bootstrap. Chọn checkpoint trong mỗi run theo validation loss; seed 20261122 cho warm-start, validation exact 80,40%, loss 1,4617. Hash checkpoint: `3d86468ab7586ef502e8b1a894909b722506192b88fcc8a9937b814fb96ff35b`.

Nguồn test: `results/solver_eval_cnn_v3_soft_3seeds_20261122.json`; 1.500 vị trí, bootstrap 10.000 lần theo source game. Kết quả exact và hiệu ghép cặp với AlphaBeta depth 3:

| Seed | CNN raw exact (CI 95%) | CNN + safety filter exact (CI 95%) | Hiệu filtered − AB3, CI ghép cặp |
|---:|---:|---:|---:|
| 20261121 | 66,7% [64,1; 69,2] | 76,3% [74,2; 78,5] | [-3,4; 1,6] điểm % |
| 20261122 | 68,6% [66,2; 71,1] | 76,7% [74,6; 78,9] | [-3,0; 2,1] điểm % |
| 20261123 | 65,5% [63,0; 68,0] | 75,7% [73,5; 77,9] | [-4,0; 1,1] điểm % |

Trung bình filtered giữa seed là 76,3%, SD 0,50 điểm phần trăm; WDL filtered trung bình 90,9%, SD 0,32 điểm phần trăm. Cả ba CI hiệu với AB3 đều chứa 0: **chưa phân biệt được với AB3, và không đạt tiêu chí cổng yêu cầu cận dưới không âm**. Safety filter tăng kết quả so với raw; không quy mức filtered cho mạng đơn lẻ. Đây là một lần đánh giá trên test đã tái sử dụng cho baseline, không phải tập test mới độc lập và không được dùng lại để chọn mô hình.

## 6. Round-robin hiện có và điều chưa thể kết luận

`docs/ROUND_ROBIN_FINAL.md` cùng `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.*` là round-robin **mốc cũ**: 100 ván/cặp, 50 opening ghép cặp, hai ghế, fair filter; gồm Random, Heuristic, AlphaBeta depth 4/6, MCTS-500, CNN v2 policy, RL-Policy và PUCT-AZ lịch sử. Ví dụ: AB4–MCTS-500 score 0,590 [0,520; 0,665]; CNN v2–PUCT score 0,235 [0,165; 0,305]. Xem file nguồn cho toàn bộ W-D-L và CI.

Không gắn nhãn bảng này là giải đấu cuối có CNN v3 hoặc AlphaZero V2. Sau khi Phase C hoàn tất và checkpoint được ingest/đánh giá, mới chạy round-robin cập nhật trên bộ opening chung, hai ghế, tối thiểu 100 ván/cặp, đồng thời giữ lại bảng cũ làm baseline lịch sử. Hiện chưa có kết quả đó.

## 7. Lịch sử học và chẩn đoán

- CNN v2: 1.500 vị trí (1.284 train, 216 validation), nhãn solver Pons; batch 64, tối đa 30 epoch, dừng 609–630 update vì khoảng 21 batch/epoch.
- CNN v3: 30.000 nhãn Pons exact, 10.000 mỗi giai đoạn; ba seed cho từng cấu hình đã chọn cùng các nhánh ablation (Pons soft/hard, 10/30/100%, AB3 teacher). Dataset và source-game/mirror hashes được kiểm tra ở ingest. Chi tiết cấu hình/manifest nằm trong `results/colab/cnn_v3_30k_*/` và `colab/data/`.
- AlphaZero lịch sử: 100 iteration, 32 ván/iteration, 100 simulations/nước, một seed và một lần resume. Một run không đủ để kết luận phương pháp; đường loss/reward không thay cho đánh giá độc lập.
- RL lịch sử: 10.000 episode, đối thủ Random; reward trung bình train không thay cho sức chơi. Chỉ một seed.
- Chẩn đoán CNN sai nước đầu tiên: `results/cnn_first_errors_seed20261122.json`, diễn giải trong `docs/CNN_LOSS_ANALYSIS.md`; chỉ áp dụng tập ván được phân tích, không khái quát mọi ván.
- Value head lịch sử: `results/value_head/` và `docs/LEARNING_RESULTS.md`; bằng chứng chưa đủ để nói value head vượt baseline một cách chắc chắn.

## 8. Phase C hiện hành — chưa có kết quả

Notebook chuẩn là `colab/03_selfplay_az.ipynb`, dùng source bundle `colab/source_bundle_20261121_v12.zip`. Không dùng `colab/06_selfplay_az_v2.(outdated).ipynb`. Run đăng ký trước:

1. Warm: `az_v2_warm_seed20261121`, seed 20261121, khởi tạo từ checkpoint CNN v3 được chọn theo validation.
2. Scratch: `az_v2_scratch_seed20261122`, seed 20261122, khởi tạo trọng số ngẫu nhiên.

Cùng cấu hình cho mỗi run: 400 simulations/nước, tối đa 40 iteration, 32 ván/iteration, 16 ván song song, 200 cập nhật/iteration, batch size 128, snapshot iteration 0/5/10/15/20/25/30/35/40. Tối đa 60 phút GPU/run và 120 phút toàn Phase C; mỗi segment tối đa 30 phút; resume giữ nguyên run ID/config. Hai run là so sánh trường hợp có kiểm soát, mỗi nhánh chỉ một seed, không đủ cho khẳng định tổng quát giữa phương pháp.

Đánh giá Phase C dự kiến: phân tích đường cong validation so với snapshot iteration 0 dưới cùng PUCT-400, loại replay overlap theo tiêu chí đã ghi; chỉ sau khi hoàn tất hai run/validation mới chấm checkpoint iteration 40 trên frozen test một lần. Sau đó mới chạy round-robin cập nhật. Không báo bất kỳ kết quả Phase C nào trước khi nhận, ingest và kiểm tra artifact.

## 9. Các bẫy đánh giá đã phát hiện

Trong các lần đánh giá ban đầu, nhóm đã phát hiện: agent tất định làm lặp cùng ván dù tăng số trận; thời gian bị gộp hai phía; `nodes` là tổng qua nhiều lượt/ván thay vì node/nước; MCTS có vấn đề thang giá trị; opening có thể đã phân thắng/thua; `budget_violations` không gắn agent. Các giao thức về sau dùng opening chung/đổi ghế, replay, thống kê tách agent, bootstrap đúng đơn vị cụm và fair opening filter. Chỉ mô tả ảnh hưởng cụ thể khi có nguồn JSON/LOG phù hợp; không nói mọi số cũ đều đã được sửa nếu không có chứng cứ.

## 10. Nguồn nên đọc và cảnh báo tài liệu stale

1. **Trạng thái mới nhất CNN v3:** `docs/CNN_V3_RESULTS.md`, `docs/CNN_V3_SELECTION.md`, JSON tương ứng, manifest trong `results/colab/cnn_v3_30k_*/`.
2. **Phase C và giới hạn Colab:** `docs/CRITERIA_V2.md`, `docs/CRITERIA_V2_ADDENDUM_20261121.md`, `docs/COLAB.md`, notebook `colab/03_selfplay_az.ipynb`.
3. **Baseline đã có:** `docs/LEARNING_RESULTS.md`, `docs/ROUND_ROBIN_FINAL.md`, `docs/RESOURCE_PROFILE.md`, cùng JSON/JSONL/meta nêu trong các file đó.
4. `docs/PROGRESS_REPORT.md`, `docs/report.md`, `docs/report_sources.md` và `docs/report_open_issues.md` được tạo trước kết quả CNN v3 hoặc chưa cập nhật tình trạng Phase C hiện nay. Đọc chúng làm lịch sử/khung, nhưng phải đối chiếu tài liệu và JSON mới hơn. Một số file cũ có dấu hiệu lỗi mã hóa tiếng Việt; không sao chép phần chữ hỏng vào báo cáo mới.
5. PDF đề bài ngoài repo chưa được xác minh trong môi trường này. Đối chiếu lại rubric và thông tin trích dẫn trước khi nộp; không bịa thông tin thư mục.

## 11. Chỉ dẫn cho agent viết báo cáo

Hãy viết báo cáo tiếng Việt học thuật, câu rõ và trung tính; phân biệt sức mạnh thực nghiệm với tối ưu exact đã chứng minh; mỗi số cần đường dẫn file nguồn, cỡ mẫu và CI thích hợp. Giữ riêng kết quả baseline lịch sử, CNN v3 mới và Phase C chưa hoàn thành. Nêu công khai rằng CNN v3 chưa đạt cổng AB3, Phase C chưa xong, round-robin cập nhật chưa chạy. Không gọi CNN v3 là đã cải thiện có ý nghĩa nếu chỉ đối chiếu mean validation; không dùng frozen test để chọn mô hình; không gộp safety-filter với năng lực mạng; không suy rộng một seed RL/AZ thành kết luận phương pháp.

Chỗ thiếu (thông tin nhóm, đối chiếu rubric PDF, citation chưa kiểm chứng) phải đánh dấu `[CẦN BỔ SUNG]` hoặc `[CẦN KIỂM TRA]`. Kết quả Phase C và round-robin cuối phải để trạng thái “chưa có” cho đến khi artifact tương ứng được ingest và đánh giá.
