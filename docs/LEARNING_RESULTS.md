# Kết quả huấn luyện và đánh giá CNN, RL, AlphaZero

> Bản này được sinh từ các manifest và JSON kết quả; chạy lại `python scripts/render_learning_report.py` để tái tạo.

## Thiết kế đánh giá

Các phép đo dùng cùng tập test Pons đóng băng gồm 1,500 vị trí, SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`, cùng solver `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Khoảng tin cậy 95% bootstrap theo ván nguồn, 10,000 lần lặp; chênh lệch với AlphaBeta depth 3 ghép cặp trên cùng vị trí/ván nguồn. Thời gian suy luận đo local và không dùng để so sánh tài nguyên Colab.

## Run Colab đã ingest

| Nhánh | Run | Bước/iteration | Resume | GPU | Cấu hình / ghi chú |
|---|---|---:|---:|---|---|
| CNN supervised | `cnn_seed20261120_colab1` | 609 bước | không | Tesla T4 | 30 epoch; seed 20261120 |
| CNN supervised | `cnn_seed20261121_colab1` | 630 bước | không | Tesla T4 | 30 epoch; seed 20261121 |
| CNN supervised | `cnn_seed20261122_colab1` | 630 bước | không | Tesla T4 | 30 epoch; seed 20261122 |
| AlphaZero self-play | `az_seed20261120_run1` | 100 iteration / 20,000 update, 3200 game | có (1 lần resume) | Tesla T4 | 32 games/iteration, 100 simulations/nước; khởi tạo từ CNN seed 20261121 |
| Policy gradient RL | `rl_seed20261120_run1` | 10,000 episode | không | Tesla T4 | policy-gradient đấu Random; seed 20261120 |

## Năng lực chọn nước theo solver

Nước tối ưu là chọn một nước có điểm exact cao nhất theo solver; giữ W/D/L là chọn nước cùng dấu với kết quả tốt nhất. Các CI trong bảng là phần trăm.

| Agent | Đúng tối ưu / n | Tỷ lệ nước tối ưu, CI 95% | Giữ W/D/L, CI 95% |
|---|---:|---:|---:|
| Random | 538/1500 | 35.9% [33.4; 38.3] | 71.1% [68.7; 73.4] |
| Heuristic | 1103/1500 | 73.5% [71.4; 75.7] | 88.3% [86.6; 89.8] |
| AlphaBeta-depth3 | 1158/1500 | 77.2% [75.0; 79.4] | 90.5% [89.0; 92.0] |
| AlphaBeta-depth4 | 1207/1500 | 80.5% [78.5; 82.4] | 91.7% [90.3; 93.1] |
| AlphaBeta-depth5 | 1246/1500 | 83.1% [81.2; 84.9] | 93.3% [92.0; 94.5] |
| AlphaBeta-depth6 | 1222/1500 | 81.5% [79.5; 83.4] | 91.5% [90.0; 92.9] |
| MCTS-500 | 1145/1500 | 76.3% [74.2; 78.4] | 91.5% [90.1; 92.9] |
| CNN-CNN_v2_s20261120-policy | 1060/1500 | 70.7% [68.3; 72.9] | 87.8% [86.1; 89.4] |
| CNN-CNN_v2_s20261121-policy | 1064/1500 | 70.9% [68.6; 73.2] | 88.0% [86.4; 89.6] |
| CNN-CNN_v2_s20261122-policy | 1068/1500 | 71.2% [68.8; 73.6] | 89.1% [87.5; 90.7] |
| CNN-RL-20261120-policy | 853/1500 | 56.9% [54.3; 59.4] | 79.7% [77.7; 81.6] |
| PUCT-AZ-20261120 (500 simulations) | 1081/1500 | 72.1% [69.7; 74.3] | 91.7% [90.1; 93.1] |

Ba seed CNN đạt trung bình mô tả 70.9% (độ lệch chuẩn giữa seed 0.3 điểm phần trăm); đây là độ biến thiên giữa ba seed, không phải CI bootstrap gộp.
Checkpoint CNN dùng trong round-robin là `cnn_seed20261121_colab1` (seed 20261121), được chọn theo validation loss thấp nhất giữa các checkpoint-best; exact-optimal top-1 ở epoch được lưu là 55.1%.

### Theo giai đoạn

| Agent | Khai cuộc (n=500) | Trung cuộc (n=500) | Tàn cuộc (n=500) |
|---|---:|---:|---:|
| CNN-CNN_v2_s20261120-policy | 62.2% [58.0; 66.4] | 65.2% [61.0; 69.4] | 84.6% [81.3; 87.7] |
| CNN-CNN_v2_s20261121-policy | 63.6% [59.2; 68.0] | 64.0% [59.9; 68.1] | 85.2% [82.0; 88.3] |
| CNN-CNN_v2_s20261122-policy | 64.0% [59.7; 68.3] | 64.4% [60.1; 68.5] | 85.2% [81.9; 88.4] |
| CNN-RL-20261120-policy | 35.6% [31.3; 39.8] | 53.4% [49.0; 57.8] | 81.6% [78.0; 85.1] |
| PUCT-AZ-20261120 | 67.8% [63.8; 71.8] | 65.0% [60.8; 69.2] | 83.4% [79.9; 86.7] |

## Cổng CNN và so sánh ghép cặp

Tiêu chí tối thiểu đặt trước yêu cầu cận dưới CI 95% của hiệu `CNN − AlphaBeta depth 3` không âm trên tổng thể. Giá trị dưới đây là điểm phần trăm; cổng chỉ đạt nếu khoảng không nằm dưới 0.

| Checkpoint | Hiệu tỷ lệ nước tối ưu, CI ghép cặp 95% | Kết luận cổng |
|---|---:|---|
| CNN-CNN_v2_s20261120-policy | [-9.1; -4.0] điểm % | chưa đạt |
| CNN-CNN_v2_s20261121-policy | [-8.8; -3.7] điểm % | chưa đạt |
| CNN-CNN_v2_s20261122-policy | [-8.6; -3.4] điểm % | chưa đạt |
| RL-Policy (safety-filtered) | [-22.9; -17.8] điểm % | agent tham chiếu, chưa qua cổng CNN |
| PUCT-AZ-20261120, 500 simulations | [-7.8; -2.5] điểm % | agent tham chiếu, chưa qua cổng CNN |

## Arena chọn checkpoint self-play

| Candidate | Incumbent | W-D-L candidate | Score | CI bootstrap 95% | Opening | Kết luận |
|---|---|---:|---:|---:|---:|---|
| AZ-iteration-100 | PUCT-CNN-seed20261121 | 96-22-82 | 0.535 | [47.5; 59.2] | 100 | không promote |

## Value head của CNN checkpoint dùng trong round-robin

Checkpoint seed 20261121: MSE 0.720 (CI 95% [0.664; 0.778]); W/D/L accuracy 53.5% (CI [50.6; 56.5]) so với baseline lớp đa số 52.1%. Ở tàn cuộc, accuracy là 45.8% (CI [40.8; 50.8]).

**Kết luận cổng:** chưa có checkpoint CNN vượt cổng tối thiểu.

## Diễn giải và giới hạn

- Ba CNN supervised đều thấp hơn AlphaBeta depth 3 theo hiệu ghép cặp; cổng tối thiểu chưa đạt. Mục tiêu 85% nước tối ưu cũng chưa đạt.
- AZ self-play đã hoàn tất 100 iteration và 3200 ván tự đấu, nhưng PUCT với 500 mô phỏng/nước trên test đạt 72.1% nước tối ưu; khoảng hiệu ghép cặp với AB3 vẫn âm. Đây là đánh giá một checkpoint/seed, không đủ kết luận xu hướng huấn luyện.
- RL chạy đủ 10,000 episode; solver đánh giá policy sau safety filter thấp hơn baseline tìm kiếm. Kết quả huấn luyện ghi nhận hiệu suất trong lúc học so với Random, không thay cho round-robin độc lập.
- Value head CNN có W/D/L accuracy điểm ước lượng trên baseline lớp đa số, nhưng CI 95% chứa baseline; ở tàn cuộc điểm thấp hơn baseline. Chưa có bằng chứng chắc rằng nó dùng độc lập thay rollout được.
- Arena AZ iteration 200 ván / 100 opening so với PUCT khởi tạo đạt score 0.535 (CI bootstrap 95% [47.5; 59.2] điểm %); cận dưới không vượt 50%, nên checkpoint không được promote.
- Test có các vị trí lấy theo ván nguồn; CI bootstrap theo cụm xử lý phụ thuộc trong cùng ván. Kết quả chỉ áp dụng cho tập test và cấu hình đã ghi, không khái quát tự động sang mọi khai cuộc.

## Nguồn dữ liệu

- Solver evaluation CNN: `results/solver_eval_cnn_v2_3seeds.json`.
- Solver evaluation AZ/RL và đối chứng: `results/solver_eval_az_rl_20261121_seed20261121.json`.
- Value head: `results/value_head/value_cnn_seed20261121_20261121.json`.
- Checkpoint arena: `benchmarks/arena/arena_az_vs_init_100openings_seed20261121.json`.
- Đường cong depth/chất lượng: `docs/solver_depth_curve_learning.svg`.
- Đường cong huấn luyện chẩn đoán (không phải strength test): `docs/training_process_trace.svg`.
- Checkpoint chỉ được dùng sau ingest; manifest, checksum và kiểm tra test leakage nằm trong `results/colab/`.
