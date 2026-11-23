# Kết quả Phase C v2: self-play PUCT

Ngày chốt kết quả: 23/11/2026. Nội dung này được sinh từ JSON nguồn; kết quả cũ được giữ nguyên.

## Thiết kế và kiểm soát

Hai run hoàn tất 40 vòng, mỗi vòng 32 ván self-play và 200 bước cập nhật, batch 128. PUCT dùng 400 mô phỏng/nước, 16 ván chạy song song. Warm-start khởi tạo từ CNN v3; scratch khởi tạo ngẫu nhiên. Huấn luyện chạy local trên Windows 11 với RTX 3050 Laptop/CUDA; thời gian không so sánh với Colab.

Checkpoint được chọn bằng validation: warm vòng 0 và scratch vòng 40, theo exact-optimal của PUCT có safety filter. Tập validation có 4.530 vị trí; sau khi loại trùng replay còn 4.490 vị trí warm và 4.497 scratch. CI là bootstrap 10.000 lần theo cụm ván nguồn. Đây là một seed mỗi nhánh; kết quả học là thăm dò.

## Đường cong validation

| Run | Vòng | PUCT có filter: exact (95% CI) | Hiệu ghép cặp so với vòng 0 (95% CI) |
|---|---:|---:|---:|
| Warm | 0 | 89.53% [88.61; 90.44]% | — |
| Warm | 5 | 85.84% [84.79; 86.85]% | -3.70 đpt [-4.61 đpt; -2.79 đpt] |
| Warm | 10 | 85.03% [84.00; 86.06]% | -4.50 đpt [-5.42 đpt; -3.60 đpt] |
| Warm | 15 | 85.39% [84.34; 86.43]% | -4.14 đpt [-5.06 đpt; -3.23 đpt] |
| Warm | 20 | 85.70% [84.65; 86.72]% | -3.83 đpt [-4.76 đpt; -2.90 đpt] |
| Warm | 25 | 85.19% [84.14; 86.21]% | -4.34 đpt [-5.26 đpt; -3.45 đpt] |
| Warm | 30 | 85.37% [84.35; 86.40]% | -4.16 đpt [-5.11 đpt; -3.23 đpt] |
| Warm | 35 | 85.08% [84.02; 86.09]% | -4.45 đpt [-5.38 đpt; -3.50 đpt] |
| Warm | 40 | 85.90% [84.87; 86.93]% | -3.63 đpt [-4.57 đpt; -2.71 đpt] |
| Scratch | 0 | 81.48% [80.34; 82.60]% | — |
| Scratch | 5 | 80.32% [79.16; 81.47]% | -1.16 đpt [-2.28 đpt; +0.00 đpt] |
| Scratch | 10 | 82.21% [81.08; 83.31]% | +0.73 đpt [-0.44 đpt; +1.90 đpt] |
| Scratch | 15 | 81.52% [80.41; 82.65]% | +0.04 đpt [-1.16 đpt; +1.22 đpt] |
| Scratch | 20 | 81.77% [80.62; 82.91]% | +0.29 đpt [-0.85 đpt; +1.47 đpt] |
| Scratch | 25 | 81.70% [80.55; 82.83]% | +0.22 đpt [-0.92 đpt; +1.37 đpt] |
| Scratch | 30 | 82.25% [81.14; 83.37]% | +0.78 đpt [-0.37 đpt; +1.93 đpt] |
| Scratch | 35 | 81.94% [80.79; 83.07]% | +0.47 đpt [-0.68 đpt; +1.63 đpt] |
| Scratch | 40 | 82.48% [81.35; 83.59]% | +1.00 đpt [-0.16 đpt; +2.14 đpt] |

Warm-start suy giảm sau self-play; scratch tăng từ vòng 0 đến 40 nhưng hiệu ghép cặp có CI chứa 0, nên chưa phân biệt được cải thiện. Không suy diễn nguyên nhân từ đường cong này.

Bộ JSON cũng chứa policy-only/PUCT, có và không safety filter cho mỗi snapshot. Loss policy/value riêng và entropy target không được ghi; do đó không có đường cong cho các đại lượng này.

## Đánh giá test đóng băng

Chỉ sau khi chọn checkpoint trên validation, đánh giá dùng đủ 1.500 vị trí đóng băng. Exact là tỷ lệ chọn nước thuộc tập nước tối ưu Pons; WDL là tỷ lệ bảo toàn kết cục thắng/hòa/thua. CI bootstrap theo cụm ván nguồn.

| Agent/cấu hình | Exact (95% CI) | WDL (95% CI) |
|---|---:|---:|
| CNN v3 policy, có filter | 76.73% [74.54; 78.91]% | 91.00% [89.47; 92.44]% |
| PUCT-400 warm vòng 0, thô | 81.20% [79.14; 83.19]% | 94.00% [92.75; 95.20]% |
| PUCT-400 warm vòng 0, có filter | 82.13% [80.14; 84.03]% | 94.00% [92.70; 95.20]% |
| PUCT-400 scratch vòng 40, thô | 72.53% [70.25; 74.82]% | 91.53% [90.04; 92.91]% |
| PUCT-400 scratch vòng 40, có filter | 76.80% [74.66; 78.97]% | 91.87% [90.46; 93.23]% |

Warm vòng 0 so với AlphaBeta depth 3: exact +4.93 đpt, CI ghép cặp [+2.66 đpt; +7.21 đpt]; WDL +3.47 đpt, CI [+1.99 đpt; +4.95 đpt]. Cả hai CI không chứa 0.
Scratch vòng 40 so với AlphaBeta depth 3: exact -0.40 đpt, CI ghép cặp [-3.01 đpt; +2.12 đpt]; WDL +1.33 đpt, CI [-0.40 đpt; +2.98 đpt]. Các CI chứa 0.

## Round-robin sau Phase C

Tám agent, 100 ván/cặp, 6 ply mở đầu, ghép cặp theo khai cuộc và đổi ghế. Bộ lọc công bằng thử 61 khai cuộc; giữ 50, loại 10 khai cuộc có thắng tức thì và 0 khai cuộc không có nước an toàn. Engine bitboard; seed 20261123; không giới hạn thời gian. Score = (thắng + 0,5 hòa)/số ván; CI bootstrap theo paired opening.

| Agent A | Agent B | W-D-L A | Score A | CI 95% |
|---|---|---:|---:|---:|
| Random | Heuristic | 0-0-100 | 0.000 | [0.000; 0.000] |
| Random | AlphaBeta-depth4 | 0-0-100 | 0.000 | [0.000; 0.000] |
| Random | MCTS-500 | 0-0-100 | 0.000 | [0.000; 0.000] |
| Random | AlphaBeta-depth6 | 0-0-100 | 0.000 | [0.000; 0.000] |
| Random | CNN-Policy | 0-0-100 | 0.000 | [0.000; 0.000] |
| Random | RL-Policy | 1-0-99 | 0.010 | [0.000; 0.030] |
| Random | PUCT | 0-0-100 | 0.000 | [0.000; 0.000] |
| Heuristic | AlphaBeta-depth4 | 22-6-72 | 0.250 | [0.180; 0.325] |
| Heuristic | MCTS-500 | 33-5-62 | 0.355 | [0.285; 0.425] |
| Heuristic | AlphaBeta-depth6 | 18-4-78 | 0.200 | [0.130; 0.270] |
| Heuristic | CNN-Policy | 39-9-52 | 0.435 | [0.335; 0.535] |
| Heuristic | RL-Policy | 86-6-8 | 0.890 | [0.825; 0.945] |
| Heuristic | PUCT | 13-6-81 | 0.160 | [0.100; 0.225] |
| AlphaBeta-depth4 | MCTS-500 | 56-4-40 | 0.580 | [0.510; 0.645] |
| AlphaBeta-depth4 | AlphaBeta-depth6 | 36-3-61 | 0.375 | [0.295; 0.460] |
| AlphaBeta-depth4 | CNN-Policy | 65-5-30 | 0.675 | [0.585; 0.760] |
| AlphaBeta-depth4 | RL-Policy | 96-1-3 | 0.965 | [0.925; 0.995] |
| AlphaBeta-depth4 | PUCT | 36-11-53 | 0.415 | [0.335; 0.495] |
| MCTS-500 | AlphaBeta-depth6 | 27-2-71 | 0.280 | [0.200; 0.365] |
| MCTS-500 | CNN-Policy | 57-3-40 | 0.585 | [0.490; 0.675] |
| MCTS-500 | RL-Policy | 90-0-10 | 0.900 | [0.840; 0.950] |
| MCTS-500 | PUCT | 25-6-69 | 0.280 | [0.190; 0.375] |
| AlphaBeta-depth6 | CNN-Policy | 75-6-19 | 0.780 | [0.710; 0.850] |
| AlphaBeta-depth6 | RL-Policy | 97-1-2 | 0.975 | [0.945; 1.000] |
| AlphaBeta-depth6 | PUCT | 47-6-47 | 0.500 | [0.405; 0.595] |
| CNN-Policy | RL-Policy | 87-2-11 | 0.880 | [0.815; 0.935] |
| CNN-Policy | PUCT | 18-4-78 | 0.200 | [0.135; 0.265] |
| RL-Policy | PUCT | 0-0-100 | 0.000 | [0.000; 0.000] |

## Cấu hình học và giới hạn

- `az_v2_warm_seed20261121`: buffer cuối 45,385; trung bình 35.46 ply/ván; 463/1.280 hòa; 8 lần resume.
- `az_v2_scratch_seed20261122`: buffer cuối 37,303; trung bình 29.15 ply/ván; 226/1.280 hòa; 7 lần resume.

Replay cap 100.000. Mỗi vòng sampling có hoàn lại `200 × 128 = 25.600` vị trí. Mirror augmentation được áp dụng ngẫu nhiên 50% trong batch, không nhân đôi buffer. Adam, learning rate 0,001 cố định, không scheduler, weight decay mặc định 0; loss = policy cross-entropy + 0,5 × MSE value. Policy target chuẩn hóa visit count; temperature 1 trong 12 ply đầu rồi 0. Value target nhìn từ người đến lượt: thắng +1, thua −1, hòa 0. Self-play không có safety filter và không đấu với checkpoint tốt nhất.

Train log lưu loss tổng, độ dài ván và số hòa từng vòng; không lưu policy/value loss riêng hay entropy policy. Validation không tính WDL/value accuracy/calibration theo snapshot. Chưa chạy PUCT-100/1600, MCTS rollout đối chứng, arena snapshot hoặc ablation LR/update. Các kiểm tra batched-vs-single inference và test dấu end-to-end vẫn còn thiếu.

## File nguồn

- Validation: `results/phase_c_local/validation_curve_20261123.json`.
- Test solver: `results/solver_eval_phase_c_local_20261123.json`.
- Round-robin: `results/round_robin_phase_c_20261123.json`; metadata `results/round_robin_phase_c_20261123.meta.json`.
