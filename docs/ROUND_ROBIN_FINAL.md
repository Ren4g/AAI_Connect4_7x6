# Round-robin cuối: báo cáo kết quả

- Nguồn: `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json`; seed 20261121; 100 ván/cặp.
- Engine `bitboard`; opening 6 ply; paired openings=True; bộ lọc `fair`.
- Sinh 56 opening, giữ 50; loại 6 opening có thắng ngay và 0 opening không có nước trả lời an toàn.
- Thời gian chỉ đo local trên cùng máy; ngân sách thời gian=None.
- Score là (thắng + 0,5 hòa)/số ván; CI 95% bootstrap 10,000 lần, seed 20261119, đơn vị opening ghép cặp.

## Ma trận score

Ô hàng i/cột j là score của agent hàng i khi đấu agent cột j; ô đối xứng bằng `1 − score`.

| Agent | Random | Heuristic | AlphaBeta-depth4 | MCTS-500 | AlphaBeta-depth6 | CNN-Policy | RL-Policy | PUCT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 | 0.000 |
| Heuristic | 1.000 | — | 0.285 | 0.325 | 0.200 | 0.545 | 0.850 | 0.365 |
| AlphaBeta-depth4 | 1.000 | 0.715 | — | 0.590 | 0.295 | 0.775 | 0.965 | 0.495 |
| MCTS-500 | 1.000 | 0.675 | 0.410 | — | 0.235 | 0.710 | 0.955 | 0.480 |
| AlphaBeta-depth6 | 1.000 | 0.800 | 0.705 | 0.765 | — | 0.830 | 0.975 | 0.600 |
| CNN-Policy | 1.000 | 0.455 | 0.225 | 0.290 | 0.170 | — | 0.770 | 0.235 |
| RL-Policy | 0.980 | 0.150 | 0.035 | 0.045 | 0.025 | 0.230 | — | 0.070 |
| PUCT | 1.000 | 0.635 | 0.505 | 0.520 | 0.400 | 0.765 | 0.930 | — |

## Kết quả từng cặp

| Agent A | Agent B | W-D-L của A | Score A | CI 95% | Opening |
|---|---|---:|---:|---:|---:|
| Random | Heuristic | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Random | AlphaBeta-depth4 | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Random | MCTS-500 | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Random | AlphaBeta-depth6 | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Random | CNN-Policy | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Random | RL-Policy | 2-0-98 | 0.020 | [0.0; 5.0] | 50 |
| Random | PUCT | 0-0-100 | 0.000 | [0.0; 0.0] | 50 |
| Heuristic | AlphaBeta-depth4 | 26-5-69 | 0.285 | [20.5; 37.0] | 50 |
| Heuristic | MCTS-500 | 32-1-67 | 0.325 | [25.0; 40.0] | 50 |
| Heuristic | AlphaBeta-depth6 | 18-4-78 | 0.200 | [13.0; 27.5] | 50 |
| Heuristic | CNN-Policy | 49-11-40 | 0.545 | [46.5; 63.0] | 50 |
| Heuristic | RL-Policy | 83-4-13 | 0.850 | [77.5; 91.5] | 50 |
| Heuristic | PUCT | 31-11-58 | 0.365 | [28.0; 45.0] | 50 |
| AlphaBeta-depth4 | MCTS-500 | 56-6-38 | 0.590 | [52.0; 66.5] | 50 |
| AlphaBeta-depth4 | AlphaBeta-depth6 | 25-9-66 | 0.295 | [22.0; 37.5] | 50 |
| AlphaBeta-depth4 | CNN-Policy | 74-7-19 | 0.775 | [69.0; 85.0] | 50 |
| AlphaBeta-depth4 | RL-Policy | 96-1-3 | 0.965 | [93.0; 99.5] | 50 |
| AlphaBeta-depth4 | PUCT | 45-9-46 | 0.495 | [40.5; 58.5] | 50 |
| MCTS-500 | AlphaBeta-depth6 | 21-5-74 | 0.235 | [16.0; 31.5] | 50 |
| MCTS-500 | CNN-Policy | 67-8-25 | 0.710 | [63.0; 79.0] | 50 |
| MCTS-500 | RL-Policy | 95-1-4 | 0.955 | [91.0; 99.0] | 50 |
| MCTS-500 | PUCT | 45-6-49 | 0.480 | [39.0; 57.0] | 50 |
| AlphaBeta-depth6 | CNN-Policy | 81-4-15 | 0.830 | [76.0; 89.5] | 50 |
| AlphaBeta-depth6 | RL-Policy | 97-1-2 | 0.975 | [94.0; 100.0] | 50 |
| AlphaBeta-depth6 | PUCT | 57-6-37 | 0.600 | [50.0; 70.0] | 50 |
| CNN-Policy | RL-Policy | 73-8-19 | 0.770 | [69.5; 84.0] | 50 |
| CNN-Policy | PUCT | 16-15-69 | 0.235 | [16.5; 30.5] | 50 |
| RL-Policy | PUCT | 6-2-92 | 0.070 | [2.5; 13.0] | 50 |

## Đo chi phí local theo agent

| Agent | Nước đi | ms/nước | p95 ms/nước | node/nước | simulation/nước |
|---|---:|---:|---:|---:|---:|
| Random | 2910 | 0.005 | 0.007 | 0.0 | 0.0 |
| Heuristic | 6718 | 0.266 | 0.435 | 0.0 | 0.0 |
| AlphaBeta-depth4 | 7634 | 6.047 | 15.521 | 337.0 | 0.0 |
| MCTS-500 | 7464 | 23.623 | 37.699 | 0.0 | 500.0 |
| AlphaBeta-depth6 | 7312 | 68.428 | 215.383 | 3866.7 | 0.0 |
| CNN-Policy | 7431 | 1.327 | 2.381 | 0.0 | 0.0 |
| RL-Policy | 6523 | 1.277 | 2.375 | 0.0 | 0.0 |
| PUCT | 8042 | 215.711 | 403.285 | 0.0 | 500.0 |

Thời gian/nước và p95 tính trực tiếp từ JSONL replay; đơn vị thời gian không so sánh giữa Colab và local.
