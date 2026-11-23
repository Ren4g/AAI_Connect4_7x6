# Round-robin cuối sau Phase C v2

Nguồn: `results/round_robin_phase_c_20261123.json`; seed 20261123; CI 95% bootstrap 10.000 lần theo paired opening.

Giao thức: 100 ván mỗi cặp; 50 khai cuộc 6 ply dùng chung cho hai ghế; engine bitboard; bộ lọc fair; không có ngân sách thời gian. Trong 61 lần thử khai cuộc, 50 được giữ; 10 bị loại do thắng tức thì khả dụng, một lần bị loại do trùng trạng thái.

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

Các so sánh trọng tâm:

| So sánh | Score A | CI 95% | Diễn giải |
|---|---:|---:|---|
| AlphaBeta-depth4 – MCTS-500 | 0.580 | [0.510; 0.645] | score trên 0,5 |
| AlphaBeta-depth4 – PUCT | 0.415 | [0.335; 0.495] | score dưới 0,5 |
| AlphaBeta-depth6 – PUCT | 0.500 | [0.405; 0.595] | CI bevat 0,5: chưa phân biệt được |
| CNN-Policy – PUCT | 0.200 | [0.135; 0.265] | score dưới 0,5 |
| PUCT – RL-Policy | 1.000 | [1.000; 1.000] | score trên 0,5 |
