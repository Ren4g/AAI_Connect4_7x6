# Bảng Truy xuất Nguồn gốc Số liệu (Claims Traceability Matrix)

Tài liệu này ánh xạ chi tiết các khẳng định định lượng và con số thực nghiệm xuất hiện trong Báo cáo cuối kỳ (`docs/report/AAI.tex`) tới tệp JSON nguồn và đường dẫn khóa tương ứng.

## 1. Kết quả tự kiểm tra ngẫu nhiên 15 số liệu (Spot-check 15 random claims)

| STT | Khẳng định trong Báo cáo | Giá trị số | Tệp JSON nguồn | Đường dẫn khóa (Key path) | Kết quả đối soát |
|---|---|---|---|---|---|
| 1 | AlphaBeta-depth5 đạt 83,07% exact trên tập solver | `83.07%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.AlphaBeta-depth5.all.optimal.rate` | **Khớp 100% (PASS)** |
| 2 | Random đạt 37,27% exact trên tập solver 1.500 vị trí | `37.27%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.Random.all.optimal.rate` | **Khớp 100% (PASS)** |
| 3 | PUCT warm0 (có lọc) đạt 94,00% WDL trên tập solver | `94.00%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.PUCT-PUCTwarm_iter00-filtered.all.wdl_preserving.rate` | **Khớp 100% (PASS)** |
| 4 | Tại vòng 40 warm-start, PUCT thô suy giảm còn 78,26% exact trên validation | `78.26%` | `results/phase_c_local/validation_curve_20261123.json` | `completed.warm_iter40.metrics.puct_raw.exact_optimal_rate` | **Khớp 100% (PASS)** |
| 5 | Trong round-robin chính thức, AlphaBeta-depth4 vs MCTS-500 có score 0,580 | `0.5800` | `results/round_robin_phase_c_20261123.json` | `[first_agent=AlphaBeta-depth4, second_agent=MCTS-500].score` | **Khớp 100% (PASS)** |
| 6 | AlphaBeta-depth4 đạt 80,47% exact trên tập solver | `80.47%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.AlphaBeta-depth4.all.optimal.rate` | **Khớp 100% (PASS)** |
| 7 | CNN v1 AB3 teacher 100% đạt 73,26% exact trung bình trên validation | `73.26%` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[2].exact_optimal_top1_mean` | **Khớp 100% (PASS)** |
| 8 | Heuristic đạt 73,53% exact trên tập solver | `73.53%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.Heuristic.all.optimal.rate` | **Khớp 100% (PASS)** |
| 9 | Trong 2.800 ván round-robin chính thức, P1 đạt score 0,5325 | `0.5325` | `results/round_robin_phase_c_20261123.games.jsonl` | `sum(result==1 + 0.5*result==0)/2800` | **Khớp 100% (PASS)** |
| 10 | CNN v1 Pons-hard 100% đạt 76,93% exact trung bình trên validation | `76.93%` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[1].exact_optimal_top1_mean` | **Khớp 100% (PASS)** |
| 11 | CNN v1 policy (có lọc) đạt 76,73% exact trên tập solver | `76.73%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.CNN-CNNv3-policy.all.optimal.rate` | **Khớp 100% (PASS)** |
| 12 | Tại vòng 0 warm-start, PUCT thô đạt 88,46% exact trên validation | `88.46%` | `results/phase_c_local/validation_curve_20261123.json` | `completed.warm_iter00.metrics.puct_raw.exact_optimal_rate` | **Khớp 100% (PASS)** |
| 13 | PUCT warm0 (có lọc) đạt 82,13% exact trên tập solver | `82.13%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.PUCT-PUCTwarm_iter00-filtered.all.optimal.rate` | **Khớp 100% (PASS)** |
| 14 | CNN v1 raw đạt 68,60% exact trên tập solver | `68.60%` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.CNN-CNNv3-raw.all.optimal.rate` | **Khớp 100% (PASS)** |
| 15 | CNN v1 Pons-soft 100% đạt 80,09% exact trung bình trên validation | `80.09%` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[0].exact_optimal_top1_mean` | **Khớp 100% (PASS)** |

## 2. Toàn bộ danh mục ánh xạ số liệu (Full Traceability Catalog)

| Thành phần | Đại lượng | Giá trị báo cáo | File nguồn | Khóa truy xuất |
|---|---|---|---|---|
| Random đạt 37,27% exact trên tập solver 1.500 vị trí | `0.37266666666666665` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.Random.all.optimal.rate` | Khớp chính xác |
| Heuristic đạt 73,53% exact trên tập solver | `0.7353333333333333` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.Heuristic.all.optimal.rate` | Khớp chính xác |
| AlphaBeta-depth4 đạt 80,47% exact trên tập solver | `0.8046666666666666` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.AlphaBeta-depth4.all.optimal.rate` | Khớp chính xác |
| AlphaBeta-depth5 đạt 83,07% exact trên tập solver | `0.8306666666666667` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.AlphaBeta-depth5.all.optimal.rate` | Khớp chính xác |
| MCTS-500 đạt 75,40% exact trên tập solver | `0.754` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.MCTS-500.all.optimal.rate` | Khớp chính xác |
| CNN v1 raw đạt 68,60% exact trên tập solver | `0.686` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.CNN-CNNv3-raw.all.optimal.rate` | Khớp chính xác |
| CNN v1 policy (có lọc) đạt 76,73% exact trên tập solver | `0.7673333333333333` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.CNN-CNNv3-policy.all.optimal.rate` | Khớp chính xác |
| PUCT warm0 (có lọc) đạt 82,13% exact trên tập solver | `0.8213333333333334` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.PUCT-PUCTwarm_iter00-filtered.all.optimal.rate` | Khớp chính xác |
| PUCT warm0 (có lọc) đạt 94,00% WDL trên tập solver | `0.94` | `results/solver_eval_phase_c_local_20261123.json` | `metrics.PUCT-PUCTwarm_iter00-filtered.all.wdl_preserving.rate` | Khớp chính xác |
| CNN v1 Pons-soft 100% đạt 80,09% exact trung bình trên validation | `0.8008830022075055` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[0].exact_optimal_top1_mean` | Khớp chính xác |
| CNN v1 Pons-hard 100% đạt 76,93% exact trung bình trên validation | `0.7693156732891833` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[1].exact_optimal_top1_mean` | Khớp chính xác |
| CNN v1 AB3 teacher 100% đạt 73,26% exact trung bình trên validation | `0.7325974981604121` | `results/cnn_v3_validation_selection_20261122.json` | `configuration_summaries[2].exact_optimal_top1_mean` | Khớp chính xác |
| Trong round-robin chính thức, AlphaBeta-depth4 vs PUCT có score 0,415 | `0.415` | `results/round_robin_phase_c_20261123.json` | `[first_agent=AlphaBeta-depth4, second_agent=PUCT].score` | Khớp chính xác |
| Trong round-robin chính thức, AlphaBeta-depth6 vs PUCT có score 0,500 | `0.5` | `results/round_robin_phase_c_20261123.json` | `[first_agent=AlphaBeta-depth6, second_agent=PUCT].score` | Khớp chính xác |
| Trong round-robin chính thức, AlphaBeta-depth4 vs MCTS-500 có score 0,580 | `0.58` | `results/round_robin_phase_c_20261123.json` | `[first_agent=AlphaBeta-depth4, second_agent=MCTS-500].score` | Khớp chính xác |
| Trong 2.800 ván round-robin chính thức, P1 đạt score 0,5325 | `0.5325` | `results/round_robin_phase_c_20261123.games.jsonl` | `sum(result==1 + 0.5*result==0)/2800` | Khớp chính xác |
| Tại vòng 0 warm-start, PUCT thô đạt 88,46% exact trên validation | `0.8846325167037862` | `results/phase_c_local/validation_curve_20261123.json` | `completed.warm_iter00.metrics.puct_raw.exact_optimal_rate` | Khớp chính xác |
| Tại vòng 40 warm-start, PUCT thô suy giảm còn 78,26% exact trên validation | `0.7826280623608018` | `results/phase_c_local/validation_curve_20261123.json` | `completed.warm_iter40.metrics.puct_raw.exact_optimal_rate` | Khớp chính xác |