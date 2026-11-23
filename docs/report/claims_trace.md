# Truy vết số liệu trong báo cáo

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

| Phase C validation curve and paired deltas | `results/phase_c_local/validation_curve_20261123.json` → `completed.*.metrics`, `paired_delta_vs_iteration_0`. |
| Phase C frozen-test exact/WDL and paired AB3 deltas | `results/solver_eval_phase_c_local_20261123.json` → `metrics.*`, `paired_difference_vs_ab3`. |
| Phase C round-robin W-D-L, score and CI | `results/round_robin_phase_c_20261123.json`; configuration/opening diagnostics from matching `.meta.json`. |
