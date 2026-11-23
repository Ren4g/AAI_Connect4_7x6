# Ngu?n v? tr?ng th?i k?t qu? d?ng cho b?o c?o

Ng?y ch?t ph?m vi: 2026-11-21. B?o c?o m? t? baseline ?? ho?n th?nh tr??c khi t?m d?ng ?? c?i thi?n m? h?nh. C?c artifact c?a nh?nh ch?n ?o?n/c?i ti?n ti?p theo kh?ng ???c xem l? k?t qu? c?a baseline.

## Ngu?n ch?nh th?c

| Ch? ?? | Ngu?n ???c d?ng | L? do |
|---|---|---|
| ??nh gi? solver: agent c? ?i?n v? ba seed CNN | `results/solver_eval_cnn_v2_3seeds.json` | 1.500 v? tr?, ??ng SHA256 t?p test ??ng b?ng, solver commit ?? ghi, bootstrap theo c?m; ch?a AlphaBeta depth 1?6, Heuristic, MCTS v? ba CNN. |
| ??nh gi? solver: RL v? AlphaZero | `results/solver_eval_az_rl_20261121_seed20261121.json` | C?ng t?p test v? solver; l? ngu?n cho RL v? PUCT-AZ. Random/MCTS ? ??y l? l??t l?y m?u kh?c, kh?ng thay c?c s? c?a l??t tr??c. |
| C?p 7 PUCT v?i policy/value CNN kh?i t?o | `results/solver_eval_level7_puct500.json` | Checkpoint kh?c PUCT-AZ; gi? th?nh c?u h?nh ri?ng. |
| T??ng t?c gi?i ??u cu?i | `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json`, `.meta.json`, `.games.jsonl` | C?u h?nh c? 100 v?n/c?p, gh?p khai cu?c, ??i gh? v? metadata l?c khai cu?c; ngu?n s? g?c cho round-robin. B?n tr?nh b?y: `docs/ROUND_ROBIN_FINAL.md`. |
| ??i ??u theo ng?n s?ch th?i gian | `benchmarks/head_to_head/benchmark_ab_mcts_standard_100_paired_seed20261126.json` | 100 v?n/50 khai cu?c gh?p c?p; kh?c giao th?c c? ??nh ?? s?u/simulation n?n ???c g?n nh?n th?m d?, kh?ng nh?p v?o b?ng round-robin. |
| Th? nghi?m Heuristic?AB6 | `benchmarks/head_to_head/benchmark_heuristic_ab6_seed20261130.json` | ??i ??u ri?ng 100 v?n; b?o c?o nh? ph?n t?ch b? sung. |
| Ablation t?m ki?m | `benchmarks/ablation/ablation_vs_heuristic_3seeds_paired.json` | Ba seed nh?ng ch? 10 v?n/c?u h?nh/seed; ch? d?ng k?t qu? chi ph? c? m? t? th?m d?, kh?ng kh?ng ??nh kh?c bi?t s?c ch?i. |
| Ch?n ?o?n n??c sai ??u ti?n CNN | `results/cnn_first_errors_seed20261122.json`, `docs/CNN_LOSS_ANALYSIS.md` | Ngu?n cho ph?n lo?i l?i; gi?i h?n l? ph?n t?ch tr?n c?c v?n thua ???c ghi, kh?ng ??i di?n m?i v?n. |
| Value head CNN | `results/value_head/value_cnn_seed20261121_20261121.json` | B?n ??nh gi? chu?n tr?n t?p test ??ng b?ng, c? CI bootstrap. |
| CNN: c?u h?nh v? ti?n tr?nh | `results/colab/cnn_seed20261120_colab1/manifest.json`, `.../cnn_seed20261121_colab1/manifest.json`, `.../cnn_seed20261122_colab1/manifest.json`; d? li?u: `datasets/` | Ba manifest ?? ingest l? ngu?n tham s?, seed, s? b??c, m?i tr??ng v? checkpoint. |
| RL v? AZ: c?u h?nh ch?y | `results/colab/rl_seed20261120_run1/manifest.json`, `results/colab/az_seed20261120_run1/manifest.json` | Artifact Colab ?? ingest; d?ng m? t? ??ng c?u h?nh, kh?ng suy di?n t? loss/reward th?nh s?c ch?i. |
| T?i nguy?n local | `results/resource_profile_20261121_v2.json`, `docs/RESOURCE_PROFILE.md` | ?o tr?n m?y local; s? Colab ch? l?y t? manifest v? kh?ng d?ng l?m so s?nh t?c ?? agent. |
| T?ng h?p k?t qu? hu?n luy?n | `docs/LEARNING_RESULTS.md`, ??i chi?u JSON g?c | B?n tr?nh b?y t?ng h?p; JSON v? manifest v?n l? ngu?n s?. |
| M?i tr??ng, quy tr?nh, quy?t ??nh v? l?i ??nh gi? | `README.md`, `docs/LOG.md`, `docs/CRITERIA.md`, `docs/CRITERIA_V2.md`, `docs/COLAB.md`, m? ngu?n v? test | D?ng l?m ngu?n m? t? thi?t k?, ti?u ch? ?? ??t v? ki?m ch?ng ch?c n?ng. |

## Kh?ng d?ng l?m k?t qu? ch?nh

| Ngu?n | Ph?n lo?i | L? do |
|---|---|---|
| `results/solver_eval_seed20261122.json` v? c?c smoke | Legacy/smoke | Kh?ng c? ?? metadata m?i v? t?p ??ng b?ng/bootstrap ho?c ch? l? ch?y ki?m tra. |
| `results/colab_inbox/` | B?n nh?n tr?ng | B?n ch?nh ?? ingest n?m d??i `results/colab/<run_id>/`. |
| `results/diagnostic_v2_*`, `results/solver_checkpoint_comparison_seed20261201.json`, `results/solver_labels_seed20261201.json`, `results/training_solver_seed20261201.json`, `results/training_gravityaware_seed20261119.json` | Th?m d?/c?i ti?n sau baseline | Kh?ng thu?c m?c k?t qu? ???c ??ng b?ng cho b?o c?o n?y; m?t s? artifact ???c t?o trong nh?nh ki?m tra/c?i thi?n v? kh?ng ???c tr?n v?i k?t qu? ch?nh. |
| `docs/LEARNING_RESULTS_V2.md`, `docs/CRITERIA_V2_ADDENDUM_20261121.md`, `docs/CRITERIA_V2_POSTHOC.md`, notebook/ bundle ??i sau | K? ho?ch ho?c ph?n t?ch sau | Kh?ng ph?i k?t qu? ho?n t?t c?a c?u h?nh baseline ???c b?o c?o. |
| `docs/report.md` c? | B?n nh?p n?n | C? th? tham kh?o c?u tr?c; kh?ng ph?i b?n s? li?u chu?n v? ???c thay b?ng b?o c?o sinh m?i. |

## Quy t?c gi?i quy?t kh?c bi?t

C?c gi? tr? Random v? MCTS c? th? ??i gi?a nh?ng l?n ??nh gi? solver v? agent ng?u nhi?n v? seed kh?c nhau. B?o c?o gi? ch?ng theo t?ng l??t ch?y, kh?ng l?y trung b?nh g?p. Hai PUCT kh?c nhau (PUCT-CNN c?p 7 v? PUCT-AZ c?p 8) d?ng checkpoint kh?c nhau, n?n ???c ghi th?nh hai h?ng. Ngu?n round-robin ???c ch?n theo metadata final; benchmark ng?n s?ch th?i gian l? giao th?c ri?ng v? ???c b?o ri?ng.

## Bổ sung sau Phase C (2026-11-23)

| Artifact | Trạng thái | Cách dùng trong báo cáo |
|---|---|---|
| `results/phase_c_local/validation_curve_20261123.json` | Chính thức cho đường cong validation Phase C v2 | 18 snapshot; exact-optimal policy-only/PUCT có và không filter; CI bootstrap theo source-game; paired delta theo vòng 0. Không chứa WDL/value accuracy hoặc policy entropy. |
| `results/solver_eval_phase_c_local_20261123.json` | Chính thức cho test đóng băng Phase C v2 | Đủ 1.500 vị trí, gồm CNN v3 và hai PUCT checkpoint được chọn bằng validation; CI bootstrap theo cụm ván. |
| `results/round_robin_phase_c_20261123.json`, `.meta.json`, `.games.jsonl` | Chính thức cho round-robin sau Phase C | 28 cặp, 100 ván/cặp, paired opening, fair filter; seed 20261123. |
| `docs/PHASE_C_V2_RESULTS.md`, `docs/ROUND_ROBIN_PHASE_C_20261123.md` | Bản trình bày sinh từ các JSON trên | Không thay thế JSON nguồn. Tái tạo bằng `python scripts/build_phase_c_docs.py`. |
