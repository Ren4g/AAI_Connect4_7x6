# Chỉ mục hiện vật và provenance thực nghiệm

Ngày kiểm kê: 2026-11-24. Tài liệu này ghi các nguồn chính được dùng trong báo cáo và hiện vật đã được lưu trong repo để kiểm tra lại kết quả.

## Dữ liệu kiểm thử đóng băng

- Vị trí: `results/solver_eval_seed20261122.positions.json`.
- Manifest: `results/solver_eval_seed20261122.manifest.json`.
- Kết quả đánh giá chính: `results/solver_eval_phase_c_local_20261123.json`.
- SHA256 vị trí đóng băng: `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`.
- Solver Pons commit: `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`.

## CNN supervised và các ablation validation

Các artifact sau đã được đưa vào Git dù `results/colab/` mặc định bị `.gitignore` loại trừ:

- CNN v2, ba seed: `results/colab/cnn_seed20261120_colab1/`, `cnn_seed20261121_colab1/`, `cnn_seed20261122_colab1/`.
- CNN v3, ba seed và năm cấu hình mỗi seed: `results/colab/cnn_v3_30k_seed20261121*/`, `results/colab/cnn_v3_30k_seed20261122*/`, `results/colab/cnn_v3_30k_seed20261123*/`.
- RL policy-gradient: `results/colab/rl_seed20261120_run1/`.
- AlphaZero lịch sử: `results/colab/az_seed20261120_run1/`.

Trong các thư mục run nêu trên, repo lưu `manifest.json`, `train_log.jsonl` và checkpoint được dùng làm bằng chứng/kết quả (`checkpoint_best.pt`, hoặc `checkpoint_policy.pt` với các run RL/AZ). Các checkpoint `checkpoint_latest.pt` dùng tiếp tục huấn luyện không được thêm vào repo. SHA256 của checkpoint và train log được đối chiếu với `files_sha256` trong manifest trước khi thêm.

Kết quả tổng hợp CNN v3 nằm tại `results/cnn_v3_validation_selection_20261122.json` và `results/solver_eval_cnn_v3_soft_3seeds_20261122.json`. Dữ liệu huấn luyện cùng manifest/hash nằm trong `colab/data/solverlabels_20261121_seed20261121_v3_30k/` và `colab/data/solverlabels_20261121_seed20261121_ab3_30k/`.

**Giới hạn provenance CNN v3:** các manifest ghi `source_code_commit: null` và source-code SHA256 `e2593baa124953f975cd9afbf2fb29a199699f928de646ece414f7df6600e83d`. Không tìm thấy source bundle v11 trong repo và không thể khôi phục chính xác source tree từ riêng SHA256; vì vậy checkpoint, manifest và log được lưu để kiểm tra kết quả, nhưng chúng không thay thế được bản mã nguồn huấn luyện đúng hash.

## Phase C V2

- Validation snapshots: `results/phase_c_local/validation_curve_20261123.json`.
- Solver test: `results/solver_eval_phase_c_local_20261123.json`.
- Hai run gồm manifest, log, checkpoint và snapshot: `results/phase_c_local/az_v2_warm_seed20261121/` và `results/phase_c_local/az_v2_scratch_seed20261122/`.
- Mã nguồn được đóng gói trong `results/phase_c_local/source_snapshot_703fd34468d0038e328078495ebb67e55104c4cdfbfde3941d90ce988c5ef8.zip`; source tree SHA256 `703fd34468d0038e328078495ebb67e55104c4cdfbfde3941d90ce988c5ef8`; archive SHA256 `2aff20dc96fa1c9129c96e607d25816ef9ef0e33deab21417758208c7c331338`.
- Biểu đồ được dẫn trong báo cáo: `docs/report/assets/phase_c_validation_curve.png`.

**Giới hạn provenance Phase C:** source snapshot là archive theo hash nêu trên; không có source bundle v12 trong repo. Manifest hai run ghi commit `d90ecfa1e3c9a37367b59f5e010b0de247d09390` và source tree SHA256 trùng hash snapshot.

## Round-robin và tài nguyên

- Round-robin Phase C: `results/round_robin_phase_c_20261123.json`, `.meta.json`, `.games.jsonl`.
- Round-robin lịch sử: `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json`, `.meta.json`, `.games.jsonl`.
- Hồ sơ RSS: `results/resource_profile_20261121_v2.json`; báo cáo tổng hợp: `docs/RESOURCE_PROFILE.md`.

JSON là nguồn kết quả máy đọc; metadata mô tả checkpoint, seed, bộ khai cuộc, số lần bootstrap và cấu hình. File `.games.jsonl` lưu từng ván và replay để tái kiểm tra kết quả đấu.
