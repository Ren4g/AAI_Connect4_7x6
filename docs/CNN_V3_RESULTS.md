# CNN v3: tổng hợp validation và frozen-test

## Lựa chọn theo validation

Pons soft · 100% đạt mean exact-optimal top-1 80.1% ± 0.29 điểm phần trăm trên 3 seed. Checkpoint trong run được chọn bằng validation loss; warm-start là `cnn_v3_30k_seed20261122` với validation exact-optimal 80.4%, loss 1.4617. Frozen test không được dùng để chọn.

## Frozen-test

Thử nghiệm dùng 1500 vị trí; SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`, solver commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`, cluster bootstrap 10,000 lần theo source game. Cột “hiệu so với AB3” là CI paired của agent trừ AlphaBeta depth 3 trên cùng các vị trí; cận dưới/ trên chứa 0 thì chưa phân biệt được.

| Seed | Chế độ | N | Exact | CI 95% exact | Giữ WDL | CI 95% WDL | Hiệu exact − AB3, CI 95% |
|---|---|---|---|---|---|---|---|
| 20261121 | Mạng thuần, không safety filter | 1500 | 66.7% | [64.1; 69.2]% | 86.5% | [84.6; 88.2]% | [-13.5; -7.6]% |
| 20261121 | Có safety filter | 1500 | 76.3% | [74.2; 78.5]% | 91.1% | [89.7; 92.5]% | [-3.4; 1.6]% |
| 20261122 | Mạng thuần, không safety filter | 1500 | 68.6% | [66.2; 71.1]% | 87.1% | [85.3; 88.9]% | [-11.6; -5.7]% |
| 20261122 | Có safety filter | 1500 | 76.7% | [74.6; 78.9]% | 91.0% | [89.5; 92.5]% | [-3.0; 2.1]% |
| 20261123 | Mạng thuần, không safety filter | 1500 | 65.5% | [63.0; 68.0]% | 86.5% | [84.8; 88.3]% | [-14.6; -8.8]% |
| 20261123 | Có safety filter | 1500 | 75.7% | [73.5; 77.9]% | 90.5% | [89.1; 92.0]% | [-4.0; 1.1]% |

**AlphaBeta depth 3:** 77.2% exact, CI [75.0; 79.4]%; giữ WDL 90.5%, CI [89.1; 92.0]%.

## Trung bình và độ lệch chuẩn giữa seed

SD ở bảng này là độ lệch chuẩn mẫu giữa ba seed; nó khác với CI bootstrap theo source game ở bảng trên.

| Nhóm | Exact mean | Exact SD | WDL mean | WDL SD |
|---|---|---|---|---|
| CNN v3 raw | 66.9% | 1.55 pp | 86.7% | 0.37 pp |
| CNN v3 filtered | 76.3% | 0.50 pp | 90.9% | 0.32 pp |
| CNN v2 raw baseline | 55.6% | 1.03 pp | 80.9% | 0.98 pp |
| CNN v2 filtered baseline | 70.9% | 0.27 pp | 88.3% | 0.72 pp |

## Diễn giải

- Kết quả học tập được báo cho cấu hình Pons-soft 100%; các cấu hình hard, AlphaBeta-3 teacher, 10% và 30% vẫn được giữ trong validation artifact nhưng không được chọn làm warm start.
- Filter và mạng thuần được báo riêng. Không quy tác dụng của safety filter cho CNN.
- Kết quả này là đánh giá một lần trên test đóng băng đã dùng cho baseline trong dự án, không phải test độc lập mới. Tuyệt đối không dùng lại để chọn cấu hình hoặc checkpoint.
- CI hiệu CNN−AB3 quyết định việc đạt cổng; giá trị mean đơn lẻ không đủ để kết luận.

## Nguồn

- Frozen-test: `results/solver_eval_cnn_v3_soft_3seeds_20261122.json`
- Validation selection: `results/cnn_v3_validation_selection_20261122.json`
- Baseline CNN v2: `results/solver_eval_cnn_v2_3seeds.json`
- Ingested manifests and checkpoints: `results/colab/cnn_v3_30k_*/`
