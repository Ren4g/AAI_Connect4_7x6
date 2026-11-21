# Báo cáo tiến độ Connect Four 7×6

Cập nhật: 21-11-2026

## Tóm tắt

Hệ thống hiện có đủ mã cho tám cấp agent, bộ đánh giá exact bằng Pons, pipeline CNN/AlphaZero/RL có checkpoint-resume và ingest, cùng các notebook Colab. Đã nhận và kiểm định artifact thực nghiệm Colab cho CNN supervised, self-play AlphaZero và policy-gradient RL. Đây là kết quả đo từ các gói đã ingest, không còn là smoke thử pipeline.

## Kết quả đã xác nhận

- Các CNN solver-label đã được chọn bằng validation, ingest và chấm trên test đóng băng; chúng chưa đạt cổng so sánh ghép cặp với AlphaBeta depth 3.
- Self-play AlphaZero đã hoàn tất; PUCT-AZ được đánh giá với số mô phỏng cố định và chưa vượt AlphaBeta depth 3 theo khoảng tin cậy ghép cặp. Arena với PUCT từ CNN khởi tạo không qua cổng thăng hạng, nên không xem checkpoint self-play là cải thiện đã xác nhận.
- Run policy-gradient RL đã hoàn tất. Policy sau safety filter yếu hơn baseline tìm kiếm trên thước đo solver; kết quả trong lúc huấn luyện đấu Random không được dùng thay round-robin độc lập.
- Value head của CNN được chọn cho giải đấu chưa cho thấy khác biệt rõ so với baseline lớp đa số; ở tàn cuộc accuracy thấp hơn baseline. Chưa có căn cứ dùng value head độc lập thay rollout.
- Các phép đo dùng chung test đóng băng và solver đã pin; bootstrap theo ván nguồn. Thời gian suy luận chỉ đo trên máy local.

Bảng số liệu, cỡ mẫu, CI đầy đủ theo giai đoạn, cổng CNN, cấu hình Colab và nguồn JSON được sinh tự động trong [LEARNING_RESULTS.md](LEARNING_RESULTS.md).

## Đánh giá phương pháp

Các gói CNN/RL/AZ chỉ được chấp nhận sau khi kiểm checksum, cấu hình và phiên bản nguồn; chạy inference CPU; kiểm tra lại local rằng tập train không chứa trạng thái của test, kể cả trạng thái gương. Solver evaluation dùng test đóng băng chung và đối chứng AlphaBeta nhiều độ sâu. Round-robin cuối đã chạy đủ cặp với opening chung, đổi ghế, bộ lọc khai cuộc công bằng và các agent đã ingest.

## Tiến độ tám cấp

| Cấp | Tình trạng hiện tại |
|---|---|
| 1–4: Random, Heuristic, AlphaBeta, MCTS | Có agent và benchmark local |
| 5: CNN supervised | Ba run Colab đã ingest, đánh giá xong; chưa đạt cổng so với AB3 |
| 6: RL | Run policy-gradient đã ingest; solver và round-robin đã đánh giá |
| 7: AlphaGo-inspired | PUCT policy/value đã được đánh giá trên test và round-robin |
| 8: AlphaZero-inspired | Self-play đã chạy; checkpoint đánh giá qua PUCT và round-robin |
| Solver evaluation | CNN, PUCT-AZ, RL và AlphaBeta nhiều độ sâu trên frozen test |
| Round-robin cuối | Hoàn tất trên local; opening chung, hai ghế và bộ lọc công bằng |

## Việc còn lại

1. Các phép đo RAM local và phần VRAM có trong manifest đã được tổng hợp ở [RESOURCE_PROFILE.md](RESOURCE_PROFILE.md). Manifest AZ/RL không ghi peak VRAM, nên trường này còn thiếu cho hai run đó.
2. Hoàn thiện hình thức báo cáo theo yêu cầu cụ thể của môn học nếu có mẫu/độ dài bắt buộc; hiện đã có báo cáo tiến độ, kết quả chi tiết, dàn ý slide và kịch bản demo.
3. Có thể mở rộng thêm seed cho RL/AZ và giải đấu nếu cần kết luận tổng quát hơn; mỗi nhánh RL/AZ hiện chỉ có một run/checkpoint nên kết luận giới hạn ở cấu hình đã đo.

## Tệp nguồn chính

- [Tiêu chí đóng băng](CRITERIA.md)
- [Báo cáo tổng kết học thuật sinh từ JSON](report.md)
- [Nhật ký thực nghiệm](LOG.md)
- [Kết quả CNN/RL/AlphaZero sinh từ JSON](LEARNING_RESULTS.md)
- [Đường cong chẩn đoán quá trình huấn luyện](training_process_trace.svg) — không diễn giải như strength test.
- [Đường cong AlphaBeta depth theo solver](solver_depth_curve_learning.svg)
- [Round-robin cuối và heatmap](ROUND_ROBIN_FINAL.md), [SVG](round_robin_final.svg)
- [Đo RAM local và VRAM Colab](RESOURCE_PROFILE.md)
- Solver JSON: `results/solver_eval_cnn_v2_3seeds.json`, `results/solver_eval_az_rl_20261121_seed20261121.json`
- Round-robin JSON/JSONL/meta: `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.*`
- Value-head JSON: `results/value_head/value_cnn_seed20261121_20261121.json`
- Manifest và checkpoint Colab đã ingest: `results/colab/`
