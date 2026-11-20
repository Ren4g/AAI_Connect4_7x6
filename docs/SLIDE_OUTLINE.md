# Dàn ý slide báo cáo tiến độ

## 1. Bài toán và mục tiêu

- Connect Four 7×6 là trò chơi đối kháng có luật đơn giản nhưng nhánh tìm kiếm lớn.
- Mục tiêu: xây dựng lộ trình tám cấp agent, từ baseline ngẫu nhiên đến tìm kiếm có học và tự chơi.
- Câu hỏi nghiên cứu: học từ solver/self-play cải thiện lựa chọn nước đi thế nào so với tìm kiếm cổ điển?

## 2. Kiến trúc và lộ trình agent

- Engine grid/bitboard dùng API chung và bộ luật có test.
- Cấp 1–5: Random, Heuristic, Alpha-Beta, MCTS, CNN policy/value có giám sát.
- Cấp 6: policy-gradient/actor-critic thăm dò; cấp 7: PUCT dùng prior/value; cấp 8: batched self-play.
- Phân biệt rõ agent đã có code, đã smoke, đã đánh giá, và chưa có kết quả huấn luyện Colab.

## 3. Phương pháp đánh giá

- Tập test đóng băng 1.500 vị trí; Pons trả điểm exact.
- Metric: tỷ lệ nước tối ưu, WDL-preserving, score (thắng + 0,5 hòa), CI bootstrap theo ván/opening.
- Đấu đối kháng dùng opening chung, đổi ghế; thời gian chỉ đo local.
- Bẫy quy trình đã sửa: ván lặp của agent tất định, số liệu hai phía bị gộp, điểm MCTS, khai cuộc thiên lệch và leakage.

## 4. Kết quả baseline hiện có

- Dùng biểu đồ `results/solver_depth_curve_frozen_20261120_seed20261125.svg`.
- AB depth 4: exact 80,5% [78,5; 82,4]; AB depth 5: 83,1% [81,2; 84,9].
- AB depth 6 thấp hơn depth 5 trên mẫu này; không giả định độ sâu tăng luôn làm exact tăng đơn điệu.
- CNN hiện tại chưa có đánh giá held-out hợp lệ; không đưa số legacy vào kết quả chính.

## 5. Pipeline huấn luyện mới

- Solver labels v2: 1.500 vị trí, train/validation theo ván, 0 giao test, solver batch stdin với 8 worker.
- Colab: CNN supervised, self-play PUCT, RL; checkpoint/resume, hash manifest, ingest local.
- Local smoke và resume đã qua; đây là bằng chứng kỹ thuật, không phải mức chơi.

## 6. Hạn chế và bước tiếp theo

- Chưa có artifact huấn luyện Colab được xác minh; chưa có kết luận về CNN mới, PUCT, RL hoặc AlphaZero.
- Cần chạy ba seed CNN, chọn bằng validation; sau đó self-play/RL theo ngân sách.
- Ingest local, đánh giá solver, arena 100 opening, round-robin cuối, phân tích ghế và lỗi.
- Kết luận hiện tại: hạ tầng đánh giá và huấn luyện tiến triển; kết quả baseline ủng hộ Alpha-Beta trung bình, còn các cấp học sâu chờ kiểm định.
