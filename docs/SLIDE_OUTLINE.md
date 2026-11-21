# Dàn ý slide báo cáo tiến độ

## 1. Bài toán và câu hỏi nghiên cứu

Connect Four 7×6 có luật ngắn gọn nhưng không gian trạng thái lớn. Đề tài xây dựng tám cấp tác nhân từ baseline ngẫu nhiên đến học có giám sát, tìm kiếm kết hợp mạng và tự chơi. Câu hỏi trung tâm: khi solver exact được dùng làm thước đo, các phương pháp học có vượt baseline tìm kiếm cổ điển không?

## 2. Kiến trúc và tiến trình agent

Trình bày engine grid/bitboard và API dùng chung. Nối tám cấp thành tiến trình: Random → Heuristic → AlphaBeta → MCTS → supervised CNN → policy-gradient RL → policy/value PUCT → AlphaZero-inspired self-play. Phân biệt trạng thái đã hiện thực, đã chạy và đã đánh giá.

## 3. Phương pháp đánh giá có kiểm soát

Mô tả test Pons đóng băng, lấy mẫu theo ván nguồn, cân bằng ba giai đoạn, canonical mirror hashes, score trận đấu và CI bootstrap theo cụm. Round-robin dùng opening chung, đổi ghế và bộ lọc thế khai cuộc đã phân định. Nêu các sai lệch đã phát hiện: ván lặp của agent tất định, thống kê gộp hai bên, điểm MCTS chưa chuẩn hóa, opening có thắng ngay và leakage.

## 4. Từ pipeline đến thí nghiệm Colab

Cho thấy ba loại job CNN, RL, self-play AlphaZero; checkpoint định kỳ, resume, manifest và SHA256; ingest local kiểm checksum, inference CPU và test leakage trước đánh giá. Chỉ tính các run đã ingest là kết quả thực nghiệm.

Đường cong reward RL và loss/self-play AlphaZero nằm ở `docs/training_process_trace.svg`; đây là diagnostic trong quá trình học, không phải phép đo strength độc lập.

## 5. Kết quả solver: năng lực chọn nước

Dùng bảng trong `docs/LEARNING_RESULTS.md`. Đặt AlphaBeta depth 3–6 cạnh CNN, RL và PUCT-AZ; chỉ rõ cổng CNN đã định trước và kết luận cổng. Dùng số liệu và CI trực tiếp từ JSON, không chép số từ log thăm dò.

## 6. Round-robin trực tiếp

Dùng `docs/ROUND_ROBIN_FINAL.md` và `docs/round_robin_final.svg`. So sánh W-D-L, score, CI theo opening và chi phí local. Nêu rõ PUCT-500 là suy luận từ checkpoint self-play; RL trong giải đấu dùng PolicyAgent kèm safety filter. Kết thúc bằng arena checkpoint: candidate không vượt cổng promotion.

## 7. Kết luận, giới hạn và hướng tiếp theo

Kết luận theo bằng chứng: CNN solver-label chưa qua cổng tối thiểu; value head chưa cho thấy lợi thế rõ so với baseline; checkpoint AZ/RL mới có một run cho mỗi phương pháp. Nêu giới hạn của một test đóng băng và một seed RL/AZ. Hướng tiếp theo: cải thiện policy/value, tăng seed/self-play, rồi chạy lại giao thức mà không đổi test hay ngưỡng hậu nghiệm.

### Nhịp kể

Đặt vấn đề → xây baseline → phát hiện sai lệch đánh giá → đóng băng giao thức → huấn luyện và ingest → đối chiếu solver/trận đấu → kết luận vừa đủ bằng chứng.
