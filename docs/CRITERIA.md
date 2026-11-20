# Tiêu chí và giao thức đánh giá (đóng băng trước thí nghiệm)

Ngày chốt: 2026-11-20. Các ngưỡng trong tài liệu này được đặt trước các thí
nghiệm tiếp theo. Mọi thay đổi sau khi xem kết quả phải có ngày, lý do và phiên
bản mới trong `docs/LOG.md`; không sửa hồi tố tiêu chí của một lần chạy.

## 1. Tập dữ liệu và chống rò rỉ

- Tập solver đóng băng là `results/solver_eval_seed20261122.positions.json` (1.500
  prefix vị trí; 500 khai cuộc, 500 trung cuộc, 500 tàn cuộc). Bản dữ liệu gốc
  không được sửa hoặc dùng để train, validation, chọn epoch, chọn kiến trúc hay
  chọn siêu tham số.
- Pha 1 phải tạo manifest cạnh tập test, khôi phục `source_game` từ log gốc,
  tính SHA256 của file và khóa vị trí chuẩn hóa gương. Nếu không khôi phục được
  cụm nguồn duy nhất thì vẫn giữ tập vị trí bất biến, nhưng không công bố CI
  bootstrap theo ván cho metric tuyệt đối; CI ghép cặp chỉ dùng được nếu cụm
  nguồn được xác định hợp lệ.
- Mọi dữ liệu huấn luyện/validation mới phải lưu băm chuỗi nước và trạng thái
  bàn cờ chuẩn hóa gương, đối chiếu với tập test đóng băng và dừng nếu giao khác
  rỗng. Chia train/validation theo ván nguồn trước khi augmentation; chỉ
  augmentation gương ở train. Không dùng log đã lấy mẫu từ test làm dữ liệu
  huấn luyện.
- Những checkpoint và kết quả đã tạo trước tiêu chí này được ghi nhãn legacy.
  Nếu kiểm toán tìm thấy giao với test thì chúng không được xem là đánh giá
  held-out hợp lệ.

## 2. Định nghĩa metric

- **Exact-optimal rate**: tỷ lệ vị trí agent chọn một nước có điểm exact lớn
  nhất theo Pons trong các nước hợp lệ. Chấp nhận mọi nước đồng hạng tối ưu.
- **WDL-preserving rate**: tỷ lệ vị trí nước được chọn có cùng dấu điểm solver
  với nước tối ưu (thắng/hòa/thua); đây là metric riêng, không thay cho exact.
- **Vị trí có quyết định**: vị trí mà tồn tại ít nhất một nước hợp lệ có dấu
  điểm solver khác dấu của nước tối ưu. Báo cáo tập này bên cạnh toàn bộ test.
- **Giá trị lý thuyết**: phân tầng vị trí theo dấu của điểm tốt nhất: thắng,
  hòa, thua cho bên đi hiện tại.
- **Match score**: `(wins + 0.5 * draws) / games`. Luôn kèm W/D/L, số ván và CI;
  không gọi score là win rate.
- CI mặc định 95%, bootstrap 10.000 lần theo cụm ván nguồn. Lấy mẫu ghép cặp
  phải dùng cùng danh sách cụm cho hai agent và báo CI của hiệu. Round-robin
  dùng khai cuộc làm cụm và ghép hai ghế của cùng opening. Ghi seed bootstrap.

## 3. Cổng CNN trước self-play

### Cổng tối thiểu

Trên tập test đóng băng, cận dưới CI bootstrap ghép cặp 95% theo ván của hiệu
`CNN exact-optimal rate - AlphaBeta depth-3 exact-optimal rate` phải `>= 0`
tổng thể. So sánh phải dùng đúng cùng các vị trí, nước đi CNN inference nguyên
trạng và cùng version solver. Báo cáo WDL-preserving cùng CI, nhưng không thay
đổi cổng exact tối thiểu này sau khi thấy kết quả.

### Mục tiêu phát triển (không phải cổng chặn)

- Exact-optimal: ít nhất 85% tổng thể và 75% ở từng giai đoạn.
- WDL-preserving: ít nhất 95% tổng thể và 90% ở từng giai đoạn.
- Mục tiêu được đặt sau khi xem baseline cũ; vì vậy phải ghi là mục tiêu phát
  triển, không diễn giải như ngưỡng độc lập được định trước từ đầu dự án.

Tối đa hai vòng cấu hình huấn luyện CNN mới; mỗi vòng tối đa ba seed cố định.
Chọn checkpoint bằng validation, không bằng tập test. Nếu hết hai vòng mà chưa
qua cổng tối thiểu, vẫn tiếp tục agent Pha 3 và Pha 5, ghi rõ CNN chưa đạt.

## 4. Cổng thăng hạng trong arena

Một checkpoint mới chỉ thay incumbent nếu đạt score điểm ước lượng `> 0.50` và
cận dưới CI bootstrap ghép cặp 95% theo opening cũng `> 0.50`, trên ít nhất 100
opening hợp lệ, mỗi opening chơi hai ghế (tối thiểu 200 ván). Hai agent dùng
cùng opening; dùng bộ lọc khai cuộc công bằng đã định nghĩa trong evaluator.
Không điều chỉnh ngưỡng sau khi xem trận.

## 5. Round-robin và ngân sách

- Trận cuối: tối thiểu 100 ván mỗi cặp, opening ghép cặp đổi ghế, bộ opening
  chung cho mọi cặp; báo W/D/L, score, CI theo opening, lợi thế ghế và thời
  gian/nước chỉ đo local trên cùng máy.
- `fast`: AlphaBeta trần depth 4 / 100 ms so với MCTS 100 ms. `standard`:
  AlphaBeta trần depth 8 / 500 ms so với MCTS 500 ms. Đây là giới hạn tối đa;
  thời gian thực tế từng agent phải báo riêng.
- Colab dùng số bước hoặc số lượt mô phỏng cố định; tuyệt đối không so wall-time
  hay throughput giữa phiên/máy Colab.

## 6. Giao thức và lưu trữ

Mỗi lần chạy có tên duy nhất gồm `run_id`, seed và ngày. Không ghi đè file. Lưu
JSON/JSONL, metadata cấu hình đầy đủ, commit/băm mã nguồn, commit/hash solver,
checkpoint, môi trường và băm các đầu vào. Số trong báo cáo cuối phải được sinh
từ JSON bằng script; mọi bảng phải kèm nguồn và cỡ mẫu.



## 7. Hi?u ch?nh value head (??t tr??c ??nh gi? checkpoint m?i)

- Nh?n value l? d?u ?i?m exact t?t nh?t c?a Pons cho b?n ?i: -1/thua, 0/h?a, +1/th?ng.
- B?o value MSE, hi?u chu?n theo bin c?a prediction, ?? ch?nh x?c W/D/L v?i ng??ng c? ??nh: prediction > 1/3 l? th?ng; < -1/3 l? thua; c?n l?i l? h?a.
- So v?i baseline lu?n ?o?n l?p WDL ph? bi?n nh?t; CI bootstrap 95% theo v?n ngu?n cho MSE v? WDL accuracy.
- ??y l? ch?n ?o?n value head, kh?ng ph?i c?ng ch?n checkpoint v? kh?ng thay th? solver move-choice evaluation.
