# Tiêu chí đánh giá CNN trước khi chuyển sang self-play

## Mục tiêu

Không dùng “85% trước Heuristic” làm tiêu chí huấn luyện vì cách nói này không
xác định đó là tỷ lệ thắng, win rate hay score. Mục tiêu chính của CNN mới là
bắt chước được nước tốt theo solver trên các vị trí chưa xuất hiện trong train.
Đối đầu Heuristic chỉ là kiểm tra hành vi bổ sung.

## Tập solver held-out

- Chia theo **ván nguồn** trước khi sinh/nhân bản mẫu; không để vị trí hoặc ảnh
  lật gương của cùng một ván lọt qua nhiều split.
- Chỉ tăng cường dữ liệu gương ở train. Validation và test giữ một đại diện cho
  mỗi lớp đối xứng.
- Tập test tối thiểu 1.500 vị trí: 500 khai cuộc (ply 7–10), 500 trung cuộc
  (ply 11–24), 500 tàn cuộc (ply 25–41). Dùng cùng một tập test cho mọi checkpoint.
- Vị trí test phải được kiểm tra trùng cả lịch sử nước đi lẫn trạng thái bàn cờ
  đã chuẩn hóa gương với train/validation.

## Cổng tối thiểu và mục tiêu

Do AlphaBeta depth 4 hiện đạt 80,5% trên mẫu chẩn đoán, 85% là mục tiêu phát triển chứ không phải điều kiện chặn chuyển giao. Checkpoint chỉ qua cổng tối thiểu nếu thỏa các mục sau trên tập test:

1. Non-inferiority với AlphaBeta depth 3: cận dưới bootstrap 95% theo cụm của `CNN optimal-rate − AB3 optimal-rate` phải lớn hơn −5 điểm phần trăm tổng thể, và lớn hơn −10 điểm ở từng giai đoạn.
2. WDL-preserving của CNN không thấp hơn AB3 quá 3 điểm phần trăm tổng thể.
3. Không có lỗi nghiêm trọng về luật/inference, và cùng một checkpoint phải được đánh giá trên toàn bộ test held-out.

Mục tiêu phát triển tiếp theo là 85% nước exact tối ưu tổng thể, ít nhất 75% mỗi giai đoạn và 95% giữ WDL tổng thể. Mục tiêu không phải điều kiện để bắt đầu RL. Sau khi qua cổng tối thiểu, RL có thể chạy với checkpoint CNN làm policy prior, trong khi benchmark solver tiếp tục theo dõi tiến bộ.

Bootstrap phải gom cụm theo ván nguồn, bao gồm mọi trạng thái và ảnh lật gương của ván đó. Cần ít nhất ba seed huấn luyện để kết quả không phụ thuộc một lần khởi tạo.

## Đối đầu Heuristic

Nếu cần nêu kết quả game, dùng **score** `win + 0,5 × draw`, không gọi đó là
win rate. Chạy ít nhất 200 ván (100 khai cuộc hợp lệ, mỗi khai cuộc đánh hai ghế),
dùng chung cùng tập khai cuộc cho CNN và baseline, công bố W/D/L, score và khoảng
tin cậy 95%. Không yêu cầu 85% score trước Heuristic; đây không còn là promotion
gate vì nhãn solver cung cấp tiêu chí trực tiếp, khách quan hơn.

## Kết quả vòng huấn luyện solver đầu tiên

Checkpoint mới `checkpoints/connect4_cnn_solver_seed20261201.pt` được huấn luyện
với policy soft-target từ điểm solver và value target W/D/L. Dữ liệu lấy theo
ván nguồn, chia train/validation/test, loại trùng đối xứng trước khi chia và chỉ
augment gương trên train. Tập test có 898 vị trí còn lại sau lọc giao với
train/validation (298 khai cuộc, 300 trung cuộc, 300 tàn cuộc; 640 ván nguồn).
Các vị trí test và train/validation không trùng khóa trạng thái đã chuẩn hóa.

Ba agent được chạy trên cùng 898 vị trí; khoảng tin cậy bootstrap 95% gom theo
ván nguồn:

| Agent | Tổng | Khai cuộc | Trung cuộc | Tàn cuộc |
|---|---:|---:|---:|---:|
| CNN cũ | 70,9% [67,9; 73,9] | 57,7% | 69,7% | 85,3% |
| CNN nhãn solver | 73,9% [71,0; 76,7] | 66,4% | 68,7% | 86,7% |
| AlphaBeta depth 3 | 80,7% [78,3; 83,1] | 62,4% | 84,0% | 95,7% |

CNN mới tăng 3,0 điểm phần trăm so với CNN cũ; bootstrap cặp theo ván cho
chênh lệch [0,0; 6,2] điểm. Tăng trưởng tập trung ở khai cuộc (+8,7 điểm), trong
khi trung cuộc giảm nhẹ và tàn cuộc tăng ít. So với AB3, chênh lệch CNN mới trừ
AB3 là −6,8 điểm; khoảng bootstrap cặp 95% là [−10,2; −3,4]. Theo giai đoạn,
khoảng này là [−2,7; +10,7] ở khai cuộc, [−21,0; −9,7] ở trung cuộc và
[−13,0; −5,3] ở tàn cuộc. Vì vậy CNN **chưa qua cổng non-inferiority** cả tổng
thể lẫn ở trung cuộc/tàn cuộc; điều kiện WDL-preserving cũng chưa được xác
nhận trên đúng tập này. Kiểm toán sau đó phát hiện tập 898 vị trí này giao với
tập đóng băng 1.500 vị trí ở 129 khóa thuộc train và 25 khóa thuộc validation.
Vì vậy cả checkpoint và kết quả 898 vị trí được gắn nhãn **legacy/leaky** và
không dùng làm bằng chứng held-out hoặc làm cổng self-play. Chưa nên chuyển
checkpoint này thành policy ưu tiên cho self-play theo cổng đã đề xuất.

Đây mới là một seed huấn luyện và test 898 vị trí, thấp hơn quy mô mục tiêu
1.500; thêm vào đó, tập test bị giao với test đóng băng như nêu trên. Các khoảng
tin cậy mô tả độ bất định theo ván trong tập legacy này, không thay thế các lần
huấn luyện độc lập. Kết quả chi tiết nằm ở
`results/solver_checkpoint_comparison_seed20261201.json`; lịch sử huấn luyện và
nhãn solver ở `results/training_solver_seed20261201.json` và
`results/solver_labels_seed20261201.json`.
