# Báo cáo tiến độ hệ thống Connect Four 7×6

> **T?nh tr?ng hi?n t?i (20-11-2026):** ?? c? code cho ?? t?m c?p agent. C?p 6 RL, c?p 7 PUCT v? c?p 8 self-play ?? ch?y smoke/resume local; ??y l? x?c nh?n ???ng ?ng, ch?a ph?i ??nh gi? s?c ch?i. ?? chu?n b? ba notebook Colab, source bundle c? checksum, ingest local v? arena gh?p c?p. Ch?a nh?n artifact hu?n luy?n Colab n?n ch?a c? k?t qu? CNN/self-play/RL ch?nh th?c, ch?a ch?y arena ?? 100 opening ho?c round-robin cu?i.
>
> Baseline solver ??ng b?ng g?n nh?t v?n l? k?t qu? ch?nh: AlphaBeta depth 4 ??t exact-optimal 80,5% [78,5; 82,4], depth 5 ??t 83,1% [81,2; 84,9]. M?t probe value head local m?t epoch ???c ghi ri?ng ? ph?n cu?i; kh?ng d?ng ch?n checkpoint. Xem [nh?t k? theo pha](LOG.md), [ti?u ch?](CRITERIA.md), [h??ng d?n Colab](COLAB.md), [d?n ? slide](SLIDE_OUTLINE.md) v? [demo](DEMO.md).

## Tóm tắt

Dự án hiện có agent cấp 1–5, benchmark ghép cặp và bộ đánh giá exact bằng Pons.
Các cấp RL, AlphaGo-inspired và AlphaZero-inspired chưa triển khai. Đánh giá
solver đóng băng trên 1.500 vị trí cho thấy AlphaBeta depth 5 đạt tỷ lệ nước tối
ưu 83,1% [81,2; 84,9], depth 4 đạt 80,5% [78,5; 82,4]. CNN chưa được đánh giá
lại trên tập này: checkpoint solver-label trước đây bị giao dữ liệu với test,
còn lịch sử vị trí của checkpoint gravity-aware không được lưu để xác nhận sạch.

## Phương pháp đánh giá

Các trận đối kháng dùng khai cuộc ghép cặp, đổi ghế và bộ lọc khai cuộc công
bằng. Match score = thắng + 0,5 hòa; CI lấy bootstrap theo opening. Đánh giá
solver dùng tập test đóng băng, phục hồi nguồn ván, chia theo giai đoạn và loại
trùng gương. CI bootstrap theo ván nguồn; phép so sánh cặp dùng cùng vị trí.
Thời gian agent chỉ đo local trên cùng máy. Các thời gian Colab sẽ chỉ mô tả tài
nguyên, không dùng để so sánh agent.

## Tiến độ thực nghiệm

Round-robin đã chạy lại có và không có bộ lọc khai cuộc; hồi quy depth cố định
xác nhận 1.000 ván giữ nguyên nước đi và kết quả. AlphaBeta depth 8 / 500 ms
đạt score 0,435 [0,345; 0,525] trước MCTS 500 ms trong 100 ván paired. AlphaBeta
depth 6 đạt score 0,845 trước Heuristic trong 100 ván. Ablation ba seed, mỗi
seed 10 ván, còn mang tính thăm dò.

Đánh giá solver đóng băng có 1.500 dòng, 1.498 lớp bàn cờ chuẩn hóa gương:

| Agent | Exact-optimal (CI 95%) | WDL-preserving (CI 95%) |
|---|---:|---:|
| Random | 35,3% [32,9; 37,7] | 70,6% [68,2; 72,9] |
| Heuristic | 73,5% [71,4; 75,7] | 88,3% [86,7; 89,8] |
| AlphaBeta depth 3 | 77,2% [75,0; 79,4] | 90,5% [89,1; 92,0] |
| AlphaBeta depth 4 | 80,5% [78,5; 82,4] | 91,7% [90,2; 93,2] |
| AlphaBeta depth 5 | 83,1% [81,2; 84,9] | 93,3% [91,9; 94,5] |
| AlphaBeta depth 6 | 81,5% [79,5; 83,4] | 91,5% [90,0; 92,9] |
| MCTS-500 | 75,3% [73,1; 77,5] | 91,5% [90,0; 92,9] |

Depth không cho kết quả đơn điệu: depth 5 vượt depth 6 trên mẫu này. Tập có
quyết định có 726 vị trí; các vị trí được chia theo giá trị lý thuyết của bên
đi: 781 thắng, 147 hòa và 572 thua. Báo cáo đầy đủ với CI từng giai đoạn, hiệu
ghép cặp với AB3 và biểu đồ được sinh từ JSON tại
[solver_eval_frozen_20261120_seed20261125.md](solver_eval_frozen_20261120_seed20261125.md).

### Chẩn đoán CNN

Solver đã phân tích 194 ván CNN thua. Có 181 ván xuất hiện ít nhất một nước CNN
kém điểm exact, 145 ván có nước làm giảm W/D/L, và 97 opening đã là thế thua
cưỡng bức cho CNN. Cả hai ván CNN thua Random thuộc nhóm này. Phân tích chi tiết
ở [CNN_LOSS_ANALYSIS.md](CNN_LOSS_ANALYSIS.md).

### Huấn luyện solver-label hiện có

Checkpoint `connect4_cnn_solver_seed20261201.pt` được huấn luyện với policy
soft-target và value target từ solver; chạy một seed mất khoảng 776 giây trên
RTX 3050. So sánh nội bộ 898 vị trí cho thấy 73,9% exact-optimal so với 70,9%
của CNN cũ. Tuy nhiên, kiểm toán sau đó phát hiện 129 khóa trong train và 25
khóa trong validation trùng tập đóng băng 1.500 vị trí. Vì vậy checkpoint và
kết quả trên 898 vị trí được gắn nhãn **legacy/leaky**, không dùng làm bằng
chứng held-out hay để qua cổng self-play. Lịch sử vị trí của checkpoint
gravity-aware cũng không được lưu đủ để xác nhận disjointness.

## Kết luận và công việc tiếp theo

Chưa có CNN nào được chứng nhận sạch theo test đóng băng. Cần tạo gói dữ liệu
train/validation loại toàn bộ test hash và source game, dựng job Colab resumable,
ingest checkpoint mới, rồi mới so sánh CNN với AB3 trên đúng 1.500 vị trí. Value
head đã có trong mạng nhưng chưa tham gia chọn nước; PUCT, self-play và RL chưa
được triển khai.

## Tệp kết quả

- Tiêu chí: [CRITERIA.md](CRITERIA.md)
- Nhật ký theo pha: [LOG.md](LOG.md)
- Phân tích lỗi CNN: [CNN_LOSS_ANALYSIS.md](CNN_LOSS_ANALYSIS.md)
- Tiêu chí CNN/self-play: [CNN_PROMOTION_CRITERIA.md](CNN_PROMOTION_CRITERIA.md)
- JSON đánh giá: `results/solver_eval_final_20261120_seed20261125.json`
- Solver manifest: `results/solver_eval_seed20261122.manifest.json`
# Cập nhật triển khai: pipeline CNN và PUCT

Đến 20-11-2026, bước tiếp theo tập trung vào tính hợp lệ của dữ liệu và khả năng
tiếp tục thí nghiệm sau khi Colab ngắt phiên. Bộ dữ liệu v2 có 1.500 vị trí được
chấm bằng Pons; 1.284 mẫu train thuộc 1.023 ván nguồn, 216 mẫu validation thuộc
181 ván nguồn. Kiểm tra khóa canonical có xét lật gương cho thấy không có giao
với tập test đóng băng. Mỗi worker dùng một tiến trình `c4solver.exe` lâu dài và
gửi nhiều vị trí trên stdin, nhờ vậy tránh chi phí tạo tiến trình cho từng vị trí.

Trainer mới huấn luyện policy và value, chọn checkpoint theo validation và lưu
đầy đủ optimizer, vị trí batch, thứ tự mẫu, RNG Python/NumPy/PyTorch/CUDA. Smoke
RTX 3050 hoàn tất hai epoch; một lần dừng giới hạn thời gian giữa epoch được resume
thành công. Đo local đủ một epoch trên toàn bộ train mất 0,65 giây thời gian tính
thuần (21 batch); đây chỉ là ước lượng thô cho kế hoạch Colab, không phải kết quả
so sánh hiệu năng agent.

Notebook Colab chạy CLI package, đọc dữ liệu và lưu checkpoint trực tiếp lên
Drive. Script ingest bắt buộc xác minh checksum, cấu hình/dataset, SHA của mã
nguồn, loại trừ test lại trên local và chạy suy luận CPU trước khi cho phép đánh
giá. Chưa có artifact Colab được trả về, do đó chưa có kết quả CNN mới.

PUCT đã được hiện thực ở mức suy luận: policy head làm prior, value head đánh giá
lá và MCTS lan truyền giá trị theo góc nhìn người chơi. Nó có thể được chấm trên
tập solver bằng số simulation cố định sau khi có checkpoint hợp lệ. Cấp 8
(self-play AlphaZero) và cấp 6 (RL) chưa được triển khai; cũng chưa có benchmark
PUCT hay round-robin cuối. Hiện kiểm thử đạt 39 passed, 1 skipped; notebook hợp lệ
theo nbformat.



## C?p nh?t sau: self-play c?p 8, RL c?p 6 v? arena

Pipeline c?p 8 hi?n ?? c?: PUCT batched qua nhi?u v?n ??ng th?i; Dirichlet noise ? g?c; nhi?t ?? theo s? ply; policy target t? visits; value target t? k?t qu? v?n; augmentation g??ng v? replay buffer. Checkpoint c? optimizer, replay buffer, c?c v?n c?n dang d?, b? ??m b??c v? RNG. Smoke RTX 3050 ho?n t?t hai v?n/two updates; ?? d?ng gi?a v?n v? resume, sau ?? ho?n th?nh ??ng iteration. Ingest th? tr?n m?t run nh? x?c nh?n checkpoint ch?y CPU v? replay kh?ng giao test. ??y ch? l? ki?m tra ???ng ?ng, kh?ng ph?i k?t qu? s?c ch?i.

C?p 6 c? trainer REINFORCE/actor-critic nh? ??u Random, ??i b?n ?i tr??c, d?ng entropy regularization v? value baseline. M?u ? v? tr? thu?c test b? lo?i; hash tr?ng th?i ?? d?ng ???c l?u ?? ingest ki?m l?i. Smoke GPU hai episode v? ingest probe ??u qua. ??y l? nh?nh th?m d?; ch?a c? ???ng cong h?c ?? d?i ho?c CI.

Arena local m?i so candidate/incumbent PUCT tr?n opening 6 ply chung, l?c khai cu?c c? th?ng t?c th? ho?c kh?ng c? n??c an to?n, ??nh hai gh? v? ?p d?ng c?ng score c?ng bootstrap theo opening ?? ??nh tr??c. Smoke arena hai opening ch? x?c minh lu?ng; kh?ng c? promotion v? kh?ng ???c d?ng l?m k?t qu? s?c ch?i. `evaluate.py` c?ng nh?n checkpoint PUCT cho round-robin cu?i.

Notebook Colab hi?n c? job ri?ng cho CNN c? gi?m s?t, self-play c?p 8 v? RL c?p 6. Kh?ng c? artifact Colab th?t n?o ???c tr? v?; v? v?y hi?n ch?a th? b?o c?o k?t qu? hu?n luy?n hay ??nh gi? checkpoint Colab. C?n c?n ng??i d?ng ch?y c?c notebook, ingest artifact, benchmark PUCT, ki?m tra promotion theo t?i thi?u 100 opening, ch?y round-robin cu?i v? t?ng h?p b?o c?o/slide cu?i. C?c s? baseline ? tr?n kh?ng thay th? ???c c?c b??c n?y.


## Ki?m tra value head (development probe)

M?t checkpoint CNN hu?n luy?n m?t epoch tr?n RTX 3050 ?? ???c d?ng ?? x?c minh
???ng ?o value head. Tr?n t?p test ??ng b?ng, value MSE l? 0,882 [0,844; 0,920]
v? WDL accuracy theo ng??ng ?? ??t tr??c l? 44,1% [41,5; 46,6], d??i baseline
?o?n l?p ph? bi?n nh?t 52,1%. ??y l? probe k? thu?t m?t seed/m?t epoch, kh?ng d?ng
ch?n checkpoint v? kh?ng ??i di?n cho k?t qu? hu?n luy?n Colab. Ph?n ph?i d? ?o?n
nghi?ng d??ng; c?n ??nh gi? l?i checkpoint m?i sau khi ho?n t?t hu?n luy?n s?ch.
Ngu?n JSON: `results/value_head_probe_local_cnn_seed71_20261120.json`.
