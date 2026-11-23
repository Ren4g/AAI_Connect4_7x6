# Tiêu chí V2 — nâng chất lượng agent học Connect Four 7×6

**Đóng băng:** 2026-11-21, 10:37 Asia/Ho_Chi_Minh
**Hạn cứng:** 2026-11-22, 10:37 Asia/Ho_Chi_Minh
**Ngân sách:** tối đa 3 giờ GPU Colab cộng dồn; local có 12 lõi logic theo `os.cpu_count()`. Không gia hạn. Kết quả cũ không bị ghi đè.

Tài liệu này được chốt trước khi chạy thí nghiệm V2. Kiểm tra manifest, log và mã nguồn của những lần chạy cũ chỉ là kiểm toán dữ liệu, không phải thí nghiệm mới. Mọi lần chạy mới phải dùng seed, ngày và đường dẫn đầu ra mới.

## 1. Câu hỏi và giả thuyết

| Pha | Giả thuyết có thể bác bỏ | Đối chứng chính |
|---|---|---|
| A.1 | CNN v2 có ít vị trí huấn luyện hữu hiệu hoặc cấu hình bước/epoch không khớp giải thích một phần sức chơi thấp. | Manifest dataset, hash vị trí, train log và công thức số batch/epoch. |
| A.2/C | Self-play cải thiện quyết định so với cùng mạng khởi tạo dùng PUCT ở cùng ngân sách mô phỏng. | CNN seed 20261121 + PUCT so với checkpoint AZ, cùng bộ trạng thái, cùng số mô phỏng. |
| A.3 | Value head AZ dự đoán kết cục tốt hơn baseline lớp đa số và giá trị hằng 0 trên các trạng thái chưa dùng chọn checkpoint. | Nhãn Pons trên validation, phân tầng giai đoạn. |
| A.4 | Các phép cập nhật giá trị, policy target, mặt nạ luật, đối xứng và replay trong self-play đúng theo góc nhìn người chơi. | Unit tests và ví dụ bàn cờ có đáp án xác định; lỗi phải sửa trước khi huấn luyện mới. |
| A.5/D | RL có thể đóng góp ngoài safety filter và ngoài đối thủ Random. | Episode 0, policy đã huấn luyện; mỗi policy đo cả có/không safety filter trên cùng vị trí. |
| A.6 | Tốc độ sinh nhãn phụ thuộc mạnh vào ply và chế độ solver; benchmark trước có thể không đại diện cho toàn bộ phân bố dữ liệu. | Pons exact / weak (`-w`), ply-stratified; opening book chỉ khi file thật tồn tại. |
| B | Thêm vị trí solver đa dạng cải thiện CNN; soft target có thể giúp xếp hạng nước hơn target cứng. | Ablation kích thước dữ liệu; exact solver so với depth-3 teacher; soft so với hard. |

## 2. Tập dữ liệu, chia tập và metric

- Test đóng băng giữ nguyên: 1.500 vị trí, 500 mỗi giai đoạn, SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`, solver commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`. Không đưa vị trí test hoặc ảnh gương vào train/validation; ingest phải kiểm tra lại local.
- Chọn checkpoint và siêu tham số chỉ bằng validation chia theo ván nguồn; gom các trạng thái gương về cùng cụm trước khi chia. Test không dùng xếp hạng thử nghiệm V2.
- **Tỷ lệ nước tối ưu exact** là metric chính; **giữ W/D/L**, value MSE/WDL và score đối đầu là metric phụ. Score đối đầu = (thắng + 0,5 hòa)/số ván; không gọi là win rate.
- CI 95% bootstrap 10.000 lần theo cụm ván nguồn; hiệu agent là bootstrap ghép cặp trên cùng vị trí/cụm. Khi so phương pháp huấn luyện, resample cụm ván và seed lồng nhau; luôn báo thêm trung bình ± SD giữa seed, tách biệt CI bootstrap.
- “Chưa phân biệt được” nếu CI hiệu chứa 0. Tuyên bố cải thiện cần cận dưới CI ghép cặp > 0, trung bình hiệu giữa seed > 0 và ít nhất 2/3 seed có hiệu dương. Báo mọi seed, kể cả seed giảm chất lượng.
- Với mọi agent mạng: báo cùng checkpoint ở chế độ policy thuần và chế độ có tactical safety filter. Filter không được gộp vào điểm mạng.

## 3. Ngưỡng, số seed và giới hạn thí nghiệm

### CNN cấp 5

- Cổng tối thiểu đối chiếu lịch sử: cận dưới CI 95% ghép cặp của CNN − AlphaBeta depth 3 > 0 trên test đã khóa. Mốc 85% exact là mục tiêu phát triển, không phải cổng bắt buộc.
- Mọi cấu hình được tuyên bố mạnh hơn cần 3 seed. Chọn kiến trúc/checkpoint bằng validation; test một lần cho từng cấu hình cuối đã khóa và báo là test tái sử dụng trong dự án, không phải đánh giá mới độc lập.
- Tối đa hai vòng huấn luyện đầy đủ. Vòng kích thước dữ liệu giữ nguyên kiến trúc, optimizer và nhãn; các mức lấy theo N độc lập sau khi benchmark A.6: 10%, 30%, 100%. Vòng label ablation giữ cố định dữ liệu/kiến trúc: teacher depth 3 so với Pons exact; sau đó soft so với hard policy target nếu còn ngân sách. Mỗi nhánh cần 3 seed để kết luận, nếu hết GPU giữa chừng chỉ báo mô tả.
- Ngưỡng dữ liệu mục tiêu là ít nhất 30.000 vị trí độc lập; nếu A.6 cho thấy 30.000 exact labels vượt ngân sách local trong ngày, không sinh tập nhỏ hơn rồi gọi là thí nghiệm quy mô lớn: ghi estimate và dừng B.
- Số epoch tối đa 30 mỗi run. Early stopping chỉ theo validation, patience 6. Tổng GPU không vượt 3 giờ kể cả smoke, train và self-play.

### AlphaZero-inspired cấp 8

- Cải thiện được chứng minh bằng end checkpoint so với mạng khởi tạo dưới PUCT, cùng số mô phỏng và bộ trạng thái. Ablation số mô phỏng `{100, 200, 500, 1000}` và policy-only so với policy+PUCT trước tiên dùng validation; chỉ cấu hình khóa trước khi xem test được đưa ra làm kết quả test.
- Tuyên bố phương pháp tự-play tổng quát hơn cần 3 seed. Cấu hình khởi tạo CNN cố định cho seed sweep; seed chỉ đổi RNG self-play/optimization. Mỗi seed có ít nhất ba mốc định trước trong validation curve. Không có checkpoint trung gian thì không nội suy đường cong.
- Checkpoint arena chỉ promote nếu score > 0,5 và cận dưới CI ghép cặp theo opening > 0,5 trên ít nhất 100 opening hợp lệ, hai ghế. Ngưỡng không đổi sau khi xem kết quả.
- Thứ tự GPU: hoàn tất B đã chọn trước; C chỉ chạy phần còn lại trong tổng 3 giờ. Mọi smoke và train đều tính vào trần này. Không vượt trần bằng nhiều Colab session.

### RL cấp 6

- Chỉ tiến hành nếu Pha A.5 cho thấy khác biệt có thể kiểm định được và GPU còn sau B/C. Cần 3 seed cho kết luận huấn luyện; đối thủ gồm Random và ít nhất Heuristic hoặc AlphaBeta depth thấp. Nếu không đủ ngân sách, chỉ giữ chẩn đoán episode 0/filter và ghi RL mới là chưa đạt yêu cầu.

## 4. Quy tắc dừng theo pha

| Pha | Dừng/chuyển tiếp khi |
|---|---|
| A | Hoàn tất các audit/đo local trong giới hạn 8 giờ wall hoặc hết hạn toàn nhiệm vụ; ghi nguyên nhân khả dĩ và dữ liệu chứng minh. Không dùng GPU ở A. |
| B | Hoàn tất tối đa hai vòng hoặc hết tổng 3 giờ Colab; dừng sớm khi nguồn nhãn không sạch, overlap test khác 0, checksum sai, hay estimate 30k vị trí vượt trần ngày. |
| C | Chỉ chạy sau A.2/A.4. Dừng nếu ba mốc validation liên tiếp không tăng so với PUCT khởi tạo hoặc hết giờ GPU. Không suy diễn xu hướng từ một seed. |
| D | Cắt trước tiên; chỉ chạy nếu còn ít nhất 30 phút GPU sau B/C và mọi pipeline/ingest đã sẵn sàng. |
| Final | Không đổi checkpoint theo test. Nếu không đủ 3 seed hoặc CI chứa 0, kết luận “chưa phân biệt được”; dừng ở đó. |

## 5. Kỷ luật thực thi

- Trước mỗi thí nghiệm: xác minh commit/source hash, seed, input checksum, split theo ván nguồn và đường dẫn output mới.
- Mỗi gói Colab phải ghi `run_id`, phase, source commit/hash, seed, config, Python/PyTorch/CUDA/GPU, checkpoint SHA256, frozen-test overlap, resume count, peak VRAM và start/end; hỗ trợ `--resume`, `--max-minutes`, `--smoke`. Ingest local kiểm tra manifest, checksum, inference và leakage trước khi cho phép đánh giá.
- Nếu sửa tiêu chí sau khi có kết quả, không thay tài liệu đã khóa; tạo `CRITERIA_V2_POSTHOC.md` kèm ngày/lý do và không gọi phân tích đó là xác nhận định trước.
- Giới hạn GPU do người dùng chạy trên Colab; mã/notebook chỉ hướng dẫn tối đa 3 giờ cộng dồn. Tốc độ so sánh agent chỉ đo local cố định. Mọi số đưa vào báo cáo được sinh tự động từ JSON.


## Phase C amendment: V2 controlled initialization comparison (2026-11-21, prospective)

This amendment is registered after the Phase A diagnosis and before running this revised Phase C. It supersedes only the original Phase C configuration for the two new runs below; it does not revise or reinterpret any prior result. The frozen test set remains unavailable for checkpoint selection. The cause motivating the change is the weak Phase A result for the historical AZ policy: PUCT-100 from the AZ checkpoint was 66.5% exact-optimal on clean validation, versus 76.7% for the initial supervised CNN under the same PUCT-100 setting (paired difference -10.2 percentage points, 95% CI -16.5 to -4.0).

- Research question: at a fixed 400 simulations per move and 40 self-play iterations, how do the validation learning curves differ between a validation-selected CNN warm start and random initialization? This tests initialization under one fixed compute configuration; it does not by itself establish general superiority of either training method.
- Run 1: `az_v2_warm_seed20261121`, seed 20261121, initialize from the best Phase B CNN checkpoint selected using validation. Run 2: `az_v2_scratch_seed20261122`, seed 20261122, initialize from random weights. Each run is limited to 60 GPU minutes; Phase C is limited to 120 GPU minutes total.
- Fixed configuration per run: 40 iterations; 32 games per iteration; 16 parallel games; 400 simulations per move; 200 training updates per iteration; batch size 128; snapshots at iterations 0, 5, 10, 15, 20, 25, 30, 35, and 40. Each resumable segment is limited to 30 minutes and must retain the same run ID and configuration.
- The notebook persists a Drive budget ledger. It reserves the requested segment before launch and reconciles actual process time on normal completion. If a runtime is interrupted before reconciliation, the reservation remains charged conservatively. Do not delete or reset the ledger. Run the two jobs sequentially.
- Primary analysis: compare each snapshot to that run's iteration-0 model on the same validation positions, using 400-simulation PUCT, exact-optimal action rate, WDL preservation, and source-game cluster bootstrap 95% CIs. Report replay-overlap exclusions per checkpoint. The two runs have one seed each, so report them as a controlled case comparison; do not claim seed-generalized superiority or use them as a substitute for the three-seed criterion.
- The iteration-40 checkpoints are fixed by protocol. Evaluate those checkpoints once on the frozen test after both runs and validation analyses are complete; do not choose an iteration or initialization from test results.
- Stop each run at 60 minutes, the phase at 120 minutes, or iteration 40, whichever comes first. An incomplete curve is reported as incomplete; do not extend the allocation. The two-run comparison is exploratory because it does not satisfy the three-seed method-comparison requirement.

See also `docs/CRITERIA_V2_POSTHOC.md` for the dated Phase C amendment and its explicit limits as a post-diagnostic protocol. The original criteria remain unchanged as the record of the initial preregistration.
