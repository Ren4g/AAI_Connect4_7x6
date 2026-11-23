# Addendum định trước — Pha B, 2026-11-21 11:44 Asia/Ho_Chi_Minh

Addendum này chốt cách tạo các nhánh ablation của Pha B sau khi hoàn tất chẩn đoán Pha A và **trước khi huấn luyện CNN trên bộ dữ liệu v3**. Không đổi metric, ngưỡng cải thiện, số seed tối thiểu hay giới hạn 3 giờ GPU của `CRITERIA_V2.md`. Các kết quả Pha A chỉ dùng để quyết định tiếp tục vì dữ liệu gốc có 1.500 mẫu; không dùng để chọn checkpoint hoặc ngưỡng.

## Cấu hình và đối chứng Pha B

- Tập Pons exact chính có 30.000 vị trí, cân bằng 10.000 mỗi giai đoạn. Validation giữ nguyên cho mọi nhánh; mọi lựa chọn dữ liệu huấn luyện dùng toàn bộ ván nguồn làm đơn vị.
- Đường cong dữ liệu dùng `--data-fraction` = 0,1 / 0,3 / 1,0. Chọn ván nguồn bằng thứ tự băm SHA256 của `fraction_seed:source_game`; do đó 10% là tập con của 30%, và 30% là tập con của 100%, không cắt rời vị trí cùng một ván.
- Nhánh soft giữ policy Pons softmax với temperature 2,0. Nhánh hard dùng one-hot cho một nước exact-optimal; khi có nhiều nước đồng hạng, chọn theo thứ tự giữa bàn cố định `3,2,4,1,5,0,6`. Value target và vị trí giữ nguyên.
- Nhánh teacher AlphaBeta depth 3 dùng policy one-hot theo nước AlphaBeta chọn trên đúng các vị trí của tập Pons 30k. Value target vẫn lấy dấu kết quả Pons exact để chỉ thay đổi nguồn nhãn policy.
- Mỗi cấu hình huấn luyện ba seed `20261121`, `20261122`, `20261123`; cùng kiến trúc, tối đa 30 epoch, batch 64, learning rate 0,001, value weight 0,5, early stopping patience 6. Checkpoint chọn theo validation loss.
- Tổng cộng tối đa 15 run CNN cho Pha B: 9 run cho ba cỡ dữ liệu, 3 run hard-policy và 3 run teacher AlphaBeta-3. Cấu hình 100% soft Pons là đường cơ sở dùng chung giữa hai ablation, không chạy trùng.
- Dùng cùng dataset v3 và cùng `DATA_FRACTION_SEED=20261121` cho mọi seed/cấu hình. Tên artifact phải chứa seed, phần trăm dữ liệu và nhánh target khi khác mặc định.

## Ngân sách thực thi

- Pha B được cấp tối đa 60 phút Colab GPU; Pha C tối đa 120 phút. Tổng cộng không vượt 180 phút, kể cả smoke hoặc resume. Nếu một pha chạm trần, dừng và báo cáo số run/seed thực tế; không chuyển phút đã dùng vượt giới hạn.
- Ước lượng ban đầu cho run CNN 30k: khoảng 1–3 phút/seed trên T4, ngoại suy tuyến tính từ log CNN 1.500 mẫu (630 bước trong 3,4 giây). Độ bất định cao; chỉ là kế hoạch tài nguyên, không phải metric agent.
- Khi B kết thúc, cấu hình cuối được khóa bằng validation rồi mới đánh giá một lần trên test đóng băng. Mọi cấu hình/seed đã thử và kết quả âm đều được liệt kê.
