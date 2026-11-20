# Chẩn đoán các ván CNN thua bằng solver

## Phương pháp

Đã rà 194 ván CNN thua trong log round-robin, gồm cả hai ván thua Random.
Tại mỗi lượt CNN, solver exact chấm mọi nước hợp lệ. Báo cáo tách:

- **Lỗi điểm exact:** nước CNN chọn có điểm solver thấp hơn phương án tốt nhất.
- **Lỗi WDL:** nước CNN chọn làm kết quả tối ưu giảm từ thắng/hòa xuống một mức
  thấp hơn. Đây là lỗi có khả năng đổi kết cục.
- **Fork:** sau nước CNN, có nước đáp trả của đối thủ khiến mọi nước phòng thủ
  tiếp theo của CNN đều cho đối thủ nước thắng tức thì.

Các nước chọn đồng hạng theo điểm solver không bị tính là lỗi. Ngoài ra, solver
đánh giá trạng thái sau sáu ply khai cuộc để nhận diện ván đã ở thế thua cưỡng
bức cho CNN trước quyết định đầu tiên của CNN.

## Kết quả

- 181/194 ván có ít nhất một nước CNN kém điểm exact tối ưu đầu tiên.
- Trong số đó, 145/194 có nước làm giảm WDL; 36 ván chỉ chọn đường kém hơn
  theo điểm exact nhưng vẫn cùng WDL.
- 13 ván không có nước CNN kém điểm exact ở bất kỳ lượt nào trong log.
- 97/194 ván đã là thế thua cưỡng bức cho CNN sau opening, theo solver.
- Ở các nước WDL-degrading đầu tiên, 14 tạo tình huống fork theo phép kiểm tra
  hai ply; 131 lỗi còn lại là positional/deeper. Trong 14 fork, 7 xuất hiện ở
  ván chưa bị solver xác định thua ngay từ opening; 7 xuất hiện sau khi thế cờ
  đã bị thua cưỡng bức.
- Không ghi nhận trường hợp CNN bỏ qua nước thắng tức thì của chính mình hay
  cho đối thủ thắng ngay ở nước kế tiếp. Điều này không loại trừ các tactic sâu
  hơn những gì phân loại fork ở trên kiểm tra.

## Hai ván CNN thua Random

Cả hai opening đã cho Random thế thắng cưỡng bức trước khi CNN ra quyết định.
Ở ván 52, mọi nước của CNN sau đó đều giữ cùng kết quả thua và không có nước
kém điểm exact so với tốt nhất. Ở ván 56, CNN chọn cột 3 (đánh số từ 1), trong
khi cột 5 kéo dài kết cục theo điểm exact (`-16` so với `-2`); cả hai vẫn là
thua cưỡng bức. Vì vậy, hai kết quả Random thắng này không chứng minh CNN bỏ sót
phòng thủ có thể cứu ván. Chúng phản ánh khai cuộc bất lợi được sinh ngẫu nhiên.

## Diễn giải

Kết quả không ủng hộ giả thuyết “CNN chủ yếu thua vì không thấy fork”: có fork
trong một phần nhỏ các lỗi WDL đầu tiên, còn đa số được solver xếp vào lỗi sâu
hơn/định vị. Nhãn solver có khả năng cải thiện việc xếp hạng chiến lược, nhưng
chẩn đoán cũng cho thấy cần tạo khai cuộc ít thiên lệch theo kết quả solver và
đưa tìm kiếm cục bộ vào agent nếu muốn xử lý tactic sâu. Không nên quy hai ván
thua Random cho chất lượng CNN trước khi lọc các opening đã thua cưỡng bức.

Dữ liệu chi tiết từng ván, lịch sử, bàn cờ, điểm từng cột và nước tối ưu nằm ở
`results/cnn_first_errors_seed20261122.json`.
