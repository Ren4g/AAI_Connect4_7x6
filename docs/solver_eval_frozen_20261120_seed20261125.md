# Đánh giá agent bằng solver exact

- Run: `solver_eval_frozen_20261120_seed20261125`; seed bootstrap: `20261125`.
- Tập đóng băng: 1500 vị trí; 1498 lớp trạng thái đối xứng.
- SHA256 test: `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`.
- Solver: commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`, binary SHA256 `13903490c56862be0eb02bad67087bd77035392786978b04cc5455e4ab5a45fe`.
- Thời gian solver local: 139.6s với 8 worker; không so thời gian này với Colab.
- CI 95% bootstrap theo ván nguồn; các cặp dùng đúng cùng vị trí.

## Kết quả tổng thể

| Agent | Nước tối ưu | CI 95% | Giữ W/D/L | CI 95% | giây/nước local | node/nước |
|---|---:|---:|---:|---:|---:|---:|
| Random | 35.3% | [32.9; 37.7] | 70.6% | [68.2; 72.9] | 0.000003 | 0.0 |
| Heuristic | 73.5% | [71.4; 75.7] | 88.3% | [86.7; 89.8] | 0.001206 | 0.0 |
| AlphaBeta-depth1 | 70.3% | [68.1; 72.5] | 86.2% | [84.4; 87.9] | 0.000526 | 5.7 |
| AlphaBeta-depth2 | 75.6% | [73.4; 77.7] | 90.5% | [89.1; 92.0] | 0.001681 | 23.5 |
| AlphaBeta-depth3 | 77.2% | [75.0; 79.4] | 90.5% | [89.1; 92.0] | 0.007210 | 100.1 |
| AlphaBeta-depth4 | 80.5% | [78.5; 82.4] | 91.7% | [90.2; 93.2] | 0.020893 | 313.9 |
| AlphaBeta-depth5 | 83.1% | [81.2; 84.9] | 93.3% | [91.9; 94.5] | 0.077208 | 1136.6 |
| AlphaBeta-depth6 | 81.5% | [79.5; 83.4] | 91.5% | [90.0; 92.9] | 0.228606 | 3543.0 |
| MCTS-500 | 75.3% | [73.1; 77.5] | 91.5% | [90.0; 92.9] | 0.037277 | 0.0 |

## Nước tối ưu theo giai đoạn

| Agent | Khai cuộc n=500 | Trung cuộc n=500 | Tàn cuộc n=500 |
|---|---:|---:|---:|
| Random | 22.8% [19.1; 26.5] | 30.8% [26.9; 34.8] | 52.2% [47.8; 56.6] |
| Heuristic | 65.8% [61.8; 69.9] | 70.2% [66.1; 74.2] | 84.6% [81.3; 87.8] |
| AlphaBeta-depth1 | 65.0% [60.9; 69.0] | 67.8% [63.5; 72.0] | 78.2% [74.4; 81.9] |
| AlphaBeta-depth2 | 62.8% [58.6; 66.9] | 72.8% [68.9; 76.6] | 91.2% [88.5; 93.7] |
| AlphaBeta-depth3 | 65.4% [61.1; 69.5] | 73.0% [68.9; 77.0] | 93.2% [91.0; 95.2] |
| AlphaBeta-depth4 | 68.0% [63.9; 71.9] | 77.8% [74.1; 81.3] | 95.6% [93.8; 97.2] |
| AlphaBeta-depth5 | 69.4% [65.3; 73.3] | 82.8% [79.4; 86.1] | 97.0% [95.4; 98.4] |
| AlphaBeta-depth6 | 66.8% [62.5; 71.2] | 80.0% [76.5; 83.4] | 97.6% [96.2; 98.8] |
| MCTS-500 | 64.0% [59.7; 68.2] | 71.2% [67.3; 75.1] | 90.6% [87.8; 93.2] |

## Vị trí có quyết định và giá trị lý thuyết

Vị trí có quyết định được định nghĩa trước là vị trí có ít nhất một nước hợp lệ làm đổi dấu W/D/L so với nước tối ưu.
Tập được chia theo dấu điểm tốt nhất của solver cho bên đi.

| Agent | Có quyết định | Giá trị thắng | Giá trị hòa | Giá trị thua |
|---|---:|---:|---:|---:|
| Random | 726 | 29.3% (n=781) | 39.5% (n=147) | 42.3% (n=572) |
| Heuristic | 726 | 69.4% (n=781) | 70.7% (n=147) | 79.9% (n=572) |
| AlphaBeta-depth1 | 726 | 65.8% (n=781) | 69.4% (n=147) | 76.7% (n=572) |
| AlphaBeta-depth2 | 726 | 71.6% (n=781) | 74.1% (n=147) | 81.5% (n=572) |
| AlphaBeta-depth3 | 726 | 76.2% (n=781) | 69.4% (n=147) | 80.6% (n=572) |
| AlphaBeta-depth4 | 726 | 78.2% (n=781) | 75.5% (n=147) | 84.8% (n=572) |
| AlphaBeta-depth5 | 726 | 82.3% (n=781) | 78.2% (n=147) | 85.3% (n=572) |
| AlphaBeta-depth6 | 726 | 81.0% (n=781) | 70.1% (n=147) | 85.0% (n=572) |
| MCTS-500 | 726 | 75.9% (n=781) | 72.8% (n=147) | 75.0% (n=572) |

## Chênh lệch ghép cặp với AlphaBeta depth 3

Giá trị là agent đang xét trừ AlphaBeta depth 3; CI bootstrap theo cùng ván nguồn.

| Agent | Nước tối ưu, tổng thể | Khai cuộc | Trung cuộc | Tàn cuộc |
|---|---:|---:|---:|---:|
| Random | [-44.9; -38.9] | [-48.4; -36.8] | [-47.4; -37.0] | [-45.5; -36.5] |
| Heuristic | [-6.1; -1.3] | [-4.1; 4.8] | [-7.0; 1.4] | [-12.0; -5.3] |
| AlphaBeta-depth1 | [-9.5; -4.3] | [-4.9; 4.1] | [-9.6; -0.8] | [-18.8; -11.2] |
| AlphaBeta-depth2 | [-3.8; 0.5] | [-7.2; 2.0] | [-4.2; 3.7] | [-4.6; 0.6] |
| AlphaBeta-depth3 | [0.0; 0.0] | [0.0; 0.0] | [0.0; 0.0] | [0.0; 0.0] |
| AlphaBeta-depth4 | [1.3; 5.3] | [-2.0; 7.3] | [1.2; 8.6] | [0.4; 4.5] |
| AlphaBeta-depth5 | [3.9; 7.8] | [-0.2; 8.2] | [6.2; 13.5] | [1.8; 5.8] |
| AlphaBeta-depth6 | [2.2; 6.3] | [-3.4; 6.3] | [3.2; 10.9] | [2.4; 6.5] |
| MCTS-500 | [-4.2; 0.3] | [-5.9; 3.1] | [-6.0; 2.4] | [-5.8; 0.4] |

## Diễn giải

Tỷ lệ exact-optimal không tăng đơn điệu theo độ sâu: AlphaBeta depth 5 cao hơn depth 6 trên tổng thể. Đây là kết quả của tìm kiếm hữu hạn cùng evaluator heuristic; độ sâu lớn hơn không tự đảm bảo chọn nước exact hơn.
CNN chưa được đưa vào lần chạy này vì checkpoint hiện có trùng một phần tập đóng băng trong train/validation. Chỉ đánh giá CNN sau khi ingest checkpoint được huấn luyện từ gói dữ liệu sạch.
