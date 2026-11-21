# Đo tài nguyên agent

Ngày đo local: 2026-11-21T10:15:32.073061+07:00; commit nền: `1f0413ddfb70eb6379ab3bd88872d588b40960c6`; SHA256 công cụ: `f964dbbb40a44510d1e9066883d47e680a6860b18ad7b6147fd96421652144d6`.

## RAM tiến trình local

Mỗi agent chạy riêng trong một cây tiến trình qua một ván đầy đủ trên bitboard, đấu Heuristic. Working Set cộng trên tiến trình launcher và các tiến trình con được lấy mẫu mỗi 100 ms sau khi agent khởi tạo; peak gồm runtime, model và agent. `peak_increment` là phần tăng thêm so với giá trị ngay trước ván. Windows có thể đếm trùng trang chia sẻ giữa các tiến trình trong cây, nên tổng Working Set là giới hạn trên gần đúng. Một ván cho mỗi agent là phép đo mô tả, không phải benchmark sức chơi.

| Agent | RSS trước ván (MiB) | Peak RSS (MiB) | Phần tăng (MiB) | Số quyết định | Thời gian ván (s) |
|---|---:|---:|---:|---:|---:|
| Random | 24.12 | 24.12 | 0.00 | 4 | 0.112 |
| Heuristic | 24.56 | 24.56 | 0.00 | 4 | 0.110 |
| AlphaBeta-depth4 | 24.56 | 24.57 | 0.01 | 7 | 0.108 |
| MCTS-500 | 24.14 | 25.40 | 1.26 | 8 | 0.328 |
| AlphaBeta-depth6 | 24.09 | 24.10 | 0.01 | 10 | 0.438 |
| CNN-Policy | 487.09 | 487.09 | 0.00 | 9 | 0.439 |
| RL-Policy | 486.21 | 486.21 | 0.00 | 5 | 0.440 |
| PUCT-500 | 483.08 | 491.82 | 8.74 | 12 | 2.100 |

## Tài nguyên run Colab

RAM là giá trị máy khả dụng ghi trong manifest khi job chạy; VRAM là peak allocated ghi bởi PyTorch khi manifest có trường tương ứng. Thiếu VRAM peak được ghi là chưa được đo, không suy ra từ model size.

| Run | Nhánh | GPU | RAM máy (MiB) | Peak VRAM allocated (MiB) |
|---|---|---|---:|---:|
| `az_seed20261120_run1` | alphazero_selfplay | Tesla T4 | 12976 | không đo/không ghi |
| `cnn_seed20261120_colab1` | cnn_supervised_solver_labels | Tesla T4 | 12976 | 32.69 |
| `cnn_seed20261121_colab1` | cnn_supervised_solver_labels | Tesla T4 | 12976 | 32.69 |
| `cnn_seed20261122_colab1` | cnn_supervised_solver_labels | Tesla T4 | 12976 | 32.69 |
| `rl_seed20261120_run1` | rl_policy_gradient | Tesla T4 | 12976 | không đo/không ghi |

Số liệu local và Colab thuộc hai máy khác nhau; chỉ RAM/VRAM Colab phản ánh lúc huấn luyện, còn RSS local phản ánh tiến trình suy luận/đấu ván trên máy cố định của dự án.
