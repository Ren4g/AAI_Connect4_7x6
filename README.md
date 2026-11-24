# Connect Four 7×6

Báo cáo tiến độ: [docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md).
Báo cáo tổng kết: [docs/report.md](docs/report.md) (tái tạo bằng `python scripts/render_final_report.py`).

Repo hiện có đủ mã cho tám cấp agent, benchmark local và pipeline huấn luyện CNN/RL/AlphaZero qua Google Colab. Mọi checkpoint Colab phải ingest local trước khi được đưa vào đánh giá.

## Kiến trúc

- `connect4/engine.py`: engine grid chuẩn 7×6.
- `connect4/engine_bit.py`: engine bitboard với API tương thích.
- `connect4/agents.py`: Random, Heuristic, AlphaBeta, MCTS, PolicyAgent và PUCT.
- `connect4/train_cnn.py`, `connect4/rl_train.py`, `connect4/selfplay.py`: CLI huấn luyện supervised CNN, policy-gradient RL và AlphaZero-inspired self-play.
- `connect4/solver.py`: adapter solver ngoài; solver chuẩn Pons nằm ở submodule `solver/connect4`.
- `evaluate.py`, `evaluate_solver.py`: round-robin trực tiếp và đánh giá exact trên test frozen.
- `benchmarks/`: dữ liệu trận đấu, ablation và replay JSONL.
- `results/`: kết quả solver, nhãn và log huấn luyện.

## Trạng thái thực nghiệm

Ba seed CNN supervised, một run RL và một run AlphaZero đã được ingest. CNN chưa đạt cổng tối thiểu đã định trước so với AlphaBeta depth 3. Value head chưa cho thấy lợi thế chắc chắn so với baseline lớp đa số. Arena self-play không qua cổng promotion; round-robin cuối đã chạy với agent cổ điển, CNN, RL và PUCT-AZ trên opening chung, hai ghế và fair filter.

- Kết quả học và CI sinh từ JSON: [docs/LEARNING_RESULTS.md](docs/LEARNING_RESULTS.md)
- Đường cong huấn luyện RL/AZ: [docs/training_process_trace.svg](docs/training_process_trace.svg) (diagnostic, không phải strength test)
- Đường cong độ sâu AlphaBeta: [docs/solver_depth_curve_learning.svg](docs/solver_depth_curve_learning.svg)
- Round-robin cuối: [docs/ROUND_ROBIN_FINAL.md](docs/ROUND_ROBIN_FINAL.md)
- Heatmap: [docs/round_robin_final.svg](docs/round_robin_final.svg)
- Tài nguyên local/Colab: [docs/RESOURCE_PROFILE.md](docs/RESOURCE_PROFILE.md)
- Tiêu chí: [docs/CRITERIA.md](docs/CRITERIA.md)
- Nhật ký: [docs/LOG.md](docs/LOG.md)
- Colab: [docs/COLAB.md](docs/COLAB.md)
- Dàn ý slide: [docs/SLIDE_OUTLINE.md](docs/SLIDE_OUTLINE.md)

## Cài đặt và kiểm tra

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Các job CNN/RL/AlphaZero cần PyTorch; cài theo phiên bản pin trong `requirements-colab.txt` hoặc môi trường GPU phù hợp. Lõi nghiệp vụ nằm trong package `connect4`, notebook chỉ điều phối CLI.

## Solver Pons

Clone kèm submodule:

```powershell
git clone --recurse-submodules https://github.com/Ren4g/AAI_Connect4_7x6.git
```

Windows cần `solver/connect4/c4solver.exe`. Nếu phải build lại:

```powershell
cd solver/connect4
g++ -std=c++11 -O3 -DNDEBUG -static -o c4solver.exe main.cpp Solver.cpp
```

Mỗi worker tạo solver process riêng và gửi nhiều vị trí qua stdin. Không chia sẻ cùng một tiến trình Pons giữa nhiều worker.

## Tạo lại báo cáo

Bảng solver/CNN/RL/AZ đọc từ JSON và các manifest Colab:

```powershell
python scripts/render_learning_report.py
```

Round-robin summary và heatmap đọc từ JSON, metadata và replay JSONL:

```powershell
python scripts/render_round_robin_report.py `
  --input benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json `
  --metadata benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json `
  --games benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.games.jsonl `
  --markdown docs/ROUND_ROBIN_FINAL.md --svg docs/round_robin_final.svg
```

Không ghi đè kết quả cũ: mỗi lần benchmark phải chọn output path mới.

## Đánh giá solver local

Ví dụ thêm checkpoint PUCT đã ingest vào test frozen:

```powershell
python evaluate_solver.py --puct-checkpoints AZ=results/colab/az_seed20261120_run1/checkpoint_policy.pt `
  --puct-simulations 500 --device cuda --seed 20261121 `
  --output results/solver_eval_new_seed20261121.json
```

## Round-robin local

`--ab-depth-extra` cho phép chấm hai độ sâu AlphaBeta cùng lúc; `--rl-checkpoint` thêm policy-gradient. Ví dụ cấu hình dùng ở giải cuối:

```powershell
python evaluate.py --games 100 --seed 20261121 --opening-plies 6 `
  --opening-filter fair --paired-openings --engine bitboard `
  --ab-depth 4 --ab-depth-extra 6 `
  --checkpoint results/colab/cnn_seed20261121_colab1/checkpoint_best.pt `
  --rl-checkpoint results/colab/rl_seed20261120_run1/checkpoint_policy.pt `
  --puct-checkpoint results/colab/az_seed20261120_run1/checkpoint_policy.pt `
  --puct-simulations 500 --device cuda `
  --output benchmarks/round_robin/round_robin_new.json
```

Paired openings dùng cùng opening cho hai ghế; fair filter loại thế đã có thắng ngay hoặc không có nước trả lời an toàn. Thời gian/throughput chỉ so sánh giữa run trên cùng máy local, không so số Colab giữa các phiên.

## Google Colab và ingest

Đọc [docs/COLAB.md](docs/COLAB.md). Notebook theo thứ tự: `colab/02_train_cnn.ipynb`, `colab/03_selfplay_az.ipynb`, `colab/04_rl_policy_gradient.ipynb`. Cùng `RUN_ID` sẽ resume từ checkpoint trên Drive. Tải gói `artifacts/<run_id>/` về `results/colab_inbox/<run_id>/`, rồi chạy:

```powershell
python scripts/ingest_colab_run.py results/colab_inbox/<run_id> `
  --dataset colab/data/solverlabels_20261120_seed20261120_v2
```

Chỉ sử dụng checkpoint từ `results/colab/<run_id>/` sau khi ingest thành công. Ingest kiểm checksum/manifest, chạy inference CPU và xác minh không có leakage với test frozen.

## Demo

Chạy giao diện chơi thử từ thư mục gốc repo:

```powershell
python -m connect4.gui
```

Giao diện hỗ trợ người chơi-với-AI và AI-với-AI, chọn các cấp agent đang có checkpoint, điều chỉnh depth/số mô phỏng và làm nổi bật bốn quân thắng. Mặc định, giao diện tự tìm các checkpoint đã có trong `results/`; có thể truyền đường dẫn riêng bằng `--checkpoint`, `--rl-checkpoint`, `--puct-checkpoint` hoặc `--az-checkpoint`. Ví dụ chọn GPU cho các agent mạng:

```powershell
python -m connect4.gui --device cuda --az-simulations 400
```

Kịch bản trình bày kết quả: [docs/DEMO.md](docs/DEMO.md). Không chạy benchmark dài trong buổi demo; dùng các report/JSONL đã lưu để trình bày và phát lại ván.
