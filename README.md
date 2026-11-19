# Connect4 7x6

Bộ khung triển khai các mục 1–5 trong `Proposal.docx`:

- `connect4/engine.py`: engine chuẩn 7×6, API `reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`.
- `connect4/engine_bit.py`: engine bitboard dùng chung API, kết quả, lượt chơi và kiểu nước đi với engine ma trận.
- `connect4/agents.py`: Random (cấp 1), Heuristic (cấp 2), Alpha-Beta (cấp 3), MCTS-UCT (cấp 4), PolicyAgent (cấp 5).
- `connect4/training.py`: sinh ván tự đấu, nhãn policy/value, loại trùng, augmentation đối xứng, xuất CSV.
- `connect4/model.py`: CNN policy/value hai đầu ra theo thiết kế mục 4 (PyTorch nạp khi sử dụng).
- `connect4/experiments.py`: đấu round-robin, chia đều quyền đi trước, thu thời gian và kết quả.
- `evaluate.py`: chạy benchmark và xuất báo cáo JSON.
- `ablation.py`: so sánh pruning/move ordering và các mức simulation MCTS.
- `connect4/solver.py`: adapter đối chiếu nước đi với solver Connect Four bên ngoài.
- `tests/test_engine.py`: test thắng ngang/dọc/chéo, clone và nước đi không hợp lệ.

Chạy test:

```powershell
python -m pytest
```

## Giao diện

Chạy giao diện Tkinter:

```powershell
python -m connect4.gui
```

Giao diện hỗ trợ `human_vs_ai` và `ai_vs_ai`, với Random, Heuristic,
Alpha-Beta và MCTS.

Có thể chọn engine bitboard cho demo:

```powershell
python -m connect4.gui --engine bitboard
```

Nạp CNN đã huấn luyện vào giao diện (thêm lựa chọn `CNN-Policy`):

```powershell
python -m connect4.gui --engine bitboard --checkpoint checkpoints/connect4_cnn_seed20261119.pt --device cuda
```

## Huấn luyện CNN

PyTorch là dependency tùy chọn:

```powershell
python -m pip install torch
python train.py --games 100 --epochs 10 --engine bitboard
```

Hoặc cài các dependency của dự án bằng `python -m pip install -r requirements.txt`.
Để tái lập lần huấn luyện đã lưu trong repository, dùng seed `20261119` và
cấu hình ở `results/training_seed20261119.json`:

```powershell
python train.py --games 4000 --epochs 12 --batch-size 256 --learning-rate 0.001 --epsilon 0.05 --seed 20261119 --engine bitboard --device cuda --checkpoint checkpoints/connect4_cnn_seed20261119.pt --log-path results/training_seed20261119.json
```

Checkpoint được lưu tại `checkpoints/connect4_cnn.pt`; mỗi epoch in
training loss, validation loss và policy accuracy. Dữ liệu được tạo với
epsilon exploration có seed, chia theo ván thành train/validation/test,
loại vị trí trùng giữa các tập và chỉ augmentation tập train. Kết quả test
cùng cấu hình được lưu trong checkpoint. Có thể nạp checkpoint bằng
`PolicyAgent.from_checkpoint(...)`.

## Kết quả huấn luyện hiện tại

Checkpoint `checkpoints/connect4_cnn_seed20261119.pt` được huấn luyện từ 4.000
ván Alpha-Beta trên bitboard, dùng seed `20261119`, batch size 256 và 12 epoch.
Mô hình tốt nhất được chọn theo validation loss. Tập test độc lập đạt policy
accuracy khoảng 50,5%; kết quả benchmark đối kháng được lưu dưới `results/`.
Trong 100 ván đối đầu Heuristic, CNN chưa thắng ván nào. CNN hiện là bản thử
nghiệm và cần dữ liệu nhãn mạnh hơn trước khi dùng làm agent cạnh tranh.

## Đánh giá và thống kê

Chạy round-robin 100 ván mỗi cặp:

```powershell
python evaluate.py --games 100 --output results/round_robin.json
```

Chạy cùng giải đấu với engine bitboard:

```powershell
python evaluate.py --games 100 --engine bitboard --output results/round_robin_bitboard.json
```

Đưa policy CNN đã huấn luyện vào round-robin:

```powershell
python evaluate.py --games 100 --engine bitboard --checkpoint checkpoints/connect4_cnn.pt
```

Có thể ghi nhận các nước vượt ngân sách 100 ms:

```powershell
python evaluate.py --games 100 --budget-ms 100
```

Báo cáo gồm tỷ lệ thắng/hòa/thua, score, Wilson 95%, thời gian trung bình,
p95, tổng số node Alpha-Beta, tổng số simulation MCTS và số lần vượt ngân sách.

## Ablation và solver đối chiếu

Chạy ablation:

```powershell
python ablation.py --games 10 --depth 4 --engine bitboard --seed 20261119 --output results/ablation_bitboard_seed20261119.json
```

Các cấu hình gồm Alpha-Beta có/không pruning, có/không move ordering và
MCTS với 50, 100, 500, 1000, 5000 simulation.
Kết quả hiện tại cho thấy pruning giảm số node khoảng 5,5 lần so với tắt
pruning ở depth 4; ordering giảm thêm số node so với cấu hình không ordering.
Các cấu hình đều thắng Random trong 10 ván, nên tỷ lệ thắng đã chạm trần và
không dùng để kết luận sức mạnh tương đối giữa chúng.

`ExternalSolver` nhận executable của solver bên ngoài. Solver phải nhận
board 42 ô và player qua `{board}`, `{player}`, rồi in ra cột 0–6:

```python
from connect4.solver import ExternalSolver
solver = ExternalSolver("path/to/solver.exe", ("{board}", "{player}"))
```

Ví dụ:

```python
from connect4.agents import AlphaBetaAgent, RandomAgent
from connect4.experiments import play_game

metrics = play_game(AlphaBetaAgent(depth=4), RandomAgent(seed=42))
print(metrics)
```
