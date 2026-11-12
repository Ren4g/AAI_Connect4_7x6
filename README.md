# Connect4 7x6

Bộ khung triển khai các mục 1–5 trong `Proposal.docx`:

- `connect4/engine.py`: engine chuẩn 7×6, API `reset`, `legal_actions`, `step`, `is_terminal`, `result`, `render`, `clone`.
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

## Huấn luyện CNN

PyTorch là dependency tùy chọn:

```powershell
python -m pip install torch
python train.py --games 100 --epochs 10
```

Checkpoint được lưu tại `checkpoints/connect4_cnn.pt`; mỗi epoch in
training loss, validation loss và policy accuracy.

## Đánh giá và thống kê

Chạy round-robin 100 ván mỗi cặp:

```powershell
python evaluate.py --games 100 --output results/round_robin.json
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
python ablation.py --games 10 --output results/ablation.json
```

Các cấu hình gồm Alpha-Beta có/không pruning, có/không move ordering và
MCTS với 50, 100, 500, 1000, 5000 simulation.

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
