# Connect4 7x6

Báo cáo tiến độ và kết quả thí nghiệm mới nhất: [docs/PROGRESS_REPORT.md](docs/PROGRESS_REPORT.md).

Kết quả được phân thư mục theo mục đích: `benchmarks/` chứa dữ liệu trận đấu,
round-robin và ablation; `results/` chứa đánh giá solver, dữ liệu/nhật ký huấn
luyện và phân tích CNN. Các benchmark chính đã chạy lại được giữ cùng JSONL
ván gốc; các bản pilot và kết quả cũ bị thay thế đã được dọn.

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
python -m connect4.gui --engine bitboard --checkpoint checkpoints/connect4_cnn_gravityaware_seed20261119.pt --device cuda
```

## Huấn luyện CNN

PyTorch là dependency tùy chọn:

```powershell
python -m pip install torch
python train.py --games 100 --epochs 10 --engine bitboard
```

Hoặc cài các dependency của dự án bằng `python -m pip install -r requirements.txt`.
Để tái lập lần huấn luyện đã lưu trong repository, dùng seed `20261119` và
cấu hình ở `results/training_gravityaware_seed20261119.json`:

```powershell
python train.py --games 4000 --epochs 12 --batch-size 256 --learning-rate 0.001 --epsilon 0.05 --teacher-depth 3 --seed 20261119 --engine bitboard --device cuda --checkpoint checkpoints/connect4_cnn_gravityaware_seed20261119.pt --log-path results/training_gravityaware_seed20261119.json
```

Checkpoint mặc định được lưu tại `checkpoints/connect4_cnn.pt`; mỗi epoch in
training loss, validation loss và policy accuracy. Dữ liệu được tạo với
epsilon exploration có seed, chia theo ván thành train/validation/test,
loại vị trí trùng giữa các tập và chỉ augmentation tập train. Kết quả test
cùng cấu hình được lưu trong checkpoint. Có thể nạp checkpoint bằng
`PolicyAgent.from_checkpoint(...)`.

## Kết quả huấn luyện hiện tại

Checkpoint `checkpoints/connect4_cnn_gravityaware_seed20261119.pt` được huấn luyện từ 4.000
ván Alpha-Beta trên bitboard, dùng seed `20261119`, batch size 256 và 12 epoch.
Nhãn được tạo bởi Alpha-Beta depth 3 với evaluator xét ô đe dọa có chơi được
ngay theo trọng lực. Mô hình tốt nhất được chọn theo validation loss ở epoch
11; tập test độc lập đạt policy accuracy 51,2%. Lúc chơi hiện chỉ dùng policy
head; value head đã được huấn luyện nhưng chưa tham gia chọn nước đi. Lượt
huấn luyện mất 280 giây trên RTX 3050, đỉnh VRAM 0,076 GB.

## Đánh giá và thống kê

Chạy round-robin 100 ván mỗi cặp, với 6 nước khai cuộc ngẫu nhiên có seed.
Loại khai cuộc có thắng tức thì sẵn hoặc không có nước an toàn; cùng một bộ
100 vị trí được dùng cho mọi cặp:

```powershell
python evaluate.py --games 100 --seed 20261119 --opening-plies 6 --engine bitboard --checkpoint checkpoints/connect4_cnn_gravityaware_seed20261119.pt --device cuda --output benchmarks/round_robin/round_robin_full_seed20261119.json
```

Chạy cùng giải đấu với engine bitboard:

```powershell
python evaluate.py --games 100 --engine bitboard --output benchmarks/round_robin/round_robin_bitboard.json
```

Để tắt khai cuộc ngẫu nhiên, dùng `--opening-plies 0`. Các ván deterministic
khi đó có thể trùng nhau. Số ván phải là số chẵn để cân bằng ghế.

Đưa policy CNN đã huấn luyện vào round-robin:

```powershell
python evaluate.py --games 100 --engine bitboard --checkpoint checkpoints/connect4_cnn_gravityaware_seed20261119.pt
```

Có thể giới hạn thời gian suy nghĩ của AlphaBeta và MCTS:

```powershell
# AlphaBeta depth 4 / 100 ms va MCTS 100 ms
python evaluate.py --games 100 --preset fast --paired-openings --include AlphaBeta MCTS

# AlphaBeta depth 8 / 500 ms va MCTS 500 ms
python evaluate.py --games 100 --preset standard --paired-openings --include AlphaBeta MCTS
```

`fast` la AlphaBeta depth 4 so voi MCTS 100 ms; AlphaBeta da dung khoang 7,37 ms/nuoc trong benchmark truoc. `paired-openings` chay moi opening hai lan, doi ghe, va CI bootstrap gom theo opening. `standard` la depth toi da 8, 500 ms; day la ngan sach toi da, khong dam bao hai agent tieu thu cung compute.

Báo cáo tách thời gian, node/lượt, simulation/lượt và vượt ngân sách theo từng
agent. `win_rate_wilson_95` dùng tổng số ván; hòa được tính là kết quả không
thắng. `decisive_win_rate_wilson_95` loại hòa khỏi mẫu số. `score_bootstrap_95`
là bootstrap phân tầng theo ghế, với hòa nhận 0,5 điểm. `--budget-ms` là giới
hạn tối đa; chỉ AlphaBeta và MCTS có tìm kiếm dừng theo thời gian, còn các agent
nhanh khác chạy bình thường và vẫn được đo thời gian.

Ở benchmark 25 ms với 100 khai cuộc, AlphaBeta đạt 67 thắng, 3 hòa, 30 thua
(score 0,685; bootstrap 95% `[0.595, 0.77]`); AlphaBeta trung bình 21,7 ms và
1.645 node/nước. MCTS trung bình 23,8 ms, khoảng 875 simulation/nước, với 13
lượt vượt ngân sách; AlphaBeta vượt một lượt. Bộ tích lũy MCTS dùng reward trên `[0, 1]` (thắng 1, hòa
0,5, thua 0), phù hợp với exploration constant `sqrt(2)`.

Khi dùng giới hạn thời gian, số simulation MCTS và độ sâu hoàn tất của
AlphaBeta phụ thuộc tốc độ máy; seed không đảm bảo tái tạo chính xác cùng ván
trên phần cứng khác. JSONL lưu chuỗi nước đi để phát lại.

Kết quả round-robin hiện tại: Random thắng 2/100 ván trước AlphaBeta, 2/100
trước CNN và không thắng ván nào trước Heuristic/MCTS. AlphaBeta depth 4 đạt
score 0,73 trước Heuristic và 0,58 trước MCTS-500; bootstrap 95% của cặp
AlphaBeta–MCTS là `[0.485, 0.675]`, còn chứa 0,5. Trong ma trận chính, CNN đạt
score 0,435 trước Heuristic, 0,21 trước AlphaBeta và 0,35 trước MCTS. Hai seed
CNN–Heuristic cho 200 ván tổng hợp: CNN thắng 85, hòa 10, thua 105, score 0,45;
bootstrap phân tầng 95% là `[0.3825, 0.5175]`, nên chưa kết luận được khác biệt
thống kê. Mỗi cặp dùng 100 bàn khai cuộc khác nhau; cùng một bộ khai cuộc được
dùng chung giữa các cặp và luân phiên ghế.
Ma trận đầy đủ nằm trong
`benchmarks/round_robin/round_robin_full_seed20261119.json`.

## Chẩn đoán CNN và tiêu chí chuyển giao

Phân tích 194 ván CNN thua bằng Pons solver ghi riêng nước kém exact score và nước làm xấu WDL. Có 181 ván có nước kém exact, 145 ván có nước WDL-degrading; 14 trong số này tạo fork theo kiểm tra tác chiến hai ply. 97 opening đã là thế thua cưỡng bức cho CNN trước nước đi đầu tiên. Cả hai ván CNN thua Random đều thuộc nhóm này; một ván không có nước kém tối ưu, ván còn lại CNN chọn đường thua nhanh hơn nhưng không thể cứu WDL. Chi tiết trong `docs/CNN_LOSS_ANALYSIS.md` và `results/cnn_first_errors_seed20261122.json`.

Cổng tối thiểu trước RL được đề xuất là CNN không kém AlphaBeta depth 3 quá 5 điểm phần trăm theo CI bootstrap cụm ván, và WDL-preserving không kém quá 3 điểm. 85% exact-optimal là mục tiêu, không phải cổng chặn. Xem `docs/CNN_PROMOTION_CRITERIA.md`.

## AlphaBeta depth curve theo solver

Mỗi depth được chấm trên cùng 300 vị trí, chia 100 theo mỗi giai đoạn. Exact-optimal rate depth 1–6 lần lượt là 74,7%, 77,3%, 79,0%, 79,7%, 81,3%, 82,0%; CI 95% bootstrap gom theo ván nguồn được lưu trong `results/solver_depth_curve_seed20261125.json`. Mẫu 300 vị trí là benchmark thăm dò, không phải test held-out của CNN.

## Solver absolute evaluation

The original solver table below is a historical, no-CI result. It remains for
comparison and is superseded for non-CNN agents by the new frozen-set evaluation.
The CNN values below are not certified held-out results: the prior training
position history was not stored.

| Agent | Exact-optimal | WDL-preserving |
|---|---:|---:|
| Random | 35.0% | 69.6% |
| Heuristic | 73.5% | 88.3% |
| AlphaBeta depth 4 | 80.5% | 91.7% |
| MCTS-500 | 75.7% | 91.3% |
| CNN-Policy | 70.4% | 87.8% |

The frozen test is `results/solver_eval_seed20261122.positions.json`, with
source-game and mirror hashes in
`results/solver_eval_seed20261122.manifest.json`. Its new non-CNN evaluation is
`results/solver_eval_final_20261120_seed20261125.json`, with 10,000 clustered
bootstrap replicates. Generate the report and depth plot from the JSON:

```powershell
python -m scripts.render_solver_report --input results/solver_eval_final_20261120_seed20261125.json --markdown docs/solver_eval_frozen_20261120_seed20261125.md --svg results/solver_depth_curve_frozen_20261120_seed20261125.svg
```

The solver-label CNN checkpoint is excluded from this frozen-set comparison:
audit found 129 train and 25 validation state hashes overlapping the locked test.
The earlier 898-position comparison remains legacy and cannot pass the CNN gate.

## Ma trận ghép cặp và benchmark mới

Các bộ kết quả paired và fair-filter đã dùng 50 khai cuộc, mỗi khai cuộc chơi cả hai ghế. Seed `20261124` không lọc loại 5/50 opening có thắng tức thì; fair filter loại 5 và sinh 5 opening thay thế. Hai ma trận trước/sau lọc nằm trong `benchmarks/round_robin/round_robin_unfiltered_paired_seed20261124.json` và `benchmarks/round_robin/round_robin_filtered_paired_seed20261124.json`; metadata và JSONL có cùng tên.

Sau fair filter, Heuristic–AlphaBeta depth 4 có score Heuristic 0,225 (21–3–76); AlphaBeta depth4–MCTS500 score AlphaBeta 0,635 (61–5–34); Heuristic–CNN score Heuristic 0,570 (53–8–39); AlphaBeta depth4–CNN score AlphaBeta 0,840 (83–2–15). Khoảng tin cậy bootstrap lấy opening làm cụm.

Benchmark standard 100 ván/50 opening paired: AlphaBeta depth trần 8 / 500 ms đạt 39–9–52 trước MCTS 500 ms (score 0,435; CI 95% theo opening `[0,345; 0,525]`). AlphaBeta vượt 2/1.346 lượt, MCTS vượt 13/1.353 lượt. Xem `benchmarks/head_to_head/benchmark_ab_mcts_standard_100_paired_seed20261126.json`.

AlphaBeta depth 6 đạt score 0,845 trước Heuristic trên 100 ván paired (CI 95% `[0,78; 0,91]`); thời gian trung bình 62 ms/nước. Ablation ba seed, mỗi seed 10 ván ghép opening trước Heuristic, cho score trung bình của AlphaBeta có ordering 0,767 và MCTS 1000/5000 lần lượt 0,833/0,833. Mẫu ablation còn nhỏ, chủ yếu dùng để so chi phí và tìm xu hướng; file `benchmarks/ablation/ablation_vs_heuristic_3seeds_paired.json`.

Chạy ma trận ghép opening bằng `--paired-openings`; mỗi cặp số ván phải chẵn, vì mỗi opening xuất hiện hai lần, đổi ghế. Score bootstrap khi đó lấy opening làm đơn vị lấy mẫu.

## Ablation và solver đối chiếu

Chạy ablation:

```powershell
python ablation.py --games 10 --depth 4 --engine bitboard --seed 20261119 --output benchmarks/ablation/ablation_bitboard_seed20261119.json
```

Các cấu hình gồm Alpha-Beta có/không pruning, có/không move ordering và
MCTS với 50, 100, 500, 1000, 5000 simulation.
Metric ablation chỉ thống kê agent đang thử, không gộp thời gian của Random.
Kết quả hiện tại cho thấy pruning giảm số node khoảng 6,4 lần so với tắt
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
# Huấn luyện và agent PUCT (tiến độ hiện tại)

Gói nhãn solver sạch đang dùng: `colab/data/solverlabels_20261120_seed20261120_v2/`.
Nhãn được sinh từ solver vendored `solver/connect4/c4solver.exe`; tám worker gửi nhiều
position qua stdin của một tiến trình solver sống lâu cho mỗi worker.

Smoke local (CUDA nếu có):

```powershell
python -m connect4.train_cnn --dataset colab/data/solverlabels_20261120_seed20261120_v2 --output-root artifacts/smoke --run-id cnn-smoke --device cuda --smoke --max-minutes 2
```

Mở `colab/02_train_cnn.ipynb` để huấn luyện CNN trên Colab. Notebook tiếp tục run
đang có trên Drive; chỉ mang checkpoint về sau khi trạng thái `complete`, rồi xác
minh bằng hướng dẫn trong `docs/COLAB.md`. PUCT có thể đánh giá với checkpoint đã
ingest bằng `evaluate_solver.py --puct-checkpoints LABEL=PATH --puct-simulations 200`.
Đây hiện là PUCT suy luận với policy/value đã huấn luyện; self-play AlphaZero cấp 8
và RL cấp 6 chưa được bàn giao.
