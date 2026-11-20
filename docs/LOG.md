# Nhật ký dự án

## 2026-11-20 — Pha 0: kiểm kê, tiêu chí và nền tảng

- Đọc `README.md`, các module trong `connect4/`, `evaluate.py`,
  `evaluate_solver.py`, `train.py`, `train_solver.py`, toàn bộ test và danh mục
  `results/`. Tiêu chí được chốt trước khi chạy test/benchmark mới tại
  [`CRITERIA.md`](CRITERIA.md).
- Khác biệt so với bối cảnh ban đầu: không có `pons_solver.py`; adapter đang
  dùng là `connect4/solver.py`. Nguồn solver chuẩn được vendored trong
  `solver/`, với mã upstream tại `solver/connect4/` và binary Windows tại
  `solver/connect4/c4solver.exe`. Không clone hoặc thay solver này.
- Root repository ở commit `b65dc3fefaf742218124e095bd9953a28a0272c4`, nhưng
  working tree có nhiều thay đổi chưa commit từ các lượt triển khai trước; do
  đó commit không đại diện đầy đủ cho source hiện tại. Upstream solver commit
  được đọc là `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`; lệnh đọc status của
  nested checkout kết thúc với `error: daemon terminated`, nên trạng thái sạch
  của solver chưa được xác nhận. SHA256 binary Windows:
  `13903490C56862BE0EB02BAD67087BD77035392786978B04CC5455E4AB5A45FE`.
- Môi trường local: Python 3.13.5; RTX 3050 Laptop GPU; PyTorch `2.9.1+cu128`,
  CUDA 12.8, CUDA khả dụng; 12 logical CPU; RAM vật lý 34.1 GB. `pytest` không
  được cài trong `.venv-gpu`; Python hệ thống chạy được test.
- `python -m pytest -q`: **30 passed, 1 skipped**. Test bị skip là inference
  encoder parity do Python hệ thống không có PyTorch. Chạy lại phép so khớp
  encoder huấn luyện/suy luận riêng trong `.venv-gpu`: thành công, tensor
  `(2, 6, 7)` giống hệt. `git diff --check` không có lỗi whitespace.
- Solver smoke `112233` trả
  `112233 -2 -1 -1 18 -2 -2 -3`, đúng kết quả chuẩn. Solver phát cảnh báo
  `Unable to load opening book: 7x6.book`; không có book ở checkout hiện tại.
- Hiện trạng tính năng: cấp 1–5 có agent; cấp 6, 7, 8 chưa hiện thực. CNN có
  policy/value head nhưng inference chỉ dùng policy. Chưa có PUCT/self-play,
  RL, giao thức checkpoint-resume Colab hay ingest manifest.
- Tập được gọi là test đóng băng nằm ở
  `results/solver_eval_seed20261122.positions.json`: 1.500 dòng (500 mỗi giai
  đoạn), SHA256 `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`,
  nhưng chưa lưu `source_game`; chỉ có 1.498 khóa bàn cờ/gương duy nhất. Kiểm
  toán với dữ liệu solver-label của checkpoint hiện tại tìm thấy 129 khóa giao
  với train và 25 với validation. Vì vậy các CNN đã huấn luyện từ dữ liệu này
  không thể được xem là đánh giá độc lập trên toàn bộ test đóng băng. Giữ file
  test bất biến; checkpoint/kết quả cũ được gắn nhãn legacy, không dùng để qua
  cổng CNN.
- `results/solver_eval_seed20261122.json` hiện chưa có CI bootstrap, không có
  CNN raw riêng, không có depth curve 1–6 trên cùng test và không có bảng theo
  giá trị lý thuyết/vị trí có quyết định. Vòng solver evaluation mới phải bổ
  sung các metric này sau khi hoàn thiện manifest cụm nguồn.
- Chưa chạy benchmark/huấn luyện mới trong lượt kiểm kê này. Mọi số GPU Colab
  trong tương lai sẽ chỉ được báo khi người dùng trả gói artifact và ingest
  local xác nhận hợp lệ.

## 2026-11-20 — Pha 1: đánh giá solver trên test đóng băng

- Tạo `results/solver_eval_seed20261122.manifest.json` cạnh file test bất biến.
  Bộ lấy mẫu được tái tạo đúng thứ tự từ `benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl`; manifest lưu nguồn ván, SHA256 và khóa trạng thái chuẩn hóa gương.
- Đo trước 50 vị trí phân tầng bằng solver: 44,4 s wall tuần tự; khai cuộc 17 vị trí trung bình 2,60 s (p95 8,23 s, max 17,50 s), trung cuộc 0,0055 s/vị trí, tàn cuộc 0,00018 s/vị trí. Cảnh báo thiếu `7x6.book` được giữ lại trong log; không đưa throughput Colab vào so sánh agent.
- Thêm `PonsSolver.scores_batch` và evaluator dùng 8 worker. Lượt cuối có 1.500 vị trí, 8 tiến trình solver, thời gian wall exact-score 139,6 s. Sau khi chấm score, agent chạy tuần tự trên cùng máy local để đo chi phí không bị nhiễu bởi solver song song.
- Kết quả tổng thể exact-optimal [CI 95% cluster]: Random 35,3% [32,9; 37,7], Heuristic 73,5% [71,4; 75,7], AB1 70,3% [68,1; 72,5], AB2 75,6% [73,4; 77,7], AB3 77,2% [75,0; 79,4], AB4 80,5% [78,5; 82,4], AB5 83,1% [81,2; 84,9], AB6 81,5% [79,5; 83,4], MCTS-500 75,3% [73,1; 77,5]. Kết quả AB depth không đơn điệu; AB5 đạt exact cao hơn AB6 trên mẫu này.
- AB3,4,5,6 lần lượt giữ W/D/L 90,5%, 91,7%, 93,3%, 91,5%. Tập có quyết định có 726 vị trí; toàn test có 781 vị trí lý thuyết thắng, 147 hòa, 572 thua cho bên đi. Thời gian local của AB4 trung bình 20,9 ms/nước, AB6 228,6 ms/nước; không áp dụng các số này cho Colab.
- Báo cáo và biểu đồ sinh từ JSON: `docs/solver_eval_frozen_20261120_seed20261125.md`, `results/solver_depth_curve_frozen_20261120_seed20261125.svg`; nguồn đầy đủ `results/solver_eval_final_20261120_seed20261125.json`.
- Chênh lệch với AB depth4 cũ rất nhỏ: exact-optimal mới 80,47% so với con số cũ 80,5%; Heuristic mới 73,53% so với 73,5%; sai khác do làm tròn. MCTS mới 75,27% so với bảng cũ 75,7%; cần giữ cả hai con số và coi bảng mới là lần đo tái lập, chưa quy kết nguyên nhân cho đến khi so sánh build/seed agent cũ.
- CNN không đưa vào lượt này. Checkpoint solver-label seed20261201 có giao với test đóng băng: 129 khóa ở train và 25 ở validation; checkpoint đó bị loại khỏi cổng. Checkpoint gravity-aware được huấn luyện từ 4.000 ván nhưng không lưu lịch sử vị trí: mức giao với test chưa biết, nên cũng chưa đủ bằng chứng để qua cổng. Không gộp hai trường hợp thành cùng một kết luận rò rỉ.
- Test sau thêm manifest và bootstrap: 32 passed, 1 skipped. Windows batch solver test và CLI smoke đa tiến trình đã chạy; smoke 6 vị trí xong trong 8,1 s với 2 worker. Toàn bộ evaluator 1.500 vị trí hoàn tất.

### 2026-11-20 — Colab CNN pipeline và PUCT (bổ sung)

- Dùng đúng solver vendored tại `solver/connect4/c4solver.exe`; xác nhận chế độ `-a` nhận nhiều dòng trong stdin. Tạo gói mới `colab/data/solverlabels_20261120_seed20261120_v2/` bằng 8 tiến trình Pons sống lâu, mỗi tiến trình gửi batch vị trí qua `scores_batch`.
- Gói v2 có 1.500 vị trí solver: train 1.284 trên 1.023 ván nguồn; validation 216 trên 181 ván nguồn; không có trùng canonical mirror với 1.500 vị trí test đóng băng. Tạo lại gói mới, không ghi đè v1.
- Hoàn thiện `python -m connect4.train_cnn`: policy/value loss, lật gương train-time, chọn checkpoint theo validation, checkpoint mới nhất có optimizer/cursor/RNG Python-NumPy-Torch-CUDA; `--resume`, `--max-minutes`, `--smoke`. Smoke CUDA RTX 3050 hoàn thành 2 epoch; thử dừng giữa epoch và resume thành công, tổng 8 bước.
- Thêm notebook `colab/02_train_cnn.ipynb`, hướng dẫn và script ingest. Notebook được xác thực bằng nbformat; ingest kiểm checksum, config/dataset/test hash, tính lại rò rỉ local và chạy inference CPU trước khi chép vào `results/colab/`.
- Thêm `PUCTAgent`: policy prior, value leaf, backprop perspective-aware; `evaluate_solver.py` nhận checkpoint PUCT theo số mô phỏng cố định. Đây là hiện thực suy luận AlphaGo-inspired; chưa huấn luyện policy/value bằng RL và chưa có kết quả hiệu năng từ checkpoint sạch.
- Kiểm chứng hiện tại: `python -m pytest -q -p no:cacheprovider` — 39 passed, 1 skipped; py_compile, nbformat validate và `git diff --check` đều qua. Skip là bài kiểm tra inference phụ thuộc PyTorch ở system Python; smoke GPU đã chạy bằng `.venv-gpu`.
- Còn lại: huấn luyện Colab nhiều seed và ingest artifact; PUCT benchmark trên checkpoint sạch; self-play cấp 8, RL cấp 6, round-robin cuối và báo cáo tổng kết. Chưa có kết quả Colab nào được xem là kết quả thí nghiệm.


### 2026-11-20 ? B? sung c?p 6, self-play c?p 8 v? arena

- Th?m `connect4/selfplay.py`: PUCT t? ch?i nhi?u game ??ng th?i, batch inference leaf theo simulation, Dirichlet ? g?c, nhi?t ?? policy, value outcome, ??i x?ng g??ng, replay buffer v? checkpoint g?m optimizer/RNG/replay/v?n d?/b? ??m. Checkpoint theo kho?ng th?i gian v? t?i ?i?m d?ng.
- Smoke GPU: 2 game/2 simulation/2 update ho?n t?t; ki?m tra pause gi?a game r?i resume; th?m m?t profile nh? 4 game ? 20 simulation ? 10 update ch?y 1,51 gi?y local. ??y ch? l? throughput probe, kh?ng suy di?n ch?t l??ng ho?c th?i gian Colab ch?c ch?n.
- Th?m `connect4/rl_train.py`: REINFORCE c? critic/value v? entropy penalty ??i ??u Random, ??i gh? theo episode, lo?i kh?a test kh?i loss v? l?u hash v? tr? ?? ingest r? l?i. Smoke 2 episode ho?n t?t kho?ng 1,20 gi?y local.
- Th?m notebook `03_selfplay_az.ipynb`, `04_rl_policy_gradient.ipynb`; c?p nh?t `docs/COLAB.md` v?i c?i ??t, resume, artifact v? ??c l??ng c? b?t ??nh.
- M? r?ng ingest cho ba pha CNN/RL/AlphaZero: checksum ??u v?o/??u ra, source-tree hash, checkpoint CPU inference, leakage test c?c b?; ki?m tra replay v? game dang d? c?a AlphaZero, hash v? tr? ?? train c?a RL. ?? th? ingest probe kh?ng-smoke cho c? ba nh?nh v?o th? m?c t?m ngo?i `results/colab/`; kh?ng xem ch?ng l? k?t qu? Colab.
- Th?m `scripts/arena_checkpoints.py` cho 100 opening paired (200 v?n), ?i?u ki?n promotion score >0,5 v? lower bootstrap CI >0,5; smoke 2 opening x?c nh?n th?ng k?/replay, kh?ng promote. `evaluate.py` nh?n checkpoint PUCT cho round-robin.
- Ti?p theo: ch?y to?n b? test/format check sau c?c module m?i; benchmark PUCT/checkpoint ?? ingest; nh?n artifact Colab th?t; ch?y arena ?? c? v? round-robin cu?i. C?p 6 ch? l? baseline RL th?m d?, ch?a c? ???ng cong h?c k?t lu?n ???c.


### 2026-11-20 ? Ngu?n Colab, hi?u ch?nh value v? r? so?t

- Th?m `scripts/package_colab_source.py`; bundle ZIP c? source-tree SHA v? ZIP SHA. Ba notebook hi?n c? ??ng 8 cell theo th? t?, x?c minh bundle hash v? source tree tr??c khi c?i/ch?y CLI.
- Th?m `scripts/evaluate_value_head.py`: d?ng checkpoint ?? ingest, hash solver labels c?ng t?p test ??ng b?ng, b?o value MSE, WDL classification/calibration v? bootstrap 10.000 l?n theo v?n ngu?n; metric/threshold ???c ghi trong `docs/CRITERIA.md` tr??c ph?p ?o.
- Ch?y m?t probe k? thu?t CNN local m?t epoch tr?n data v2, ingest th?nh c?ng, r?i ?o value tr?n frozen test. G?n nh?n **development probe**, kh?ng d?ng ch?n checkpoint v? kh?ng thay cho ba seed Colab: MSE 0,882 [0,844; 0,920], WDL accuracy 44,1% [41,5; 46,6], baseline l?p ph? bi?n nh?t 52,1%; d? ?o?n t?p trung g?n +0,39, c? d?u hi?u l?c quan. Ngu?n ??y ??: `results/value_head_probe_local_cnn_seed71_20261120.json`.
- Tinh ch?nh RL log action ??y ?? m?i episode; self-play log chu?i n??c t?ng game v? ingest ki?m to?n b? hash tr?ng th?i t?ng ???c train, k? c? v? tr? ?? r?i replay buffer.
- B? sung GUI l?a ch?n checkpoint CNN, RL policy ho?c PUCT; b? sung PUCT v?o round-robin cu?i v? script arena paired-opening.
- X?c minh hi?n tr?ng: 41 passed, 3 skipped; py_compile, help CLI, 3 notebook nbformat-valid (8 cell), `git diff --check` qua. Ba skip l? ph?n ph? thu?c PyTorch ? system Python; c?c smoke GPU v? ingest probe ?? ch?y b?ng `.venv-gpu`.
- C?n thi?u k?t qu? th?t t? Colab, benchmark PUCT ?? c?, arena 100 opening tr?n checkpoint th?t, round-robin cu?i v? ph?n b?o c?o/slide c? s? li?u h?c ???c. C?c probe ??a ph??ng kh?ng ???c xem l? k?t lu?n s?c ch?i.
