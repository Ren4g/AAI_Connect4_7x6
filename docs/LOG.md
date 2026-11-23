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

### 2026-11-20 — Colab artifact ingested

- {"date": "2026-11-20", "event": "colab_run_ingested", "run_id": "cnn_seed20261120_colab1", "artifact": "results\\colab\\cnn_seed20261120_colab1", "seed": 20261120, "steps": 609, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-20 — Colab artifact ingested

- {"date": "2026-11-20", "event": "colab_run_ingested", "run_id": "cnn_seed20261121_colab1", "artifact": "results\\colab\\cnn_seed20261121_colab1", "seed": 20261121, "steps": 630, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-20 — Colab artifact ingested

- {"date": "2026-11-20", "event": "colab_run_ingested", "run_id": "cnn_seed20261122_colab1", "artifact": "results\\colab\\cnn_seed20261122_colab1", "seed": 20261122, "steps": 630, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

Nhận định kỹ thuật cho Pha 2
Value Head hiện tại còn yếu và bị overconfident (tự tin thái quá): Ở các bin dự đoán cực trị (~ -0.76 hoặc +0.80), giá trị thực tế trung bình chỉ đạt quanh -0.45 đến +0.71. Độ chính xác phân loại Thắng/Hòa/Thua (WDL accuracy) dao động quanh 50,5% – 53,5% (gần như ngang bằng mức đoán bừa theo lớp đa số 52,1%), và tụt mạnh ở tàn cuộc (42,4% – 45,8%).

Kết luận cho Cấp 7 & Cấp 8: Không thể dùng đơn lẻ Value Head này thay thế hoàn toàn rollout mà thiếu tìm kiếm sâu, nhưng hoàn toàn dùng được làm giá trị khởi tạo (warm-start) cho Cấp 7 (PUCT + rollout ngắn) và Cấp 8 (AlphaZero self-play), nơi vòng lặp tự đấu sẽ tự hiệu chỉnh lại đầu Value qua hàng nghìn ván thực chiến.
### 2026-11-20 — Colab artifact ingested

- {"date": "2026-11-20", "event": "colab_run_ingested", "run_id": "az_seed20261120_run1", "artifact": "results\\colab\\az_seed20261120_run1", "seed": 20261120, "steps": 20000, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-21 — Colab artifact ingested

- {"date": "2026-11-21", "event": "colab_run_ingested", "run_id": "rl_seed20261120_run1", "artifact": "results\\colab\\rl_seed20261120_run1", "seed": 20261120, "steps": 10000, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}


### 2026-11-21 — Đã đánh giá các run Colab và hoàn tất round-robin cuối

- Ingest artifact RL sau khi kiểm checksum/cấu hình, inference CPU và audit leakage với test đóng băng trên local; AlphaZero và ba run CNN đã ingest trước đó. Artifact nằm trong `results/colab/`.
- Đánh giá RL và checkpoint self-play AlphaZero trên test Pons đóng băng 1.500 vị trí cùng Random, Heuristic, AlphaBeta depth 1–6 và MCTS-500. JSON nguồn: `results/solver_eval_az_rl_20261121_seed20261121.json`; test/solver hashes khớp giao thức đóng băng.
- Đánh giá value head của CNN được chọn bằng validation trên cùng test. JSON: `results/value_head/value_cnn_seed20261121_20261121.json`. Bảng và CI sinh từ JSON ở `docs/LEARNING_RESULTS.md`.
- Hoàn tất round-robin local với opening ghép cặp sáu ply, đổi ghế cho Random, Heuristic, AlphaBeta depth 4/6, MCTS-500, CNN chọn bằng validation, RL policy và AlphaZero PUCT. Bộ lọc công bằng loại sáu opening ứng viên có nước thắng ngay trước khi chấp nhận bộ chung 50 opening; JSONL lưu toàn bộ ván.
- Summary, replay, metadata và checksum: `benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.*`; report/heatmap: `docs/ROUND_ROBIN_FINAL.md` và `docs/round_robin_final.svg`.
- Chi phí suy luận của round-robin chỉ đo trên local. Tài nguyên Colab là metadata của run, không so sánh throughput giữa các máy/phiên.

### 2026-11-21 — Arena checkpoint AlphaZero

- So sánh checkpoint self-play cuối với PUCT từ CNN seed khởi tạo bằng 100 opening ghép cặp, cả hai ghế, 100 simulation/nước và fair opening filter.
- Candidate score là 0,535 trên 200 ván; CI bootstrap 95% theo opening [0,475; 0,5925]. Cận dưới không vượt 0,50 nên checkpoint không được promote.
- JSON và replay: `benchmarks/arena/arena_az_vs_init_100openings_seed20261121.json` và `.games.jsonl`. Đây là một run/seed, không đủ để kết luận xu hướng tổng quát.

### 2026-11-21 — Checkpoint arena

- Result: `benchmarks\arena\arena_az_vs_init_100openings_seed20261121.json`; replayable games: `benchmarks\arena\arena_az_vs_init_100openings_seed20261121.games.jsonl`; score=0.5350, games=200, openings=100, promoted=False.

### 2026-11-21 — Đo tài nguyên local và tổng hợp VRAM Colab

- Đo tám agent trong tiến trình/cây tiến trình riêng, mỗi agent chạy một ván bitboard đầy đủ với Heuristic. Working Set được lấy mẫu mỗi 100 ms; riêng PyTorch chạy trong tiến trình con của Python launcher nên phép đo cộng cả descendants. Đây là số mô tả từ một ván, không phải phân bố peak theo nhiều trận; tổng Working Set có thể tính trùng trang chia sẻ.
- Kết quả nguồn: `results/resource_profile_20261121_final.json`; bảng: `docs/RESOURCE_PROFILE.md`; công cụ tái tạo: `scripts/measure_agent_resources.py`.
- CNN/RL/PuCT local có peak process-tree RSS khoảng 483–492 MiB trong phép đo này; các agent cổ điển khoảng 22–23 MiB. Peak Colab CNN manifest là 32.69 MiB allocated trên Tesla T4. Manifest AZ/RL không ghi `peak_vram_bytes`; báo cáo nêu rõ thiếu số liệu, không nội suy.
- RAM khả dụng Colab ghi trong manifest khoảng 12,976 MiB cho cả năm run. Số này là RAM máy tại môi trường Colab, không phải RSS riêng của tiến trình huấn luyện.

### 2026-11-21 — Ghi peak VRAM tích lũy cho các job có resume

- CNN, RL và AlphaZero nay reset peak counter sau bước nạp model/checkpoint, rồi ghi `peak_vram_bytes` vào manifest khi lưu run.
- Khi resume, lấy max giữa peak của manifest cũ và peak ở phiên hiện tại để manifest cuối giữ mức cao nhất qua toàn bộ job.
- Chạy smoke CPU cho cả ba CLI, kiểm tra cả ba manifest có `peak_vram_bytes: 0`; unit test cover trường hợp CUDA tích lũy và CPU. Smoke lưu dưới `artifacts/resource_metrics_smoke/` (được ignore).
- Hai manifest RL/AZ đã ingest là artifact lịch sử, không có peak VRAM nên không thể bổ sung số thật sau khi job kết thúc. Không thay thế bằng ước lượng.
- Tái chạy ba smoke checkpoint CPU với peak cũ giả lập 123,456 bytes; mỗi manifest sau resume giữ nguyên 123,456 và tăng `resume_count` lên 1. Đây chỉ là kiểm tra aggregation của metadata, không phải phép đo VRAM.

### 2026-11-21 — Báo cáo tổng kết học thuật

- Thêm `docs/report.md`, báo cáo tổng kết các phương pháp tám cấp, giao thức solver/round-robin, kết quả có CI, cổng CNN, value head, phân tích nước sai đầu tiên, giới hạn và khả năng tái lập.
- Các bảng solver, stage, hiệu ghép cặp, score đối đầu, arena và counts lỗi đầu tiên được kết xuất trực tiếp từ JSON. Script tái tạo: `scripts/render_final_report.py`.
- Xác nhận bằng test tạo report trong thư mục tạm và so khớp các mốc 80.5% (AB4) và 181/194 ván CNN có sai điểm exact.

### 2026-11-21 — Cập nhật Colab source bundle và kiểm tra cell notebook

- Cập nhật ba notebook huấn luyện để dùng source bundle `20261121_v6`, chứa manifest peak VRAM; cập nhật lệnh đóng gói mặc định và hướng dẫn tải bundle lên Drive.
- Kiểm tra AST từng cell Python và sửa các dòng bị nối liền trong notebook CNN (SOURCE_ZIP/SOURCE_MANIFEST và `print`/`def tree_hash`) cùng khai báo manifest trong AZ/RL.
- `nbformat` validate cả ba notebook và `tests/test_colab_pipeline.py` chạy qua. Gói v6 cần được tạo sau commit cuối; run lịch sử đã complete không thể dùng resume để khôi phục VRAM training cũ.


### 2026-11-21 ? ??ng k? tr??c ti?u ch? V2

- ?? kh?a `docs/CRITERIA_V2.md` tr??c c?c ph?p ?o/th? nghi?m V2; h?n m?t ng?y ??n 2026-11-22 10:37 Asia/Ho_Chi_Minh, tr?n Colab 3 gi? GPU.
- M?y local ghi nh?n `os.cpu_count() = 12`; c?c benchmark th?i gian s? ch? ch?y tr?n m?y n?y.
- C?c k?t qu? l?ch s? ti?p t?c ???c gi? nguy?n; vi?c ??c manifest v? log l?ch s? ???c xem l? ki?m to?n provenance.


### 2026-11-21 documentation encoding correction

- The preceding appended V2 log block was written with damaged Vietnamese characters by a shell encoding mismatch. The UTF-8 files `docs/CRITERIA_V2.md` and `docs/CRITERIA_V2_ADDENDUM_20261121.md` are the authoritative preregistration; the damaged block must not be used as a source of criteria.
- New Markdown additions are written explicitly as UTF-8. `docs/COLAB.md` V2 instructions were rewritten and verified as UTF-8.
- No Colab training results have been received in this phase; no CNN v3 or AZ v2 quality claim is made.


### 2026-11-21 V2 local diagnostics and Colab preparation

- Pha A audited the existing CNN, AZ, and RL manifests and logs. Existing CNN v2 used 1,284 training and 216 validation positions; batch size 64 explains 21 steps per epoch. Existing AZ artifacts have no intermediate checkpoints, so the historic 0/25/50/100 learning curve cannot be reconstructed.
- Clean, source-game-grouped validation diagnostics exclude states found in the relevant replay buffers: CNN raw policy 55.3% exact-optimal versus 73.5% with the tactical filter (paired gain +18.1 percentage points, 95% CI +13.0 to +23.3); CNN PUCT-100 76.7%; historic AZ PUCT-100 66.5% (paired AZ minus initial CNN: -10.2 points, CI -16.5 to -4.0). These are validation diagnostics, not frozen-test results.
- The RL filter improved exact-optimal selection, but trained-filter versus reconstructed episode-0-filter was +2.8 points (95% CI -3.3 to +8.9); the episode-0 weights were reconstructed under a different local Torch version. This is exploratory and insufficient to justify another RL GPU run.
- Generated 30,000 Pons-exact training labels (10,000 per stage), with 25,470 training and 4,530 validation positions. Frozen-test and mirror overlap is zero. The 12-process labeling run took 2,172.45 seconds. Generated a matched 30,000-position AlphaBeta-3 policy-target set in 54.89 seconds; Pons exact values are retained.
- The 30,000-position CNN smoke completed locally on RTX 3050 in 0.508 seconds for two epochs and eight updates; it validates the pipeline only and is not a quality result. Added nested source-game data fractions, hard-policy target, grouped dataset verification, PUCT safety-filter variants, and AZ iteration snapshots.
- Full local test suite: 67 passed. Notebook format validation: five notebooks valid. `git diff --check` passed. The source bundle v9 ZIP SHA256 is recorded in its sidecar; no Colab result has been received or claimed.
- Next action for the user: run `colab/05_train_cnn_v3_30k.ipynb` within the preregistered 60-minute Phase B GPU cap. Return each run ZIP and manifest. Run Phase C only if time remains within the 180-minute total budget, using `colab/06_selfplay_az_v2.ipynb`.


### 2026-11-21 Phase C protocol amendment, prospective

- Revised the new Phase C protocol after the Phase A diagnosis showed historical AZ PUCT-100 at 66.5% exact-optimal on clean validation versus 76.7% for initial CNN PUCT-100; paired difference -10.2 percentage points, 95% CI [-16.5, -4.0]. This does not alter old outcomes.
- Registered two 400-simulation runs: warm start from validation-selected CNN v3 (`az_v2_warm_seed20261121`) and random initialization (`az_v2_scratch_seed20261122`). Each has 40 iterations, 32 games/iteration, 16 parallel games, 200 updates/iteration, batch 128, and snapshots every 5 iterations including 0 and 40.
- Added hard budget accounting to the Colab notebook: maximum 30 minutes per resumable segment, 60 per run, 120 total across the two sequential runs. The Drive ledger reserves before launch; an unclean disconnect leaves the reservation conservatively charged. Artifact zips include snapshot files recursively and the ledger.
- This is a prospective protocol amendment for the new Phase C, not a reinterpretation of the earlier locked V2 criteria or V1 results. One seed per initialization supports only a case comparison, not a method-level superiority claim under the three-seed rule.
- Phase B notebooks and datasets were not modified. Notebook changes are restricted to `colab/03_selfplay_az.ipynb`; package the updated source as bundle v12.

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261121", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261121", "seed": 20261121, "steps": 11940, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261121_ab3", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261121_ab3", "seed": 20261121, "steps": 10348, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261121_frac10", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261121_frac10", "seed": 20261121, "steps": 1160, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261121_frac30", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261121_frac30", "seed": 20261121, "steps": 3630, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261121_hard", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261121_hard", "seed": 20261121, "steps": 9950, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261122", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261122", "seed": 20261122, "steps": 11940, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261122_ab3", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261122_ab3", "seed": 20261122, "steps": 9552, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261122_frac10", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261122_frac10", "seed": 20261122, "steps": 1040, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261122_frac30", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261122_frac30", "seed": 20261122, "steps": 3630, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261122_hard", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261122_hard", "seed": 20261122, "steps": 11144, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261123", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261123", "seed": 20261123, "steps": 11940, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261123_ab3", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261123_ab3", "seed": 20261123, "steps": 11144, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261123_frac10", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261123_frac10", "seed": 20261123, "steps": 1120, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261123_frac30", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261123_frac30", "seed": 20261123, "steps": 3630, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Colab artifact ingested

- {"date": "2026-11-22", "event": "colab_run_ingested", "run_id": "cnn_v3_30k_seed20261123_hard", "artifact": "D:\\Workspace\\A_MLE\\labFinal - Connect4_7x6\\results\\colab\\cnn_v3_30k_seed20261123_hard", "seed": 20261123, "steps": 11144, "frozen_test_sha256": "3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01", "local_leakage_check": "passed", "inference_check": "passed"}

### 2026-11-22 — Ingest CNN v3 và khóa checkpoint Phase C theo validation

- 15 archive CNN v3 được kiểm tra và ingest bằng source bundle v11 đúng source-tree SHA; artifact checksum, dataset checksum, inference CPU và leakage test local đều đạt. Hash của Colab run không khớp working tree mới vì mã nguồn đã thay đổi sau bundle; không bỏ qua kiểm tra, thay vào đó chạy ingester từ snapshot v11.
- So sánh cấu hình bằng validation exact-optimal top-1, ba seed/cấu hình: chọn Pons soft · 100% (80.09% ± 0.29 điểm phần trăm).
- Checkpoint warm-start chọn theo validation loss trong cấu hình đã khóa: cnn_v3_30k_seed20261122, epoch 30, loss 1.4617, SHA256 `3d86468ab7586ef502e8b1a894909b722506192b88fcc8a9937b814fb96ff35b`. Frozen test chưa được dùng để chọn.

### 2026-11-22 — Đánh giá CNN v3 và cấu hình Phase C được chốt

- Đánh giá một lần ba checkpoint Pons-soft 100% trên 1.500 vị trí test đóng băng; Pons commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41`, SHA test `3d8f9c747889d44821d8fd2ebe338570f59fef67f7e8961dd05153ef82d43b01`, 10.000 bootstrap theo source game.
- CNN thuần đạt exact 65,5–68,6%; có safety filter đạt 75,7–76,7%. Hiệu exact ghép cặp CNN-filter trừ AlphaBeta depth 3 có CI lần lượt [-3,4; 1,6], [-3,0; 2,1], [-4,0; 1,1] điểm phần trăm; cả ba chưa phân biệt được với mốc và chưa đạt cổng tối thiểu đã đăng ký.
- Lựa chọn khởi tạo vẫn là Pons-soft 100% theo validation (80,09% ± 0,29 pp); frozen test không được dùng để chọn. Checkpoint seed 20261122, hash `3d86468ab7586ef502e8b1a894909b722506192b88fcc8a9937b814fb96ff35b`.
- Phase C tiếp tục là thí nghiệm tự chơi có kiểm soát warm-start so với scratch; kết quả này không hàm ý CNN đã vượt giáo viên và không dự báo trước kết quả AlphaZero. Dùng `colab/03_selfplay_az.ipynb`, bundle v12, 400 mô phỏng/nước, 40 iteration, 32 ván/iteration, 16 song song, 200 cập nhật/iteration, batch 128; hai run tối đa 60 phút, toàn pha 120 phút.
- Nguồn kết quả: `results/solver_eval_cnn_v3_soft_3seeds_20261122.json`, `results/cnn_v3_validation_selection_20261122.json`; báo cáo tự sinh: `docs/CNN_V3_RESULTS.md`.

### 2026-11-23 — Hoàn tất Phase C v2 local và đánh giá cuối

- Hai run Phase C hoàn tất 40 vòng, mỗi run 1.280 ván/8.000 cập nhật; warm-start resume 8 lần, scratch 7 lần; cấu hình 400 simulations/nước, 32 ván/vòng, 16 song song, 200 cập nhật/vòng, batch 128. RTX 3050 Laptop, 12 CPU threads. Manifest và checksum ở `results/phase_c_local/az_v2_warm_seed20261121/` và `az_v2_scratch_seed20261122/`.
- Đường cong validation có 18 snapshot (0,5,...,40), 4.490 vị trí warm và 4.497 scratch sau khi loại replay overlap; bootstrap 10.000 lần theo cụm ván nguồn. Warm PUCT-filter giảm từ 89,53% xuống 85,90% exact; scratch tăng từ 81,48% lên 82,48%, hiệu ghép cặp +1,00 pp, CI [−0,16; +2,14] pp nên chưa phân biệt được.
- Chọn warm iteration 0 và scratch iteration 40 theo validation, không dùng test để chọn. Test đóng băng 1.500 vị trí: PUCT-400 warm có filter đạt 82,13% exact [80,14; 84,03] và 94,00% WDL [92,70; 95,20]; scratch có filter đạt 76,80% exact [74,66; 78,97] và 91,87% WDL [90,46; 93,23]. Hiệu warm so với AB3: exact +4,93 pp, CI paired [+2,66; +7,21]; WDL +3,47 pp, CI [+1,99; +4,95]. Nguồn: `results/solver_eval_phase_c_local_20261123.json`.
- Round-robin mới: 8 agent, 28 cặp, 100 ván/cặp, 50 khai cuộc 6 ply dùng chung đổi ghế, bitboard, fair filter; 61 lần thử, giữ 50, loại 10 khai cuộc có immediate win và 1 trùng trạng thái. Seed 20261123; PUCT checkpoint warm iteration 0, 400 simulations. JSON, metadata và JSONL replay: `results/round_robin_phase_c_20261123.{json,meta.json,games.jsonl}`.
- Chẩn đoán train: replay capacity 100.000, đạt 45.385 warm/37.303 scratch; optimizer Adam LR 1e-3 cố định, value loss weight 0,5, không scheduler/weight decay tường minh. Train log không có policy/value loss riêng hay target entropy; chưa chạy PUCT 100/1600, MCTS-rollout đối chứng hay LR/update ablation. Không quy nguyên nhân suy giảm cho một giả thuyết đơn lẻ.
- Tạo tài liệu kết quả từ JSON bằng `python scripts/build_phase_c_docs.py`; báo cáo tổng hợp và file xuất được tái tạo bằng `python scripts/build_report_assets.py`. Các file mới: `docs/PHASE_C_V2_RESULTS.md`, `docs/ROUND_ROBIN_PHASE_C_20261123.md`; log được ghi UTF-8.
