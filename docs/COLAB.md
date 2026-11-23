# Chạy các job GPU trên Google Colab

Các notebook chỉ điều phối CLI trong package `connect4`; dữ liệu chuẩn được lưu trên Drive để tiếp tục sau khi runtime bị ngắt. Không đưa token hay thông tin đăng nhập vào notebook.

## Chuẩn bị

1. Tạo thư mục `MyDrive/connect4/artifacts` và `MyDrive/connect4/data`.
2. Tải lên gói dữ liệu solver sạch tại `colab/data/solverlabels_20261120_seed20261120_v2/` sau khi được tạo: toàn bộ `train_validation.npz`, `hashes.json`, `frozen_test_hashes.json`, `manifest.json`.
3. Sau khi checkout commit mới nhất, chạy `python scripts/package_colab_source.py` để tạo `colab/source_bundle_20261121_v6.zip` cùng manifest. Tải cả hai file lên `MyDrive/connect4/`, giữ nguyên tên. Script loại môi trường ảo/cache, ghi commit, SHA256 ZIP và source tree hash; notebook kiểm tra hash trước khi chạy.
4. Mở `colab/02_train_cnn.ipynb` bằng Colab, chọn Runtime → Change runtime type → T4 GPU, sửa khối tham số đầu notebook rồi chạy toàn bộ. Chạy ba lượt tuần tự với seed `20261120`, `20261121`, `20261122` và `RUN_ID` riêng; mỗi lượt là một run độc lập.

Notebook in `nvidia-smi`, `nproc`, `free -h`, phiên bản Python/PyTorch/CUDA trước khi chạy. Tham số mặc định chạy tối đa 60 phút mỗi phiên; checkpoint mới nhất được ghi trên Drive sau mỗi epoch và có thể resume bằng cách chạy lại notebook. `--max-minutes` dừng CLI gọn; nếu runtime bị ngắt đột ngột, lần sau dùng `--resume`.

Ước lượng ban đầu cho một seed, 30 epoch: khoảng 1–5 phút trên runtime GPU Colab tương đương T4; độ bất định cao do chưa đo Colab và còn phụ thuộc thời gian mount/đọc Drive. Ước lượng suy ra từ RTX 3050 local: 1 epoch đủ dữ liệu (1.284 train, batch 64) hoàn tất trong 0,65 s, smoke 2 epoch tí hon dưới 1 s tính sau khi khởi động Python. Không dùng các tốc độ này so sánh agent. Nếu run chưa hoàn tất, tiếp tục các lát 60 phút; mỗi lát có thể dừng sớm khi hoàn tất.

## Nhận kết quả

Sau khi trạng thái là `complete`, tải nguyên thư mục `artifacts/<run_id>/` về `results/colab_inbox/<run_id>/`. Không dùng checkpoint trực tiếp cho bảng kết quả. Chạy local:

```powershell
python scripts/ingest_colab_run.py results/colab_inbox/<run_id> --dataset colab/data/solverlabels_20261120_seed20261120_v2
```

Chỉ thư mục đã được ingest vào `results/colab/<run_id>/` mới đủ điều kiện đánh giá. Hiện có job CNN có giám sát và job self-play cấp 8. Tốc độ Colab thay đổi theo GPU được cấp phát; thời lượng 60 phút là giới hạn mỗi phiên, không phải dự báo hoàn thành.

Với CNN đã ingest, đo riêng value head trên test đóng băng (không dùng kết quả để chọn checkpoint):

```powershell
python scripts/evaluate_value_head.py --run results/colab/<cnn_run> --checkpoint checkpoint_best.pt --output results/value_head/value_<run_id>.json
```

## Cấp 8: AlphaZero-inspired self-play

Cấu hình mặc định lịch sử bên dưới không phải cấu hình Phase C hiện hành. Phase C dùng `colab/03_selfplay_az.ipynb` với source bundle v12; xem mục “Phase C V2” ở cuối tài liệu để biết checkpoint, run ID, ngân sách và cách resume. Không dùng notebook `colab/06_selfplay_az_v2.(outdated).ipynb`.

Notebook chạy PUCT với prior policy, value của mạng, nhiễu Dirichlet ở gốc, nhiệt độ theo ply, replay buffer và cập nhật policy/value. Việc đánh giá checkpoint thực hiện local sau khi ingest; runtime Colab chỉ dùng cho huấn luyện và số bước/mô phỏng đã định trước.

## Cấp 6: REINFORCE/actor-critic

Notebook `colab/04_rl_policy_gradient.ipynb` huấn luyện policy/value CNN qua ván với Random, luân phiên quân người học, entropy regularization và critic baseline. Job kiểm tra trạng thái trước khi thêm mỗi mẫu vào loss, loại vị trí thuộc test, lưu checksum tập vị trí đã dùng, optimizer và RNG sau mỗi game/định kỳ. Mặc định giới hạn mỗi lượt Colab là 60 phút và tiếp tục cùng `RUN_ID`.

Ước lượng ban đầu: khoảng 1–4 giờ cho 10.000 episode, độ bất định cao. Ước lượng ngoại suy từ smoke RTX 3050 (2 episode hoàn tất trong 1,20 giây tính thuần); mỗi bước cần suy luận theo lượt nên throughput phụ thuộc overhead Python/GPU và Colab được cấp. Dùng lát 60 phút đầu để đo số episode thực tế; đây là RL thăm dò đối đầu Random, không phải phép so sánh trực tiếp với self-play AlphaZero.

Sau job CNN có giám sát, có thể chạy self-play cấp 8 (`03`) và RL cấp 6 (`04`) độc lập, mỗi job dùng `RUN_ID` và checkpoint riêng. Mỗi artifact phải được ingest trước khi dùng trong đánh giá.

## Chọn checkpoint self-play

Sau khi ingest checkpoint cấp 8, so nó với incumbent PUCT bằng arena local ghép cặp:

```powershell
python scripts/arena_checkpoints.py --candidate results/colab/<candidate_run>/checkpoint_policy.pt --incumbent results/colab/<incumbent_run>/checkpoint_policy.pt --simulations 100 --games 200 --opening-plies 6 --seed 20261120 --device cuda --output benchmarks/arena/arena_<run_id>.json --games-log benchmarks/arena/arena_<run_id>.games.jsonl --promote-to checkpoints/puct_<run_id>.pt
```

Promotion yêu cầu score > 0,50 và cận dưới bootstrap 95% theo opening > 0,50 trên ít nhất 100 opening hợp lệ, mỗi opening đánh hai ghế. Kết quả và replay được lưu riêng trong `benchmarks/arena/`; smoke không được promote. Với người đi từ scratch/agent khác, hãy dùng một checkpoint làm incumbent cố định cho cả lần so sánh.

Notebook theo mốc thí nghiệm dùng source snapshot khác nhau và không thay thế chéo: baseline CNN/RL dùng v6; Phase B CNN v3 dùng v11; Phase C AlphaZero V2 dùng v12. Mỗi notebook kiểm tra hash source bundle đã được ghim trong chính notebook. Các manifest RL/AZ lịch sử chưa có trường peak VRAM đầy đủ; muốn đo peak training VRAM cho các run đó phải huấn luyện lại từ đầu, vì resume checkpoint hoàn tất không khôi phục mức peak phiên cũ.

## Phase B V2: solver-supervised CNN

Upload `colab/data/solverlabels_20261121_seed20261121_v3_30k/` to `MyDrive/connect4/data/`. This package contains 30,000 exact-Pons positions, balanced across three stages, split by source game, and checked against the frozen test set and its mirrors. Upload the separate AlphaBeta-3 teacher package only when running that ablation.

Build the current source bundle and upload the ZIP and its JSON sidecar to Drive:

```powershell
python -m scripts.package_colab_source --output colab/source_bundle_20261121_v11.zip
```

Open `colab/05_train_cnn_v3_30k.ipynb`, set `SMOKE=False`, and run one configuration per artifact. The preregistered maximum is 15 runs: 0.1, 0.3, and 1.0 nested source-game fractions with three seeds each; three full-data hard-policy runs; and three full-data AlphaBeta-3 teacher runs. Seeds are 20261121, 20261122, and 20261123. Keep `DATA_FRACTION_SEED=20261121`. The notebook derives `_frac10`, `_frac30`, and `_hard` suffixes; add `_ab3` to the run ID for the teacher dataset. Resume only with the same run ID and configuration.

Each notebook segment is capped at 10 minutes. Phase B has a 60-minute total GPU allocation; stop at that cap even if not all runs completed. Existing small-run timing suggests roughly 1-3 minutes per full-data CNN run, but this estimate has high uncertainty and excludes Drive overhead. Report actual elapsed time and completed runs from manifests. Do not use the frozen test to choose an architecture, target, or checkpoint.

After a run, download the complete ZIP from `deliveries/` and place it in `results/colab_inbox/`. Phase B v3 artifacts were trained with source bundle v11; because the current checkout can advance after packaging, ingest those archives against the exact source snapshot instead of weakening the source hash check:

```powershell
.venv-gpu\Scripts\python.exe scripts/ingest_colab_archives.py --source-bundle colab/source_bundle_20261121_v11.zip --artifact-dir results/colab_inbox --pattern 'cnn_v3_30k*.zip'
```

The helper verifies the source bundle sidecar and Python tree, dataset and artifact hashes, model inference, and local frozen-test leakage before copying artifacts into `results/colab/<run_id>/`. It can be rerun safely; it refuses to overwrite a different run. For older individual run directories, `scripts/ingest_colab_run.py` remains available when invoked from the exact source snapshot and with the dataset package used by that run. Evaluate only ingested checkpoints. Keep every run and negative result.

## Phase C V2: controlled AlphaZero initialization comparison

Use `colab/03_selfplay_az.ipynb` with source bundle v12 and the unchanged v3 dataset package. The validation-only comparison selected `checkpoints/cnn_v3_30k_best.pt`, from run `cnn_v3_30k_seed20261122` (Pons soft targets, full data; SHA and validation table are in `docs/CNN_V3_SELECTION.md`). Upload this file to `MyDrive/connect4/checkpoints/cnn_v3_30k_best.pt` for the warm-start run. The notebook defaults to `RUN_MODE='warm'`; run it first. Then change only `RUN_MODE='scratch'` and run the second job. The notebook derives the required run IDs and seeds: `az_v2_warm_seed20261121` / 20261121 and `az_v2_scratch_seed20261122` / 20261122. Scratch sets `INIT_CHECKPOINT=None` and starts from random weights.

Both runs use 40 iterations, 32 games per iteration, 16 parallel games, 400 simulations per move, 200 training updates per iteration, batch size 128, and snapshots at iterations 0, 5, 10, 15, 20, 25, 30, 35, and 40. Each segment is capped at 30 minutes; each run at 60 GPU minutes; the shared Phase C budget at 120 GPU minutes. Run jobs sequentially. The notebook writes `artifacts/az_v2_phase_c_budget.json` on Drive before launching each segment. It reserves time before execution, reconciles actual elapsed process time on normal completion, and leaves an interrupted reservation charged conservatively. Do not delete or reset this ledger. Resume with the same run ID and config.

Create the bundle locally after the notebook update and upload both ZIP and manifest sidecar:

```powershell
python -m scripts.package_colab_source --output colab/source_bundle_20261121_v12.zip
```

The run ZIP includes the full snapshot directory recursively and a copy of the budget ledger. Download both run ZIPs to `results/colab_inbox/<run_id>/` and ingest them locally. Compare snapshots with their iteration-0 model on validation at 400-simulation PUCT; exclude source games occurring in that checkpoint's replay. After the validation curve is finalized, evaluate the fixed iteration-40 checkpoints on the frozen test once. Since there is only one seed per initialization, treat warm-start versus scratch as a controlled case comparison, not a seed-generalized claim. Stop at the 120-minute total even if either curve is incomplete.

## Phase D: RL

Do not spend the current GPU allocation on another RL run. The validation diagnostic found a clear safety-filter contribution but did not distinguish the trained policy from reconstructed episode 0 under the filter. Reopen RL only after preregistering a new controlled experiment and its criteria.

## UTF-8 note

Keep Markdown documentation encoded as UTF-8. When editing from PowerShell, use an explicit UTF-8 writer or Python `Path.read_text(encoding='utf-8')` and `write_text(encoding='utf-8')`; verify added text for literal replacement or question-mark corruption before committing.
