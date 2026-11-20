# Chạy các job GPU trên Google Colab

Các notebook chỉ điều phối CLI trong package `connect4`; dữ liệu chuẩn được lưu trên Drive để tiếp tục sau khi runtime bị ngắt. Không đưa token hay thông tin đăng nhập vào notebook.

## Chuẩn bị

1. Tạo thư mục `MyDrive/connect4/artifacts` và `MyDrive/connect4/data`.
2. Tải lên gói dữ liệu solver sạch tại `colab/data/solverlabels_20261120_seed20261120_v2/` sau khi được tạo: toàn bộ `train_validation.npz`, `hashes.json`, `frozen_test_hashes.json`, `manifest.json`.
3. Chạy `python scripts/package_colab_source.py --output colab/source_bundle_20261120_v5.zip`. Tải cả `colab/source_bundle_20261120_v5.zip` và `colab/source_bundle_20261120_v5.manifest.json` lên `MyDrive/connect4/`, giữ nguyên tên. Script loại môi trường ảo/cache, ghi commit, SHA256 ZIP và source tree hash; notebook tự kiểm tra cả hai hash.
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

Sau khi có checkpoint CNN được chọn bằng validation, tải nó lên Drive và gán đường dẫn vào `INIT_CHECKPOINT` trong `colab/03_selfplay_az.ipynb`. Nếu chưa có checkpoint hợp lệ, để `None` để bắt đầu từ trọng số ngẫu nhiên; ghi rõ nhánh khởi tạo khi báo cáo. Notebook dùng PUCT với prior policy, value của mạng, nhiễu Dirichlet ở gốc, nhiệt độ theo ply, replay buffer và cập nhật policy/value. Nhiều ván tự chơi chạy đồng thời để gom suy luận leaf thành batch. Mặc định mỗi lát chạy tối đa 60 phút, 100 iteration, 32 ván/iteration, 16 ván song song, 100 simulation/nước, 200 bước cập nhật/iteration; lần chạy lại cùng `RUN_ID` tự resume.

Ước lượng tài nguyên hiện còn độ bất định cao: smoke local trên RTX 3050 hoàn tất 2 ván, 2 simulation/nước, 2 cập nhật trong khoảng 1,04 giây tính thuần; cấu hình mặc định lớn hơn nhiều và chưa có benchmark đủ dài để ngoại suy chắc chắn. Dành ít nhất 60 phút cho lát Colab đầu để lấy throughput thật của runtime đó; checkpoint lưu trên Drive định kỳ mỗi 120 giây và tại điểm dừng. Đây là ước lượng vận hành, không phải số đo so sánh agent.

Đưa artifact cấp 8 về cùng thư mục inbox và ingest bằng cùng lệnh. Ingest kiểm tra các khóa trạng thái trong replay và ván đang dang dở đối chiếu với test đóng băng. Artifact `--smoke` bị từ chối để tránh nhầm với checkpoint thí nghiệm.

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
