# Kịch bản demo ngắn

1. Mở `docs/PROGRESS_REPORT.md`, chỉ ra câu hỏi nghiên cứu, baseline solver và giới hạn của checkpoint CNN cũ.
2. Mở biểu đồ `results/solver_depth_curve_frozen_20261120_seed20261125.svg`; giải thích exact-optimal và CI theo ván.
3. Chạy smoke CNN local:

   ```powershell
   python -m connect4.train_cnn --dataset colab/data/solverlabels_20261120_seed20261120_v2 --output-root artifacts/demo --run-id cnn-smoke-demo --seed 42 --device cuda --smoke --max-minutes 2
   ```

4. Chạy smoke self-play:

   ```powershell
   python -m connect4.selfplay --dataset colab/data/solverlabels_20261120_seed20261120_v2 --output-root artifacts/demo --run-id az-smoke-demo --seed 42 --device cuda --smoke --max-minutes 2
   ```

5. Chỉ ra `manifest.json`, `checkpoint_latest.pt`, replay buffer và notebook Colab; giải thích `--resume` sau khi runtime dừng.
6. Kết thúc bằng quy trình xác minh: trả artifact về local → ingest → solver evaluation → paired-opening arena → round-robin. Không trình bày smoke hoặc artifact chưa ingest như kết quả chơi.
