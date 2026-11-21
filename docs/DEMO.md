# Kịch bản demo ngắn

## Trình tự trình bày

1. Mở `docs/PROGRESS_REPORT.md` để nêu câu hỏi nghiên cứu và hiện trạng tám cấp agent.
2. Mở `docs/LEARNING_RESULTS.md`. Giải thích tỷ lệ nước tối ưu/WDL-preserving và CI bootstrap theo ván nguồn; đối chiếu CNN, RL, PUCT với AlphaBeta. Chỉ ra trạng thái cổng CNN trong bảng ghép cặp.
3. Mở `docs/ROUND_ROBIN_FINAL.md` và `docs/round_robin_final.svg`. Chọn một vài cặp đáng chú ý, đọc W-D-L, score, CI theo opening và số opening hợp lệ. Nhấn mạnh thời gian là đo local.
4. Mở JSONL replay để cho thấy mỗi ván lưu opening, ghế, chuỗi nước và kết quả. Mở manifest một run Colab để trình bày checkpoint hash, resume và kiểm định leakage; artifact chỉ được đánh giá sau ingest.
5. Kết thúc bằng giới hạn: các run RL/AZ hiện là bằng chứng cho checkpoint/seed cụ thể; không ngoại suy thành đường cong học dài hạn.

## Lệnh tái tạo báo cáo từ kết quả đã lưu

```powershell
python scripts/render_learning_report.py
python scripts/render_round_robin_report.py --input benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.json --metadata benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.meta.json --games benchmarks/round_robin/round_robin_final_colab_agents_seed20261121_v2.games.jsonl --markdown docs/ROUND_ROBIN_FINAL.md --svg docs/round_robin_final.svg
```

Không chạy lại round-robin trong buổi demo: đây là benchmark dài. Dùng JSON/JSONL đã lưu để trình bày và phát lại ván.
