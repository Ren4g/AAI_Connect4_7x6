from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci_text(interval) -> str:
    if interval is None:
        return "—"
    return f"[{100 * interval[0]:.1f}; {100 * interval[1]:.1f}]"


def render(report: dict) -> str:
    lines = [
        "# Đánh giá agent bằng solver exact",
        "",
        f"- Run: `{report['run_id']}`; seed bootstrap: `{report['bootstrap']['seed']}`.",
        f"- Tập đóng băng: {report['positions']} vị trí; {report['unique_mirror_canonical_states']} lớp trạng thái đối xứng.",
        f"- SHA256 test: `{report['frozen_positions_sha256']}`.",
        f"- Solver: commit `{report['solver_commit']}`, binary SHA256 `{report['solver_binary_sha256']}`.",
        f"- Thời gian solver local: {report['solver_wall_seconds']:.1f}s với {report['solver_workers']} worker; không so thời gian này với Colab.",
        "- CI 95% bootstrap theo ván nguồn; các cặp dùng đúng cùng vị trí.",
        "",
        "## Kết quả tổng thể",
        "",
        "| Agent | Nước tối ưu | CI 95% | Giữ W/D/L | CI 95% | giây/nước local | node/nước |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["metrics"].items():
        all_metrics = values["all"]
        cost = values["local_cost"]
        lines.append(
            f"| {name} | {pct(all_metrics['optimal']['rate'])} | "
            f"{ci_text(all_metrics['optimal']['cluster_bootstrap_95'])} | "
            f"{pct(all_metrics['wdl_preserving']['rate'])} | "
            f"{ci_text(all_metrics['wdl_preserving']['cluster_bootstrap_95'])} | "
            f"{cost['mean_seconds_per_move']:.6f} | {cost['mean_nodes_per_move']:.1f} |"
        )
    lines += ["", "## Nước tối ưu theo giai đoạn", "",
              "| Agent | Khai cuộc n=500 | Trung cuộc n=500 | Tàn cuộc n=500 |",
              "|---|---:|---:|---:|"]
    for name, values in report["metrics"].items():
        parts = []
        for stage in ("opening", "middle", "endgame"):
            metric = values[f"stage_{stage}"]["optimal"]
            parts.append(f"{pct(metric['rate'])} {ci_text(metric['cluster_bootstrap_95'])}")
        lines.append(f"| {name} | " + " | ".join(parts) + " |")
    lines += ["", "## Vị trí có quyết định và giá trị lý thuyết", "",
              "Vị trí có quyết định được định nghĩa trước là vị trí có ít nhất một nước hợp lệ làm đổi dấu W/D/L so với nước tối ưu.",
              "Tập được chia theo dấu điểm tốt nhất của solver cho bên đi.", "",
              "| Agent | Có quyết định | Giá trị thắng | Giá trị hòa | Giá trị thua |",
              "|---|---:|---:|---:|---:|"]
    for name, values in report["metrics"].items():
        parts = [str(values["decision_positions"]["positions"])]
        for group in ("theoretical_win", "theoretical_draw", "theoretical_loss"):
            metric = values[group]["optimal"]
            parts.append(f"{pct(metric['rate'])} (n={values[group]['positions']})")
        lines.append(f"| {name} | " + " | ".join(parts) + " |")
    lines += ["", "## Chênh lệch ghép cặp với AlphaBeta depth 3", "",
              "Giá trị là agent đang xét trừ AlphaBeta depth 3; CI bootstrap theo cùng ván nguồn.", "",
              "| Agent | Nước tối ưu, tổng thể | Khai cuộc | Trung cuộc | Tàn cuộc |",
              "|---|---:|---:|---:|---:|"]
    for name, values in report["paired_difference_vs_ab3"].items():
        cells = [ci_text(values["optimal"])]
        for stage in ("opening", "middle", "endgame"):
            cells.append(ci_text(values[f"optimal_{stage}"]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Diễn giải", "",
              "Tỷ lệ exact-optimal không tăng đơn điệu theo độ sâu: AlphaBeta depth 5 cao hơn depth 6 trên tổng thể. "
              "Đây là kết quả của tìm kiếm hữu hạn cùng evaluator heuristic; độ sâu lớn hơn không tự đảm bảo chọn nước exact hơn.",
              "CNN chưa được đưa vào lần chạy này vì checkpoint hiện có trùng một phần tập đóng băng trong train/validation. "
              "Chỉ đánh giá CNN sau khi ingest checkpoint được huấn luyện từ gói dữ liệu sạch.", ""]
    return "\n".join(lines)


def render_svg(report: dict) -> str:
    agents = [name for name in report["agents"] if name.startswith("AlphaBeta-depth")]
    width, height = 760, 430
    left, right, top, bottom = 72, 24, 34, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.5, 1.0
    xs = [left + i * plot_w / (len(agents) - 1) for i in range(len(agents))]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="380" y="23" text-anchor="middle" font-family="sans-serif" font-size="16">AlphaBeta: độ sâu và tỷ lệ nước tối ưu</text>']
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.0%}</text>')
    points = []
    for index, name in enumerate(agents):
        metric = report["metrics"][name]["all"]["optimal"]
        rate = metric["rate"]
        low, high = metric["cluster_bootstrap_95"]
        y = top + (y_max - rate) / (y_max - y_min) * plot_h
        y_low = top + (y_max - high) / (y_max - y_min) * plot_h
        y_high = top + (y_max - low) / (y_max - y_min) * plot_h
        points.append(f"{xs[index]:.1f},{y:.1f}")
        svg.append(f'<line x1="{xs[index]:.1f}" y1="{y_low:.1f}" x2="{xs[index]:.1f}" y2="{y_high:.1f}" stroke="#174a7e" stroke-width="2"/>')
        svg.append(f'<circle cx="{xs[index]:.1f}" cy="{y:.1f}" r="5" fill="#174a7e"/>')
        svg.append(f'<text x="{xs[index]:.1f}" y="{height-bottom+23}" text-anchor="middle" font-family="sans-serif" font-size="12">{index+1}</text>')
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#174a7e" stroke-width="2"/>')
    svg.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">AlphaBeta search depth</text>')
    svg.append(f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">Exact-optimal rate (95% cluster bootstrap CI)</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tables and plots from a solver-evaluation JSON")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--svg", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    markdown, svg = Path(args.markdown), Path(args.svg)
    if markdown.exists() or svg.exists():
        raise FileExistsError("Refusing to overwrite a generated report or plot")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render(report), encoding="utf-8")
    svg.write_text(render_svg(report), encoding="utf-8")
    print(f"Wrote {markdown} and {svg}")


if __name__ == "__main__":
    main()
