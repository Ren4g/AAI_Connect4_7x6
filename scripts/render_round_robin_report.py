"""Render paired round-robin JSON into a report and score-matrix SVG."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def interval_text(values: list[float]) -> str:
    return f"[{100 * values[0]:.1f}; {100 * values[1]:.1f}]"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * q
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] if lower == upper else ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def render_markdown(results: list[dict], metadata: dict, source: str,
                    games: list[dict] | None = None) -> str:
    agents = metadata["agents"]
    lines = [
        "# Round-robin cuối: báo cáo kết quả",
        "",
        f"- Nguồn: `{source}`; seed {metadata['seed']}; {metadata['games_per_pair']} ván/cặp.",
        f"- Engine `{metadata['engine']}`; opening {metadata['opening_plies']} ply; paired openings={metadata['paired_openings']}; bộ lọc `{metadata['opening_filter']}`.",
        f"- Sinh {metadata['opening_generation']['attempts']} opening, giữ {metadata['opening_generation']['accepted']}; loại {metadata['opening_generation']['filtered_immediate_win']} opening có thắng ngay và {metadata['opening_generation']['filtered_no_safe_action']} opening không có nước trả lời an toàn.",
        f"- Thời gian chỉ đo local trên cùng máy; ngân sách thời gian={metadata['time_budget_seconds']}.",
        (f"- Score là (thắng + 0,5 hòa)/số ván; CI 95% bootstrap {metadata.get('bootstrap', {}).get('replicates', 10000):,} lần, "
         f"seed {metadata.get('bootstrap', {}).get('seed', 20261119)}, đơn vị opening ghép cặp."),
        "",
        "## Ma trận score",
        "",
        "Ô hàng i/cột j là score của agent hàng i khi đấu agent cột j; ô đối xứng bằng `1 − score`.",
        "",
        "| Agent | " + " | ".join(agents) + " |",
        "|---|" + "---:|" * len(agents),
    ]
    pair_map = {(row["first_agent"], row["second_agent"]): row for row in results}
    for row_name in agents:
        cells = []
        for col_name in agents:
            if row_name == col_name:
                cells.append("—")
                continue
            row = pair_map.get((row_name, col_name))
            if row is not None:
                cells.append(f"{row['score']:.3f}")
            else:
                row = pair_map[(col_name, row_name)]
                cells.append(f"{1 - row['score']:.3f}")
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Kết quả từng cặp",
        "",
        "| Agent A | Agent B | W-D-L của A | Score A | CI 95% | Opening |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['first_agent']} | {row['second_agent']} | "
            f"{row['wins']}-{row['draws']}-{row['losses']} | {row['score']:.3f} | "
            f"{interval_text(row['score_bootstrap_95'])} | {row['unique_openings']} |"
        )
    lines += [
        "",
        "## Đo chi phí local theo agent",
        "",
        "| Agent | Nước đi | ms/nước | p95 ms/nước | node/nước | simulation/nước |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    totals = {name: {"moves": 0, "times": [], "nodes": 0, "simulations": 0}
              for name in agents}
    if games is not None:
        for game in games:
            names = {1: game["player_one_agent"], 2: game["player_two_agent"]}
            for player, seconds in zip(game["move_players"], game["move_times_seconds"]):
                totals[names[player]]["times"].append(seconds)
            for seat, player in ((0, 1), (1, 2)):
                item = totals[names[player]]
                item["moves"] += sum(acting_player == player for acting_player in game["move_players"])
                item["nodes"] += game["nodes_by_player"][seat]
                item["simulations"] += game["simulations_by_player"][seat]
    for name in agents:
        item = totals[name]
        moves = max(item["moves"], 1)
        mean_seconds = sum(item["times"]) / len(item["times"]) if item["times"] else 0.0
        p95_seconds = percentile(item["times"], .95) if item["times"] else None
        lines.append(
            f"| {name} | {item['moves']} | {1000 * mean_seconds:.3f} | "
            f"{1000 * p95_seconds:.3f}" if p95_seconds is not None else
            f"| {name} | {item['moves']} | {1000 * mean_seconds:.3f} | —"
        )
        lines[-1] += f" | {item['nodes'] / moves:.1f} | {item['simulations'] / moves:.1f} |"
    lines += [
        "",
        "Thời gian/nước và p95 tính trực tiếp từ JSONL replay; đơn vị thời gian không so sánh giữa Colab và local.",
        "",
    ]
    return "\n".join(lines)


def render_svg(results: list[dict], metadata: dict) -> str:
    agents = metadata["agents"]
    cell = 92
    left, top = 180, 100
    width, height = left + cell * len(agents) + 20, top + cell * len(agents) + 24
    pair_map = {(row["first_agent"], row["second_agent"]): row for row in results}
    esc = lambda text: (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="50%" y="28" text-anchor="middle" font-family="sans-serif" font-size="19">Round-robin score matrix</text>']
    for j, name in enumerate(agents):
        x = left + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top-12}" text-anchor="middle" font-family="sans-serif" font-size="11">{esc(name)}</text>')
    for i, row_name in enumerate(agents):
        y = top + i * cell
        parts.append(f'<text x="{left-12}" y="{y+cell/2+4}" text-anchor="end" font-family="sans-serif" font-size="11">{esc(row_name)}</text>')
        for j, col_name in enumerate(agents):
            x = left + j * cell
            if i == j:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#e5e7eb" stroke="white"/>')
                continue
            row = pair_map.get((row_name, col_name))
            score = row["score"] if row else 1 - pair_map[(col_name, row_name)]["score"]
            red, green = round(220 * (1 - score)), round(190 * score)
            fill = f"#{red:02x}{green:02x}70"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle" font-family="sans-serif" font-size="14">{score:.3f}</text>')
    parts += [f'<text x="{width/2}" y="{height-4}" text-anchor="middle" font-family="sans-serif" font-size="11">Row agent score against column agent; green = higher row score</text>', '</svg>']
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True, help="Replay JSONL produced by the same run")
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.markdown, args.svg):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    results, metadata = json.loads(args.input.read_text(encoding="utf-8")), json.loads(args.metadata.read_text(encoding="utf-8"))
    game_records = [json.loads(line) for line in args.games.read_text(encoding="utf-8").splitlines()]
    if not game_records or game_records[0].get("record_type") != "run":
        raise ValueError("replay JSONL is missing its run metadata record")
    games = [row for row in game_records[1:] if row.get("record_type") == "game"]
    if len(games) != sum(row["games"] for row in results):
        raise ValueError("replay game count does not match summarized pair results")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(results, metadata, args.input.as_posix(), games), encoding="utf-8")
    args.svg.write_text(render_svg(results, metadata), encoding="utf-8")
    print(f"Wrote {args.markdown} and {args.svg}")


if __name__ == "__main__":
    main()
