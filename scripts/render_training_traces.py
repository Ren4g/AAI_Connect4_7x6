"""Render RL and AlphaZero training diagnostics without implying held-out strength."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def points(values: list[float], x0: float, y0: float, width: float, height: float,
           low: float, high: float) -> str:
    if not values:
        return ""
    result = []
    for i, value in enumerate(values):
        x = x0 + (width * i / max(1, len(values) - 1))
        y = y0 + height * (high - value) / max(1e-12, high - low)
        result.append(f"{x:.1f},{y:.1f}")
    return " ".join(result)


def render(rl_rows: list[dict], az_rows: list[dict]) -> str:
    width, height = 940, 650
    left, right = 84, 35
    plot_width = width - left - right
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="470" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Training process diagnostics</text>',
           '<text x="470" y="52" text-anchor="middle" font-family="sans-serif" font-size="12">Training curves only; they are not independent strength evaluations</text>']

    # RL: rolling mean episode reward against the training opponent.
    rewards = [float(row["reward"]) for row in rl_rows if "reward" in row]
    window = 100
    rolling = [sum(rewards[max(0, i - window + 1):i + 1]) /
               len(rewards[max(0, i - window + 1):i + 1])
               for i in range(len(rewards))]
    x0, y0, pw, ph = left, 88, plot_width, 190
    svg.append('<text x="470" y="77" text-anchor="middle" font-family="sans-serif" font-size="15">RL: rolling mean episode reward (100 episodes), vs Random during training</text>')
    for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = y0 + ph * (1 - (tick + 1) / 2)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.1f}</text>')
    svg.append(f'<polyline points="{points(rolling,x0,y0,pw,ph,-1,1)}" fill="none" stroke="#315f9b" stroke-width="1.5"/>')
    svg.append(f'<text x="{x0+pw/2}" y="{y0+ph+22}" text-anchor="middle" font-family="sans-serif" font-size="11">Episode</text>')

    # AlphaZero: loss and self-play outcomes per iteration.
    losses = [float(row["mean_loss"]) for row in az_rows if "mean_loss" in row]
    draws = []
    for row in az_rows:
        total = sum(int(row.get(key, 0)) for key in
                    ("selfplay_wins_player1", "selfplay_wins_player2", "selfplay_draws"))
        if total:
            draws.append(int(row.get("selfplay_draws", 0)) / total)
    x0, y0, pw, ph = left, 356, plot_width, 190
    svg.append('<text x="470" y="338" text-anchor="middle" font-family="sans-serif" font-size="15">AlphaZero: per-iteration training loss and self-play draw fraction</text>')
    if losses:
        lo, hi = min(losses), max(losses)
        pad = max(.05, (hi - lo) * .1)
        lo, hi = lo - pad, hi + pad
        for i in range(5):
            value = lo + (hi - lo) * i / 4
            y = y0 + ph * (1 - i / 4)
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#eee"/>')
            svg.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{value:.2f}</text>')
        svg.append(f'<polyline points="{points(losses,x0,y0,pw,ph,lo,hi)}" fill="none" stroke="#a04b32" stroke-width="1.6"/>')
    if draws:
        draw_points = []
        for i, value in enumerate(draws):
            x = x0 + pw * i / max(1, len(draws) - 1)
            y = y0 + ph * (1 - value)
            draw_points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline points="{" ".join(draw_points)}" fill="none" stroke="#31856a" stroke-width="1.6"/>')
    svg += [
        f'<text x="{x0+pw/2}" y="{y0+ph+22}" text-anchor="middle" font-family="sans-serif" font-size="11">Iteration</text>',
        '<line x1="305" y1="590" x2="335" y2="590" stroke="#a04b32" stroke-width="2"/><text x="342" y="594" font-family="sans-serif" font-size="12">Mean training loss</text>',
        '<line x1="510" y1="590" x2="540" y2="590" stroke="#31856a" stroke-width="2"/><text x="547" y="594" font-family="sans-serif" font-size="12">Self-play draw fraction</text>',
        '<text x="470" y="628" text-anchor="middle" font-family="sans-serif" font-size="11">Self-play outcomes measure the current self-play population; use the locked solver test and arena for strength claims.</text>',
        '</svg>',
    ]
    return "\n".join(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-log", type=Path, default=Path("results/colab/rl_seed20261120_run1/train_log.jsonl"))
    parser.add_argument("--az-log", type=Path, default=Path("results/colab/az_seed20261120_run1/train_log.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/training_process_trace.svg"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose another path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(read_jsonl(args.rl_log), read_jsonl(args.az_log)), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
