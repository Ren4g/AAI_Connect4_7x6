from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

from connect4.agents import AlphaBetaAgent, HeuristicAgent, MCTSAgent, RandomAgent
from connect4.experiments import play_game
from connect4.engine import GameResult


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() or None


def checkpoint_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_az_games(path: Path) -> list[dict]:
    games = []
    index = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            for actions in row.get("game_actions", []):
                games.append({
                    "record_type": "game",
                    "pair": ["PUCT-AZ-selfplay-one", "PUCT-AZ-selfplay-two"],
                    "game_index": index,
                    "actions": [int(action) for action in actions],
                    "source_iteration": row.get("iteration"),
                })
                index += 1
    return games


class NoImmediateWinAgent(RandomAgent):
    """Random legal play that avoids ending a game when other moves exist."""

    def choose_action(self, state) -> int:
        legal = state.legal_actions()
        non_winning = []
        for action in legal:
            child = state.clone()
            child.step(int(action))
            if child.result() is not GameResult(int(state.current_player)):
                non_winning.append(action)
        return self.random.choice(non_winning or legal)


def make_agent(kind: str, seed: int, cnn_checkpoint: Path | None,
               device: str):
    if kind == "Random":
        return RandomAgent(seed)
    if kind == "NoImmediateWin":
        return NoImmediateWinAgent(seed)
    if kind == "Heuristic":
        return HeuristicAgent()
    if kind == "AlphaBeta-depth2":
        return AlphaBetaAgent(depth=2)
    if kind == "MCTS-50":
        return MCTSAgent(simulations=50, seed=seed)
    if kind == "CNN-policy":
        if cnn_checkpoint is None:
            raise ValueError("CNN-policy source requires --cnn-checkpoint")
        from connect4.agents import PolicyAgent
        return PolicyAgent.from_checkpoint(cnn_checkpoint, device=device)
    raise ValueError(f"Unknown source agent {kind!r}")


def generate(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = Path(args.cnn_checkpoint) if args.cnn_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    specs = [
        ("Random", "Random"),
        ("Heuristic", "Random"),
        ("AlphaBeta-depth2", "Random"),
        ("MCTS-50", "Random"),
        ("CNN-policy", "Random"),
    ]
    if args.long_games:
        specs = [("NoImmediateWin", "NoImmediateWin")]
    if args.smoke:
        specs = specs[:4]
    games_per_source = 2 if args.smoke else args.games_per_source
    if games_per_source < 1:
        raise ValueError("games-per-source must be positive")
    started = datetime.now().astimezone().isoformat()
    started_clock = time.perf_counter()
    rows = [{
        "record_type": "run",
        "run_id": output.stem,
        "seed": args.seed,
        "games_per_source": games_per_source,
        "sources": [list(pair) for pair in specs],
        "engine": "ConnectFour grid",
        "cnn_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "started_at": started,
    }]
    static_agents = {}

    def agent(kind: str, seed: int):
        if kind in {"Heuristic", "AlphaBeta-depth2", "CNN-policy"}:
            if kind not in static_agents:
                static_agents[kind] = make_agent(kind, seed, checkpoint, args.device)
            return static_agents[kind]
        return make_agent(kind, seed, checkpoint, args.device)

    for source_index, (style, opponent) in enumerate(specs):
        for index in range(games_per_source):
            seed = args.seed + source_index * 1_000_000 + index
            if style == opponent:
                first = agent(style, seed)
                second = agent(opponent, seed + 500_000)
            elif index % 2 == 0:
                first = agent(style, seed)
                second = agent(opponent, seed + 500_000)
            else:
                first = agent(opponent, seed + 500_000)
                second = agent(style, seed)
            first_name, second_name = ((style, opponent) if index % 2 == 0
                                       else (opponent, style))
            game = play_game(first, second, first_agent=first_name, second_agent=second_name)
            rows.append({
                "record_type": "game", "pair": [style, opponent],
                "game_index": index, "actions": list(game.actions),
                "result": int(game.result), "plies": game.moves,
            })
        print(f"generated {games_per_source} games for {style}|{opponent}", flush=True)

    if args.az_log is not None:
        az_path = Path(args.az_log)
        az_games = load_az_games(az_path)
        rows.extend(az_games)
    else:
        az_path = None

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")
    ended = datetime.now().astimezone().isoformat()
    manifest = {
        "run_id": output.stem,
        "phase": "training_source_games",
        "created_at": ended,
        "source_code_commit": git_commit(),
        "generator_sha256": checkpoint_sha256(Path(__file__)),
        "seed": args.seed,
        "games_per_source": games_per_source,
        "generated_games": len(rows) - 1 - (len(az_games) if args.az_log else 0),
        "az_games_imported": len(az_games) if args.az_log else 0,
        "source_pairs": [list(pair) for pair in specs],
        "cnn_checkpoint_path": str(checkpoint) if checkpoint else None,
        "cnn_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "az_log_path": str(az_path) if az_path else None,
        "az_log_sha256": checkpoint_sha256(az_path) if az_path else None,
        "output_path": str(output),
        "output_sha256": checkpoint_sha256(output),
        "started_at": started,
        "ended_at": ended,
        "elapsed_seconds": time.perf_counter() - started_clock,
        "smoke": args.smoke,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    with manifest_path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, indent=2)
        target.write("\n")
    print(json.dumps({key: manifest[key] for key in (
        "run_id", "generated_games", "az_games_imported", "elapsed_seconds",
        "output_sha256",
    )}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic game-source logs for solver labels")
    parser.add_argument("--games-per-source", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cnn-checkpoint", default="results/colab/cnn_seed20261121_colab1/checkpoint_best.pt")
    parser.add_argument("--az-log", default="results/colab/az_seed20261120_run1/train_log.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--long-games", action="store_true",
                        help="Generate only balanced no-immediate-win self-play games")
    args = parser.parse_args()
    if args.smoke:
        args.az_log = None
    generate(args)


if __name__ == "__main__":
    main()
