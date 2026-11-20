from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from connect4.engine import ConnectFour, GameResult
from connect4.solver import PonsSolver, SolverError, _sign


def winning_actions(state: ConnectFour, player: int) -> list[int]:
    wins = []
    for action in state.legal_actions():
        child = state.clone()
        child.step(action)
        if child.result() is GameResult(player):
            wins.append(action)
    return wins


def opponent_fork_replies(state_after_cnn: ConnectFour, opponent: int) -> list[dict[str, object]]:
    """Find opponent replies for which every CNN response allows an opponent win."""
    forks = []
    for reply in state_after_cnn.legal_actions():
        after_reply = state_after_cnn.clone()
        after_reply.step(reply)
        if after_reply.is_terminal():
            continue
        defenses = []
        all_responses_lose = True
        for cnn_response in after_reply.legal_actions():
            after_defense = after_reply.clone()
            after_defense.step(cnn_response)
            wins = (
                winning_actions(after_defense, opponent)
                if not after_defense.is_terminal() else []
            )
            if not wins:
                all_responses_lose = False
            defenses.append({"cnn_response_1_based": cnn_response + 1,
                             "opponent_wins_1_based": [column + 1 for column in wins]})
        if all_responses_lose:
            forks.append({"opponent_reply_1_based": reply + 1,
                          "cnn_responses": defenses})
    return forks


def analyze_losses(log_path: str, solver_path: str, timeout_seconds: float = 300.0) -> dict[str, object]:
    games = []
    with Path(log_path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("record_type") != "game":
                continue
            p1 = record["player_one_agent"]
            p2 = record["player_two_agent"]
            cnn_player = 1 if p1 == "CNN-Policy" else 2 if p2 == "CNN-Policy" else 0
            if not cnn_player:
                continue
            winner = record["result"]
            if winner == cnn_player or winner == 3:
                continue
            games.append((record, cnn_player))

    game_reports = []
    first_errors = Counter()
    first_wdl_errors = Counter()
    with PonsSolver(solver_path, timeout_seconds=timeout_seconds) as solver:
        for record, cnn_player in games:
            state = ConnectFour()
            first_error = None
            first_wdl_error = None
            actions = record["actions"]
            opening_plies = len(record.get("opening_actions", []))
            opening_actions = actions[:opening_plies]
            opening_state = ConnectFour()
            for move in opening_actions:
                opening_state.step(move)
            opening_scores = solver.scores(opening_actions) if opening_actions else None
            opening_best = (
                max(opening_scores[column] for column in opening_state.legal_actions())
                if opening_scores is not None else 0
            )
            opening_cnn_wdl = (
                _sign(opening_best)
                if int(opening_state.current_player) == cnn_player
                else -_sign(opening_best)
            )
            for index, action in enumerate(actions):
                # `move_players` in the JSONL contains only post-opening timed
                # moves; replay player identity from the full action index.
                player = 1 if index % 2 == 0 else 2
                if (
                    index >= opening_plies
                    and player == cnn_player
                    and (first_error is None or first_wdl_error is None)
                ):
                    if state.is_terminal():
                        raise ValueError(f"terminal prefix in game {record.get('game_index')}")
                    score_vector = solver.scores(actions[:index])
                    legal = state.legal_actions()
                    best = max(score_vector[column] for column in legal)
                    chosen = score_vector[action]
                    is_score_bad = chosen < best
                    is_wdl_bad = _sign(chosen) < _sign(best)
                    own_wins = winning_actions(state, cnn_player)
                    child = state.clone()
                    child.step(action)
                    opponent_wins = (
                        winning_actions(child, 3 - cnn_player)
                        if not child.is_terminal()
                        else []
                    )
                    fork_replies = (
                        opponent_fork_replies(child, 3 - cnn_player)
                        if not child.is_terminal() else []
                    )
                    classification = (
                        "missed_immediate_win"
                        if own_wins and action not in own_wins
                        else "conceded_fork"
                        if fork_replies
                        else "conceded_immediate_win"
                        if opponent_wins
                        else "deeper_or_positional"
                    )
                    if is_score_bad and first_error is None:
                        best_actions = [column for column in legal if score_vector[column] == best]
                        first_error = {
                            "action_index": index,
                            "ply_before_move": index,
                            "history": actions[:index],
                            "board": state.board,
                            "chosen_column_0_based": action,
                            "chosen_column_1_based": action + 1,
                            "chosen_exact_score": chosen,
                            "best_exact_score": best,
                            "wdl_preserving": not is_wdl_bad,
                            "best_columns_0_based": best_actions,
                            "best_columns_1_based": [column + 1 for column in best_actions],
                            "solver_score_by_column": list(score_vector),
                            "own_immediate_wins": [column + 1 for column in own_wins],
                            "opponent_immediate_wins_after_chosen": [
                                column + 1 for column in opponent_wins
                            ],
                            "opponent_fork_replies": fork_replies,
                            "classification": classification,
                        }
                        first_errors[classification] += 1
                    if is_wdl_bad:
                        first_wdl_error = {
                            "action_index": index,
                            "history": actions[:index],
                            "chosen_column_1_based": action + 1,
                            "chosen_exact_score": chosen,
                            "best_exact_score": best,
                            "best_columns_1_based": [column + 1 for column in legal if score_vector[column] == best],
                            "classification": classification if is_score_bad else "solver_wdl_degrading",
                        }
                        first_wdl_errors[first_wdl_error["classification"]] += 1
                state.step(action)
            game_reports.append(
                {
                    "pair": record["pair"],
                    "game_index": record["game_index"],
                    "cnn_player": cnn_player,
                    "result": record["result"],
                    "plies": len(actions),
                    "opening_actions": record.get("opening_actions", []),
                    "opening_best_score_for_side_to_move": opening_best,
                    "opening_cnn_wdl": opening_cnn_wdl,
                    "opening_already_forced_loss_for_cnn": opening_cnn_wdl < 0,
                    "first_solver_error": first_error,
                    "first_wdl_error": first_wdl_error,
                }
            )
    return {
        "game_log": log_path,
        "solver": solver_path,
        "solver_mode": "distance-sensitive exact score",
        "definition": "first CNN decision whose chosen action has a lower exact solver score than the best legal action; WDL degradation is recorded separately",
        "losing_games": len(games),
        "games_with_solver_error": sum(item["first_solver_error"] is not None for item in game_reports),
        "games_without_exact_score_error": sum(item["first_solver_error"] is None for item in game_reports),
        "first_error_classifications": dict(first_errors),
        "games_with_wdl_degrading_error": sum(item["first_wdl_error"] is not None for item in game_reports),
        "first_wdl_error_classifications": dict(first_wdl_errors),
        "games_already_forced_loss_at_opening_for_cnn": sum(
            item["opening_already_forced_loss_for_cnn"] for item in game_reports
        ),
        "games": game_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find CNN's first solver-suboptimal move in losses")
    parser.add_argument("--game-log", default="benchmarks/round_robin/round_robin_full_seed20261119.games.jsonl")
    parser.add_argument("--solver", default="solver/connect4/c4solver.exe")
    parser.add_argument("--solver-timeout", type=float, default=300.0)
    parser.add_argument("--output", default="results/cnn_first_errors_seed20261122.json")
    args = parser.parse_args()
    report = analyze_losses(args.game_log, args.solver, timeout_seconds=args.solver_timeout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"CNN losses={report['losing_games']}, first solver errors="
        f"{report['games_with_solver_error']}, no exact-score errors="
        f"{report['games_without_exact_score_error']}",
        flush=True,
    )
    print(f"Classification: {report['first_error_classifications']}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
