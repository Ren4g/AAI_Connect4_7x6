import json

from connect4.engine import ConnectFour
from scripts.prepare_ab3_teacher_dataset import label_worker
from scripts.generate_training_games import NoImmediateWinAgent, load_az_games


def test_az_iteration_log_converts_to_unique_game_records(tmp_path):
    source = tmp_path / "az.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({"iteration": 1, "game_actions": [[3, 2], [4, 1, 0]]}),
            json.dumps({"iteration": 2, "game_actions": [[6, 5, 4]]}),
        ]) + "\n",
        encoding="utf-8",
    )

    rows = load_az_games(source)

    assert [row["game_index"] for row in rows] == [0, 1, 2]
    assert [row["actions"] for row in rows] == [[3, 2], [4, 1, 0], [6, 5, 4]]
    assert [row["source_iteration"] for row in rows] == [1, 1, 2]
    assert all(row["record_type"] == "game" for row in rows)


def test_no_immediate_win_random_avoids_winning_move_if_available():
    state = ConnectFour()
    for action in (0, 6, 1, 6, 2, 5):
        state.step(action)

    action = NoImmediateWinAgent(seed=3).choose_action(state)

    assert action in state.legal_actions()
    assert action != 3


def test_ab3_teacher_label_selects_a_forced_immediate_win():
    start, actions = label_worker((0, [{"moves": [0, 6, 1, 6, 2, 5]}], 3))

    assert start == 0
    assert actions == [3]
