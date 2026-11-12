import pytest

from connect4.experiments import percentile, wilson_interval
from connect4.agents import AlphaBetaAgent
from connect4.engine import ConnectFour, GameResult, Player


def play(state: ConnectFour, columns: list[int]) -> None:
    for column in columns:
        state.step(column)


def test_horizontal_win() -> None:
    state = ConnectFour()
    play(state, [0, 0, 1, 1, 2, 2, 3])
    assert state.result() is GameResult.PLAYER_ONE


def test_vertical_win() -> None:
    state = ConnectFour()
    play(state, [0, 1, 0, 1, 0, 1, 0])
    assert state.result() is GameResult.PLAYER_ONE


def test_diagonal_win() -> None:
    state = ConnectFour()
    play(state, [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert state.result() is GameResult.PLAYER_ONE


def test_invalid_move_and_clone() -> None:
    state = ConnectFour()
    with pytest.raises(ValueError):
        state.step(7)
    clone = state.clone()
    clone.step(3)
    assert state.board != clone.board
    assert state.current_player is Player.ONE


def test_statistics_helpers() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    lower, upper = wilson_interval(50, 100)
    assert 0.4 < lower < 0.5 < upper < 0.6


def test_alpha_beta_ablation_options() -> None:
    state = ConnectFour()
    assert AlphaBetaAgent(depth=2, pruning=False).choose_action(state) in state.legal_actions()
    assert AlphaBetaAgent(depth=2, move_ordering=False).choose_action(state) in state.legal_actions()

