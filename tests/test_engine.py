import pytest

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

