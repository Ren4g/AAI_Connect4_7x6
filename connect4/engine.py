from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

ROWS = 6
COLS = 7


class Player(IntEnum):
    ONE = 1
    TWO = 2

    @property
    def opponent(self) -> "Player":
        return Player.TWO if self is Player.ONE else Player.ONE


class GameResult(IntEnum):
    ONGOING = 0
    PLAYER_ONE = 1
    PLAYER_TWO = 2
    DRAW = 3


@dataclass(frozen=True)
class Move:
    row: int
    column: int
    player: Player


class ConnectFour:
    """Standard 7x6 Connect Four game state."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.current_player = Player.ONE
        self.last_move: Optional[Move] = None
        self._result = GameResult.ONGOING

    def clone(self) -> "ConnectFour":
        state = ConnectFour()
        state.board = deepcopy(self.board)
        state.current_player = self.current_player
        state.last_move = self.last_move
        state._result = self._result
        return state

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [column for column in range(COLS) if self.board[0][column] == 0]

    def step(self, action: int) -> Move:
        if self.is_terminal():
            raise ValueError("Cannot play a move after the game has ended")
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(COLS):
            raise ValueError(f"Action must be an integer column from 0 to {COLS - 1}")
        if self.board[0][action] != 0:
            raise ValueError(f"Column {action} is full")

        row = next(row for row in range(ROWS - 1, -1, -1) if self.board[row][action] == 0)
        move = Move(row, action, self.current_player)
        self.board[row][action] = int(self.current_player)
        self.last_move = move
        if self._has_connect_four(row, action, int(self.current_player)):
            self._result = GameResult(int(self.current_player))
        elif not self.legal_actions():
            self._result = GameResult.DRAW
        else:
            self.current_player = self.current_player.opponent
        return move

    def is_terminal(self) -> bool:
        return self._result is not GameResult.ONGOING

    def result(self) -> GameResult:
        return self._result

    def render(self) -> str:
        rows = [" ".join("." if cell == 0 else str(cell) for cell in row) for row in self.board]
        return "\n".join(rows) + "\n" + " ".join(str(column) for column in range(COLS))

    def _has_connect_four(self, row: int, column: int, player: int) -> bool:
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row_step, column_step in directions:
            count = 1
            for sign in (-1, 1):
                current_row = row + sign * row_step
                current_column = column + sign * column_step
                while (
                    0 <= current_row < ROWS
                    and 0 <= current_column < COLS
                    and self.board[current_row][current_column] == player
                ):
                    count += 1
                    current_row += sign * row_step
                    current_column += sign * column_step
            if count >= 4:
                return True
        return False

