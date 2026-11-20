from __future__ import annotations

from typing import Optional

from .engine import GameResult, Move, Player

ROWS = 6
COLS = 7
STRIDE = 7


class ConnectFour:
    """Standard 7x6 Connect Four game state using Bitboards."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.position: int = 0
        self.mask: int = 0
        self.moves_played: int = 0
        self.last_move: Optional[Move] = None
        self._result: GameResult = GameResult.ONGOING

    def _side_to_move(self) -> Player:
        return Player.ONE if self.moves_played % 2 == 0 else Player.TWO

    def player_bits(self, player: Player | int) -> int:
        """Return a player's stones without materializing the 6×7 grid."""
        requested = Player(int(player))
        if requested is self._side_to_move():
            return self.position
        return self.position ^ self.mask

    @property
    def current_player(self) -> Player:
        if self._result in (GameResult.PLAYER_ONE, GameResult.PLAYER_TWO):
            return Player(int(self._result))
        if self._result is GameResult.DRAW:
            return Player.ONE if self.moves_played % 2 else Player.TWO
        return self._side_to_move()

    def clone(self) -> "ConnectFour":
        state = ConnectFour.__new__(ConnectFour)
        state.position = self.position
        state.mask = self.mask
        state.moves_played = self.moves_played
        state.last_move = self.last_move
        state._result = self._result
        return state

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [column for column in range(COLS) if self._is_column_available(column)]

    def step(self, action: int) -> Move:
        if self.is_terminal():
            raise ValueError("Cannot play a move after the game has ended")
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(COLS):
            raise ValueError(f"Action must be an integer column from 0 to {COLS - 1}")
        if not self._is_column_available(action):
            raise ValueError(f"Column {action} is full")

        bottom_bit = 1 << (action * STRIDE)
        new_piece = (self.mask + bottom_bit) & ~self.mask

        idx = new_piece.bit_length() - 1
        row = (ROWS -1) - (idx - action * STRIDE)
        mover = self._side_to_move()
        move = Move(row, action, mover)

        self.position |= new_piece
        self.mask |= new_piece
        self.last_move = move
        self.moves_played += 1

        if self._has_connect_four(self.position):
            self._result = GameResult(int(move.player))
        elif self.moves_played == ROWS * COLS:
            self._result = GameResult.DRAW

        self.position ^= self.mask      # đổi góc nhìn sang đối thủ
        return move

    def is_terminal(self) -> bool:
        return self._result is not GameResult.ONGOING

    def result(self) -> GameResult:
        return self._result

    def to_grid(self) -> list[list[int]]:
        mine = self.position
        theirs = self.position ^ self.mask
        p1, p2 = (mine, theirs) if self._side_to_move() is Player.ONE else (theirs, mine)

        grid = []
        for row in range(ROWS):
            from_bottom = (ROWS - 1) - row
            line = []
            for col in range(COLS):
                bit = 1 << (col * STRIDE + from_bottom)
                line.append(1 if p1 & bit else 2 if p2 & bit else 0)
            grid.append(line)
        return grid

    @property
    def board(self) -> list[list[int]]:
        return self.to_grid()

    def render(self) -> str:
        mine = self.position
        theirs = self.position ^ self.mask
        p1, p2 = (mine, theirs) if self._side_to_move() is Player.ONE else (theirs, mine)

        lines = []
        for row in range(ROWS):
            from_bottom = (ROWS - 1) - row
            cells = []
            for col in range(COLS):
                bit = 1 << (col * STRIDE + from_bottom)
                cells.append("1" if p1 & bit else "2" if p2 & bit else ".")
            lines.append(" ".join(cells))
        return "\n".join(lines) + "\n" + " ".join(str(c) for c in range(COLS))

    def _is_column_available(self, col)->bool:
        topbit = 1 << (col * STRIDE + (ROWS -1))
        return (self.mask & topbit) == 0

    def _has_connect_four(self, bits: int) -> bool:
        for shift in (1, STRIDE, STRIDE - 1, STRIDE + 1):
            m = bits & (bits >> shift)
            if m & (m >> (2 * shift)):
                return True
        return False

def debug_bits(bitboard: int, title: str = "") -> None:
    print(f"--- {title} (Giá trị số: {bitboard}) ---")
    for r in range(STRIDE - 1, -1, -1):
        row_str = []
        for c in range(COLS):
            bit_index = c * STRIDE + r

            is_on = (bitboard >> bit_index) & 1
            row_str.append(str(is_on))
        label = "Đệm" if r == 6 else f"H{5 - r} "
        print(f"{label}: " + " ".join(row_str))

if __name__ == "__main__":
    g = ConnectFour()
    for a in [0, 1, 0, 1, 0, 1, 0]:
        g.step(a)
    assert g.result() == GameResult.PLAYER_ONE

    g = ConnectFour()
    for a in [0, 0, 0, 0, 0, 0]:
        g.step(a)
    assert g.legal_actions() == [1, 2, 3, 4, 5, 6]

    g = ConnectFour()
    for a in [0, 0, 1, 1, 2, 2, 3]:
        g.step(a)
    assert g.result() == GameResult.PLAYER_ONE

    g = ConnectFour()
    for a in [3, 3, 4]:
        g.step(a)
    print(g.render())
