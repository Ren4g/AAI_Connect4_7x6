from __future__ import annotations

from collections.abc import Sequence


def encode_planes(
    board: Sequence[Sequence[int]], player: int
) -> list[list[list[int]]]:
    """Encode a top-row-first 6x7 board as current-player and opponent planes."""
    if len(board) != 6 or any(len(row) != 7 for row in board):
        raise ValueError("board must have shape 6x7 with row 0 at the top")
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    opponent = 3 - player
    return [
        [[int(cell == target) for cell in row] for row in board]
        for target in (player, opponent)
    ]
