from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence

from .engine import ConnectFour


@dataclass(frozen=True)
class SolverComparison:
    positions: int
    exact_matches: int
    match_rate: float


class ExternalSolver:
    """Adapter for a command-line Connect Four solver.

    The command receives a flattened 42-cell board and the player to move.
    It must print one legal column (0-6) to stdout. Use `{board}` and
    `{player}` placeholders in command_args.
    """

    def __init__(
        self,
        executable: str,
        command_args: Sequence[str] = ("{board}", "{player}"),
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.command_args = tuple(command_args)
        self.timeout_seconds = timeout_seconds

    def choose_action(self, state: ConnectFour) -> int:
        board = "".join(str(cell) for row in state.board for cell in row)
        arguments = [
            argument.format(board=board, player=int(state.current_player))
            for argument in self.command_args
        ]
        completed = subprocess.run(
            [self.executable, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        match = re.search(r"\b([0-6])\b", completed.stdout)
        if match is None:
            raise ValueError("External solver output does not contain a column from 0 to 6")
        action = int(match.group(1))
        if action not in state.legal_actions():
            raise ValueError(f"External solver returned illegal column {action}")
        return action


def compare_agent_to_solver(
    agent: object,
    solver: ExternalSolver,
    positions: Iterable[ConnectFour],
) -> SolverComparison:
    total = matches = 0
    for state in positions:
        if state.is_terminal():
            continue
        agent_action = agent.choose_action(state)
        solver_action = solver.choose_action(state)
        total += 1
        matches += agent_action == solver_action
    return SolverComparison(total, matches, matches / total if total else 0.0)
