"""Score agents against Pascal Pons' perfect Connect Four solver.

c4solver reads one position per line from stdin, written as the sequence of
1-based columns that were played ("4453"). With -a it prints
"<sequence> s1 s2 ... s7": the exact score of playing each column. Positive =
win, negative = loss, 0 = draw; with the strong solver a larger magnitude means
the result arrives sooner. Illegal columns are ignored here by using the
engine's legal_actions().

The solver only needs the move history, so positions are stored as tuples of
0-based columns (the engine's convention) and replayed through ConnectFour.
"""
from __future__ import annotations

import json
import queue
import random
import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .engine import COLS, ConnectFour


class SolverError(RuntimeError):
    """The solver process failed, timed out, or printed something unexpected."""


def encode_moves(moves: Sequence[int]) -> str:
    """Engine columns (0-6) -> solver notation (1-7)."""
    return "".join(str(move + 1) for move in moves)


class PonsSolver:
    """One long-lived c4solver process in analyze mode (-a).

    `command` is the executable, or a list such as ["wsl", "./c4solver"].
    Not thread-safe: use one instance per thread or process.
    """

    def __init__(
        self,
        command: str | Sequence[str],
        book: str | Path | None = None,
        weak: bool = False,
        timeout_seconds: float = 120.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        args = [command] if isinstance(command, str) else list(command)
        args.append("-a")
        if weak:
            args.append("-w")  # only win/draw/loss, much faster, sign-only scores
        if book is not None:
            args += ["-b", str(book)]
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[int, ...]] = {}
        self._lines: queue.Queue[str | None] = queue.Queue()
        # stderr is inherited so a missing opening book is visible in the terminal.
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def scores(self, moves: Sequence[int]) -> tuple[int, ...]:
        """Score of every column for the side to move after `moves`.

        `moves` must be a legal, unfinished game: the solver prints nothing on
        stdout for invalid input, so validate with the engine before calling.
        """
        if not moves:
            raise ValueError("at least one move is required")
        return self.scores_batch([moves])[0]

    def scores_batch(self, positions: Iterable[Sequence[int]]) -> list[tuple[int, ...]]:
        """Analyze many histories with one stdin write and one long-lived process.

        Validate each history with the local engine before sending it. The
        solver prints one output line for every valid, unfinished position.
        Duplicate histories within the batch and the process cache are reused.
        """
        histories = [tuple(int(action) for action in moves) for moves in positions]
        if not histories:
            return []
        keys = []
        for moves in histories:
            if not moves:
                raise ValueError("each position must contain at least one move")
            state = ConnectFour()
            for index, action in enumerate(moves):
                state.step(action)
                if state.is_terminal() and index != len(moves) - 1:
                    raise ValueError("position history continues after a terminal move")
            if state.is_terminal():
                raise ValueError("solver positions must be unfinished")
            keys.append(encode_moves(moves))

        missing = list(dict.fromkeys(key for key in keys if key not in self._cache))
        if missing:
            try:
                assert self._process.stdin is not None
                self._process.stdin.write("".join(key + "\n" for key in missing))
                self._process.stdin.flush()
                for expected in missing:
                    line = self._lines.get(timeout=self.timeout_seconds)
                    if line is None:
                        raise SolverError("solver process exited while reading batch")
                    parts = line.split()
                    if len(parts) != 1 + COLS or parts[0] != expected:
                        raise SolverError(f"unexpected solver output for {expected!r}: {line!r}")
                    self._cache[expected] = tuple(int(value) for value in parts[1:])
            except queue.Empty as error:
                self.close()
                raise SolverError("solver timed out while reading a batch") from error
            except OSError as error:
                raise SolverError("solver process is not running") from error
        return [self._cache[key] for key in keys]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()

    def __enter__(self) -> "PonsSolver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class Tally:
    positions: int = 0
    optimal: int = 0  #
    value_preserving: int = 0

    @property
    def optimal_rate(self) -> float:
        return self.optimal / self.positions if self.positions else 0.0

    @property
    def value_preserving_rate(self) -> float:
        return self.value_preserving / self.positions if self.positions else 0.0


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def phase(ply: int) -> str:
    return "opening" if ply <= 10 else "middle" if ply <= 24 else "endgame"


def evaluate_against_solver(
    agent: object,
    solver: PonsSolver,
    positions: Iterable[Sequence[int]],
) -> dict[str, Tally]:
    """Return tallies for "all" and for each game phase.

    With a weak solver the scores are signs only, so `optimal` and
    `value_preserving` coincide.
    """
    tallies: dict[str, Tally] = defaultdict(Tally)
    for moves in positions:
        state = ConnectFour()
        for move in moves:
            state.step(move)
        if state.is_terminal():
            continue
        scores = solver.scores(moves)
        legal = state.legal_actions()
        best = max(scores[column] for column in legal)
        action = int(agent.choose_action(state))
        if action not in legal:
            raise ValueError(f"Agent returned illegal column {action}")
        is_optimal = scores[action] == best
        keeps_value = _sign(scores[action]) == _sign(best)
        for key in ("all", phase(len(moves))):
            tally = tallies[key]
            tally.positions += 1
            tally.optimal += is_optimal
            tally.value_preserving += keeps_value
    return dict(tallies)


def positions_from_logs(
    path: str | Path,
    per_game: int = 3,
    min_ply: int = 7,
    seed: int = 0,
) -> list[tuple[int, ...]]:
    """Sample unfinished positions from the JSONL game log.

    Every proper prefix of a finished game is an unfinished position. Mirror
    images are skipped because they have mirrored scores (no new information).
    """
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    positions: list[tuple[int, ...]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("record_type") != "game":
                continue
            actions = record["actions"]
            plies = list(range(min_ply, len(actions)))
            for ply in rng.sample(plies, min(per_game, len(plies))):
                prefix = tuple(actions[:ply])
                mirror = tuple(COLS - 1 - column for column in prefix)
                if prefix in seen or mirror in seen:
                    continue
                seen.add(prefix)
                positions.append(prefix)
    return positions
