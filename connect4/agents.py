from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .engine import COLS, GameResult, Player, ROWS
from .state import GameState

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)
_CENTER_RANK = {column: rank for rank, column in enumerate(CENTER_ORDER)}
_BIT_STRIDE = 7
_CENTER_MASK = sum(1 << (3 * _BIT_STRIDE + row) for row in range(6))
_BITBOARD_WINDOWS = tuple(
    sum(1 << ((column + dx * offset) * _BIT_STRIDE + row + dy * offset) for offset in range(4))
    for dx, dy, columns, rows in (
        (1, 0, range(4), range(6)),
        (0, 1, range(7), range(3)),
        (1, 1, range(4), range(3)),
        (1, -1, range(4), range(3, 6)),
    )
    for column in columns
    for row in rows
)
_WINDOW_COORDS = tuple(
    tuple((ROWS - 1 - (row + dy * offset), column + dx * offset) for offset in range(4))
    for dx, dy, columns, rows in (
        (1, 0, range(4), range(6)),
        (0, 1, range(7), range(3)),
        (1, 1, range(4), range(3)),
        (1, -1, range(4), range(3, 6)),
    )
    for column in columns
    for row in rows
)


class Agent:
    def choose_action(self, state: GameState) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)

    def choose_action(self, state: GameState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        return self.random.choice(actions)


def _window_score(window: list[int], player: int, playable: bool = False) -> float:
    opponent = 1 if player == 2 else 2
    own = window.count(player)
    other = window.count(opponent)
    empty = window.count(0)
    if own == 4:
        return 100000.0
    if own == 3 and empty == 1:
        return 100.0 if playable else 20.0
    if own == 2 and empty == 2:
        return 10.0
    if other == 3 and empty == 1:
        return -120.0 if playable else -20.0
    if other == 2 and empty == 2:
        return -8.0
    return 0.0


def evaluate(state: GameState, player: Player | int) -> float:
    player = Player(int(player))
    player_bits = getattr(state, "player_bits", None)
    if player_bits is not None:
        own_bits = player_bits(player)
        opponent_bits = player_bits(player.opponent)
        score = 3.0 * (own_bits & _CENTER_MASK).bit_count()
        occupied = own_bits | opponent_bits
        for window in _BITBOARD_WINDOWS:
            own = (own_bits & window).bit_count()
            other = (opponent_bits & window).bit_count()
            empty = 4 - own - other
            playable = False
            if empty == 1 and (own == 3 or other == 3):
                missing = window & ~occupied
                missing_index = missing.bit_length() - 1
                row_from_bottom = missing_index % _BIT_STRIDE
                playable = row_from_bottom == 0 or bool(occupied & (1 << (missing_index - 1)))
            score += _window_score_counts(own, other, empty, playable)
        return score

    board = state.board
    score = 3.0 * sum(board[row][COLS // 2] == int(player) for row in range(ROWS))
    for coordinates in _WINDOW_COORDS:
        window = [board[row][column] for row, column in coordinates]
        empty_cells = [
            (row, column)
            for (row, column), cell in zip(coordinates, window)
            if cell == 0
        ]
        playable = False
        if len(empty_cells) == 1:
            row, column = empty_cells[0]
            playable = row == ROWS - 1 or board[row + 1][column] != 0
        score += _window_score(window, int(player), playable)
    return score


def _window_score_counts(own: int, other: int, empty: int, playable: bool = False) -> float:
    if own == 4:
        return 100000.0
    if own == 3 and empty == 1:
        return 100.0 if playable else 20.0
    if own == 2 and empty == 2:
        return 10.0
    if other == 3 and empty == 1:
        return -120.0 if playable else -20.0
    if other == 2 and empty == 2:
        return -8.0
    return 0.0


class HeuristicAgent(Agent):
    def choose_action(self, state: GameState) -> int:
        player = state.current_player
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")

        winning = self._winning_actions(state, player)
        if winning:
            return min(winning, key=CENTER_ORDER.index)

        safe_actions: list[tuple[float, int]] = []
        for action in actions:
            child = state.clone()
            child.step(action)
            if child.is_terminal():
                continue
            if not self._winning_actions(child, player.opponent):
                safe_actions.append((evaluate(child, player), action))

        candidates = safe_actions
        if not candidates:
            candidates = []
            for action in actions:
                child = state.clone()
                child.step(action)
                candidates.append((evaluate(child, player), action))
        best_score = max(score for score, _ in candidates)
        best_actions = [action for score, action in candidates if score == best_score]
        return min(best_actions, key=CENTER_ORDER.index)

    @staticmethod
    def _winning_actions(state: GameState, player: Player) -> list[int]:
        wins: list[int] = []
        for action in state.legal_actions():
            child = state.clone()
            child.step(action)
            if child.result() == GameResult(int(player)):
                wins.append(action)
        return wins


class AlphaBetaAgent(Agent):
    def __init__(
        self,
        depth: int = 4,
        pruning: bool = True,
        move_ordering: bool = True,
        time_budget_seconds: float | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        if time_budget_seconds is not None and time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        self.depth = depth
        self.pruning = pruning
        self.move_ordering = move_ordering
        self.time_budget_seconds = time_budget_seconds
        self.nodes = 0

    def choose_action(self, state: GameState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        self.nodes = 0
        player = state.current_player
        ordered_actions = sorted(actions, key=_CENTER_RANK.__getitem__) if self.move_ordering else actions
        if self.time_budget_seconds is not None:
            # Leave a small guard band for returning the action and timing overhead.
            deadline = time.perf_counter() + self.time_budget_seconds * 0.95
            best_action = ordered_actions[0]
            for depth in range(1, self.depth + 1):
                try:
                    best_action = self._search_root(state, ordered_actions, depth, player, deadline)
                except _SearchTimeout:
                    break
            return best_action

        return self._search_root(state, ordered_actions, self.depth, player, None)

    def _search_root(
        self,
        state: GameState,
        actions: list[int],
        depth: int,
        player: Player,
        deadline: float | None,
    ) -> int:
        best_score = float("-inf")
        best_action = actions[0]
        alpha, beta = float("-inf"), float("inf")
        for action in actions:
            child = state.clone()
            child.step(action)
            score = self._search(child, depth - 1, alpha, beta, player, deadline)
            if score > best_score:
                best_score, best_action = score, action
            if self.pruning:
                alpha = max(alpha, best_score)
        return best_action

    def _search(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root: Player,
        deadline: float | None = None,
    ) -> float:
        if deadline is not None and time.perf_counter() >= deadline:
            raise _SearchTimeout
        self.nodes += 1
        if state.result() == GameResult(int(root)):
            return 1000000.0 + depth
        if state.result() == GameResult(int(root.opponent)):
            return -1000000.0 - depth
        if state.is_terminal() or depth == 0:
            return evaluate(state, root)
        maximizing = state.current_player == root
        value = float("-inf") if maximizing else float("inf")
        actions = state.legal_actions()
        if self.move_ordering:
            actions = sorted(actions, key=_CENTER_RANK.__getitem__)
        for action in actions:
            child = state.clone()
            child.step(action)
            child_value = self._search(child, depth - 1, alpha, beta, root, deadline)
            if maximizing:
                value = max(value, child_value)
                if self.pruning:
                    alpha = max(alpha, value)
            else:
                value = min(value, child_value)
                if self.pruning:
                    beta = min(beta, value)
            if self.pruning and beta <= alpha:
                break
        return value


class _SearchTimeout(Exception):
    pass


@dataclass
class _MCTSNode:
    state: GameState
    parent: Optional["_MCTSNode"] = None
    action: Optional[int] = None
    visits: int = 0
    value: float = 0.0
    children: dict[int, "_MCTSNode"] = field(default_factory=dict)


class MCTSAgent(Agent):
    def __init__(
        self,
        simulations: int = 500,
        exploration: float = math.sqrt(2),
        seed: Optional[int] = None,
        time_budget_seconds: float | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if time_budget_seconds is not None and time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        self.simulations = simulations
        self.exploration = exploration
        self.random = random.Random(seed)
        self.time_budget_seconds = time_budget_seconds
        self.simulations_run = 0

    def choose_action(self, state: GameState) -> int:
        if not state.legal_actions():
            raise ValueError("No legal actions available")
        root = _MCTSNode(state.clone())
        self.simulations_run = 0
        deadline = (
            # Rollouts finish as a unit, so reserve 5% to keep wall time within budget.
            time.perf_counter() + self.time_budget_seconds * 0.95
            if self.time_budget_seconds is not None else None
        )
        for _ in range(self.simulations):
            if deadline is not None and self.simulations_run and time.perf_counter() >= deadline:
                break
            node = root
            while not node.state.is_terminal():
                actions = node.state.legal_actions()
                unexpanded = [action for action in actions if action not in node.children]
                if unexpanded:
                    break
                node = max(node.children.values(), key=self._ucb)
            else:
                actions = []
                unexpanded = []
            if unexpanded:
                action = self.random.choice(unexpanded)
                child_state = node.state.clone()
                child_state.step(action)
                node = _MCTSNode(child_state, node, action)
                node.parent.children[action] = node
            winner = self._rollout(node.state)   # rollout trả về rollout.result()
            while node is not None:
                node.visits += 1
                if node.parent is not None:
                    mover = node.parent.state.current_player
                    node.value += self._reward(winner, mover)
                node = node.parent
            self.simulations_run += 1
        return max(root.children.values(), key=lambda child: child.visits).action

    def _ucb(self, node: _MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        return node.value / node.visits + self.exploration * math.sqrt(math.log(node.parent.visits) / node.visits)

    @staticmethod
    def _reward(winner: GameResult, player: Player) -> float:
        """Return a UCT reward on [0, 1] from this edge's mover perspective."""
        if winner is GameResult.DRAW:
            return 0.5
        return 1.0 if winner is GameResult(int(player)) else 0.0

    def _rollout(self, state: GameState) -> GameResult:
        rollout = state.clone()
        while not rollout.is_terminal():
            rollout.step(self.random.choice(rollout.legal_actions()))
        return rollout.result()

class PolicyAgent(Agent):
    """Level-5 policy/value agent backed by a callable model."""

    def __init__(self, policy: Callable[[GameState], Sequence[float]]) -> None:
        self.policy = policy

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "PolicyAgent":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install PyTorch to load a trained policy") from error
        from .model import ConnectFourCNN
        from .encoding import encode_planes

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = ConnectFourCNN().to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        def policy(state: GameState) -> Sequence[float]:
            player = int(state.current_player)
            inputs = torch.as_tensor(
                encode_planes(state.board, player), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.inference_mode():
                logits, _ = model(inputs)
            return logits[0].detach().cpu().tolist()

        return cls(policy)

    def choose_action(self, state: GameState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        player = state.current_player
        winning = HeuristicAgent._winning_actions(state, player)
        if winning:
            return min(winning, key=CENTER_ORDER.index)

        safe_actions: list[int] = []
        for action in actions:
            child = state.clone()
            child.step(action)
            if child.is_terminal():
                continue
            if not HeuristicAgent._winning_actions(child, player.opponent):
                safe_actions.append(action)
        if safe_actions:
            actions = safe_actions

        scores = self.policy(state)
        if len(scores) != COLS:
            raise ValueError(f"Policy must return {COLS} scores")
        return max(actions, key=lambda action: (float(scores[action]), -CENTER_ORDER.index(action)))

