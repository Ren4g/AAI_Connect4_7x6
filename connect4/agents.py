from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .engine import COLS, ConnectFour, GameResult, Player, ROWS

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


class Agent:
    def choose_action(self, state: ConnectFour) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)

    def choose_action(self, state: ConnectFour) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        return self.random.choice(actions)


def _window_score(window: list[int], player: int) -> float:
    opponent = 1 if player == 2 else 2
    own = window.count(player)
    other = window.count(opponent)
    empty = window.count(0)
    if own == 4:
        return 100000.0
    if own == 3 and empty == 1:
        return 100.0 if other == 0 else 10.0
    if own == 2 and empty == 2:
        return 10.0
    if other == 3 and empty == 1:
        return -120.0 if own == 0 else -20.0
    if other == 2 and empty == 2:
        return -8.0
    return 0.0


def evaluate(state: ConnectFour, player: Player) -> float:
    board = state.board
    score = 3.0 * sum(board[row][COLS // 2] == int(player) for row in range(ROWS))
    for row in range(ROWS):
        for column in range(COLS - 3):
            score += _window_score(board[row][column : column + 4], int(player))
    for row in range(ROWS - 3):
        for column in range(COLS):
            score += _window_score([board[row + offset][column] for offset in range(4)], int(player))
    for row in range(ROWS - 3):
        for column in range(COLS - 3):
            score += _window_score([board[row + offset][column + offset] for offset in range(4)], int(player))
            score += _window_score([board[row + 3 - offset][column + offset] for offset in range(4)], int(player))
    return score


class HeuristicAgent(Agent):
    def choose_action(self, state: ConnectFour) -> int:
        player = state.current_player
        best_score = float("-inf")
        best_actions: list[int] = []
        for action in state.legal_actions():
            child = state.clone()
            child.step(action)
            if child.result() is GameResult(int(player)):
                score = float("inf")
            else:
                score = evaluate(child, player)
            if score > best_score:
                best_score, best_actions = score, [action]
            elif score == best_score:
                best_actions.append(action)
        return min(best_actions, key=CENTER_ORDER.index)


class AlphaBetaAgent(Agent):
    def __init__(self, depth: int = 4, pruning: bool = True, move_ordering: bool = True) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.pruning = pruning
        self.move_ordering = move_ordering
        self.nodes = 0

    def choose_action(self, state: ConnectFour) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        self.nodes = 0
        player = state.current_player
        best_score = float("-inf")
        best_action = actions[0]
        alpha, beta = float("-inf"), float("inf")
        ordered_actions = sorted(actions, key=CENTER_ORDER.index) if self.move_ordering else actions
        for action in ordered_actions:
            child = state.clone()
            child.step(action)
            score = self._search(child, self.depth - 1, alpha, beta, player)
            if score > best_score:
                best_score, best_action = score, action
            if self.pruning:
                alpha = max(alpha, best_score)
        return best_action

    def _search(self, state: ConnectFour, depth: int, alpha: float, beta: float, root: Player) -> float:
        self.nodes += 1
        if state.result() is GameResult(int(root)):
            return 1000000.0 + depth
        if state.result() is GameResult(int(root.opponent)):
            return -1000000.0 - depth
        if state.is_terminal() or depth == 0:
            return evaluate(state, root)
        maximizing = state.current_player is root
        value = float("-inf") if maximizing else float("inf")
        actions = state.legal_actions()
        if self.move_ordering:
            actions = sorted(actions, key=CENTER_ORDER.index)
        for action in actions:
            child = state.clone()
            child.step(action)
            child_value = self._search(child, depth - 1, alpha, beta, root)
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


@dataclass
class _MCTSNode:
    state: ConnectFour
    parent: Optional["_MCTSNode"] = None
    action: Optional[int] = None
    visits: int = 0
    value: float = 0.0
    children: dict[int, "_MCTSNode"] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}


class MCTSAgent(Agent):
    def __init__(self, simulations: int = 500, exploration: float = math.sqrt(2), seed: Optional[int] = None) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.simulations = simulations
        self.exploration = exploration
        self.random = random.Random(seed)
        self.simulations_run = 0

    def choose_action(self, state: ConnectFour) -> int:
        root = _MCTSNode(state.clone())
        root_player = state.current_player
        for _ in range(self.simulations):
            node = root
            while not node.state.is_terminal() and not set(node.state.legal_actions()) - set(node.children):
                node = max(node.children.values(), key=self._ucb)
            unexpanded = [action for action in node.state.legal_actions() if action not in node.children]
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
                    if winner is GameResult(int(mover)):
                        node.value += 1.0
                    elif winner is not GameResult.DRAW:
                        node.value -= 1.0
                node = node.parent
        self.simulations_run = self.simulations
        return max(root.children.values(), key=lambda child: child.visits).action

    def _ucb(self, node: _MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        return node.value / node.visits + self.exploration * math.sqrt(math.log(node.parent.visits) / node.visits)

    def _rollout(self, state: ConnectFour) -> float:
        rollout = state.clone()
        while not rollout.is_terminal():
            rollout.step(self.random.choice(rollout.legal_actions()))
        if rollout.result() is GameResult.DRAW:
            return 0.0
        return 1.0 if rollout.result() is GameResult(int(state.current_player)) else -1.0


class PolicyAgent(Agent):
    """Level-5 policy/value agent backed by a callable model."""

    def __init__(self, policy: Callable[[ConnectFour], list[float]]) -> None:
        self.policy = policy

    def choose_action(self, state: ConnectFour) -> int:
        scores = self.policy(state)
        actions = state.legal_actions()
        if len(scores) != COLS:
            raise ValueError(f"Policy must return {COLS} scores")
        return max(actions, key=lambda action: scores[action])

