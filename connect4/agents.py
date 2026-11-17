from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .engine import COLS, GameResult, Player, ROWS
from .state import GameState

CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


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


def evaluate(state: GameState, player: Player | int) -> float:
    player = Player(int(player))
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
    def __init__(self, depth: int = 4, pruning: bool = True, move_ordering: bool = True) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.pruning = pruning
        self.move_ordering = move_ordering
        self.nodes = 0

    def choose_action(self, state: GameState) -> int:
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

    def _search(self, state: GameState, depth: int, alpha: float, beta: float, root: Player) -> float:
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
    state: GameState
    parent: Optional["_MCTSNode"] = None
    action: Optional[int] = None
    visits: int = 0
    value: float = 0.0
    children: dict[int, "_MCTSNode"] = field(default_factory=dict)


class MCTSAgent(Agent):
    def __init__(self, simulations: int = 500, exploration: float = math.sqrt(2), seed: Optional[int] = None) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.simulations = simulations
        self.exploration = exploration
        self.random = random.Random(seed)
        self.simulations_run = 0

    def choose_action(self, state: GameState) -> int:
        if not state.legal_actions():
            raise ValueError("No legal actions available")
        root = _MCTSNode(state.clone())
        self.simulations_run = 0
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
                    if winner == GameResult(int(mover)):
                        node.value += 1.0
                    elif winner != GameResult.DRAW:
                        node.value -= 1.0
                node = node.parent
        self.simulations_run = self.simulations
        return max(root.children.values(), key=lambda child: child.visits).action

    def _ucb(self, node: _MCTSNode) -> float:
        if node.visits == 0:
            return float("inf")
        return node.value / node.visits + self.exploration * math.sqrt(math.log(node.parent.visits) / node.visits)

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

        checkpoint = torch.load(path, map_location=device)
        model = ConnectFourCNN().to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        def policy(state: GameState) -> Sequence[float]:
            player = int(state.current_player)
            opponent = 1 if player == 2 else 2
            planes = [
                [[float(cell == player) for cell in row] for row in state.board],
                [[float(cell == opponent) for cell in row] for row in state.board],
            ]
            inputs = torch.tensor([planes], dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits, _ = model(inputs)
            return logits[0].detach().cpu().tolist()

        return cls(policy)

    def choose_action(self, state: GameState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("No legal actions available")
        scores = self.policy(state)
        if len(scores) != COLS:
            raise ValueError(f"Policy must return {COLS} scores")
        return max(actions, key=lambda action: (float(scores[action]), -CENTER_ORDER.index(action)))

