import pytest

from connect4.experiments import percentile, wilson_interval
from connect4.agents import AlphaBetaAgent
from connect4.engine import ConnectFour, GameResult, Player
from connect4.engine_bit import ConnectFour as BitboardConnectFour
from connect4.agents import HeuristicAgent, MCTSAgent, PolicyAgent, RandomAgent, evaluate
from connect4.training import split_training_records

ENGINES = (ConnectFour, BitboardConnectFour)


def play(state: ConnectFour, columns: list[int]) -> None:
    for column in columns:
        state.step(column)


@pytest.mark.parametrize("engine", ENGINES)
def test_horizontal_win(engine) -> None:
    state = engine()
    play(state, [0, 0, 1, 1, 2, 2, 3])
    assert state.result() is GameResult.PLAYER_ONE


@pytest.mark.parametrize("engine", ENGINES)
def test_vertical_win(engine) -> None:
    state = engine()
    play(state, [0, 1, 0, 1, 0, 1, 0])
    assert state.result() is GameResult.PLAYER_ONE


@pytest.mark.parametrize("engine", ENGINES)
def test_diagonal_win(engine) -> None:
    state = engine()
    play(state, [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert state.result() is GameResult.PLAYER_ONE


def test_invalid_move_and_clone() -> None:
    for engine in ENGINES:
        state = engine()
        with pytest.raises(ValueError):
            state.step(7)
        clone = state.clone()
        clone.step(3)
        assert state.board != clone.board
        assert state.current_player is Player.ONE


@pytest.mark.parametrize("engine", ENGINES)
def test_full_column_and_post_terminal_rejection(engine) -> None:
    state = engine()
    for _ in range(6):
        state.step(0)
    assert 0 not in state.legal_actions()
    with pytest.raises(ValueError):
        state.step(0)


@pytest.mark.parametrize("engine", ENGINES)
def test_draw_and_terminal_player(engine) -> None:
    actions = [1, 2, 2, 4, 6, 2, 0, 5, 6, 5, 1, 4, 3, 1, 1, 1, 4, 5, 0, 4, 0,
               4, 5, 1, 6, 0, 2, 6, 6, 6, 5, 4, 2, 3, 2, 3, 3, 0, 0, 3, 5, 3]
    state = engine()
    for action in actions:
        state.step(action)
    assert state.result() is GameResult.DRAW
    assert state.is_terminal()
    assert state.legal_actions() == []


def test_statistics_helpers() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    lower, upper = wilson_interval(50, 100)
    assert 0.4 < lower < 0.5 < upper < 0.6


def test_alpha_beta_ablation_options() -> None:
    state = ConnectFour()
    assert AlphaBetaAgent(depth=2, pruning=False).choose_action(state) in state.legal_actions()
    assert AlphaBetaAgent(depth=2, move_ordering=False).choose_action(state) in state.legal_actions()


@pytest.mark.parametrize("engine", ENGINES)
def test_all_search_agents_accept_both_state_representations(engine) -> None:
    state = engine()
    for action in [3, 2, 4, 1, 3, 5]:
        state.step(action)
    for agent in (
        RandomAgent(seed=7),
        HeuristicAgent(),
        AlphaBetaAgent(depth=3),
        MCTSAgent(simulations=32, seed=7),
    ):
        assert agent.choose_action(state) in state.legal_actions()


def test_grid_and_bitboard_evaluators_match() -> None:
    grid, bitboard = ConnectFour(), BitboardConnectFour()
    for action in [3, 2, 4, 1, 3, 5, 0, 4, 2, 6, 3, 1]:
        grid.step(action)
        bitboard.step(action)
        assert bitboard.board == grid.board
        assert bitboard.result() == grid.result()
        if not grid.is_terminal():
            for player in (Player.ONE, Player.TWO):
                assert evaluate(bitboard, player) == evaluate(grid, player)


def test_time_limited_agents_return_legal_actions() -> None:
    for engine in ENGINES:
        state = engine()
        assert AlphaBetaAgent(depth=42, time_budget_seconds=0.01).choose_action(state) in state.legal_actions()
        mcts = MCTSAgent(simulations=1_000_000, seed=9, time_budget_seconds=0.01)
        assert mcts.choose_action(state) in state.legal_actions()
        assert 1 <= mcts.simulations_run < mcts.simulations


@pytest.mark.parametrize("engine", ENGINES)
def test_policy_agent_takes_winning_move_and_blocks_immediate_threat(engine) -> None:
    agent = PolicyAgent(lambda state: [0, 0, 0, 0, 0, 0, 100])
    winning_state = engine()
    play(winning_state, [0, 6, 1, 6, 2, 5])
    assert agent.choose_action(winning_state) == 3

    blocking_state = engine()
    play(blocking_state, [6, 0, 6, 1, 5, 2])
    assert agent.choose_action(blocking_state) == 3


def test_training_splits_are_game_grouped_and_position_disjoint() -> None:
    splits = split_training_records(20, seed=42, epsilon=0.05, engine_factory=BitboardConnectFour)
    train_ids = {record.game_id for record in splits["train"]}
    validation_ids = {record.game_id for record in splits["validation"]}
    test_ids = {record.game_id for record in splits["test"]}
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert validation_ids.isdisjoint(test_ids)

    def key(record):
        mirror = tuple(tuple(reversed(row)) for row in record.board)
        return min(record.board, mirror), record.player

    keys = {name: {key(record) for record in items} for name, items in splits.items()}
    assert keys["train"].isdisjoint(keys["validation"] | keys["test"])
    assert keys["validation"].isdisjoint(keys["test"])

