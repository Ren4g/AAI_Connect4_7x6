from connect4.agents import PUCTAgent
from connect4.engine import ConnectFour


def test_puct_returns_legal_action_and_runs_fixed_budget():
    state = ConnectFour()
    for action in (0, 6, 1, 6, 2, 5):
        state.step(action)
    agent = PUCTAgent(lambda _state: ([0.0] * 7, 0.0), simulations=40, seed=7)
    action = agent.choose_action(state)
    assert action in state.legal_actions()
    assert agent.simulations_run == 40


def test_puct_value_search_finds_immediate_win():
    state = ConnectFour()
    for action in (0, 6, 1, 6, 2, 5):
        state.step(action)
    agent = PUCTAgent(lambda _state: ([0.0] * 7, 0.0), simulations=100, seed=3)
    assert agent.choose_action(state) == 3


def test_puct_rejects_terminal_position():
    state = ConnectFour()
    for action in (0, 0, 1, 1, 2, 2, 3):
        state.step(action)
    agent = PUCTAgent(lambda _state: ([0.0] * 7, 0.0), simulations=5)
    try:
        agent.choose_action(state)
    except ValueError as error:
        assert "No legal" in str(error)
    else:
        raise AssertionError("terminal position should not return a move")


def test_puct_safety_filter_avoids_an_immediate_losing_reply():
    state = ConnectFour()
    for action in (2, 1, 3, 5, 0, 0, 6, 4, 0, 2, 4, 0, 4, 1):
        state.step(action)
    logits = [10.0, 9.0, 8.0, 0.0, 7.0, 6.0, 5.0]
    raw = PUCTAgent(lambda _state: (logits, 0.0), simulations=1, seed=1)
    filtered = PUCTAgent(lambda _state: (logits, 0.0), simulations=1,
                         seed=1, safety_filter=True)

    assert raw.choose_action(state) in {0, 1, 2, 4, 5, 6}
    assert filtered.choose_action(state) == 3
