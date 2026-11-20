import pytest

from connect4.engine import ConnectFour


def test_batched_puct_returns_visit_targets_for_each_active_game():
    torch = pytest.importorskip("torch")
    from connect4.selfplay import BatchedPUCT

    class ConstantNet(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros((len(inputs), 7), device=inputs.device), torch.zeros(
                len(inputs), device=inputs.device
            )

    roots = [ConnectFour(), ConnectFour()]
    search = BatchedPUCT(ConstantNet(), torch.device("cpu"), simulations=4,
                         c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    policies, actions = search.search(roots, [1.0, 0.0], __import__("numpy").random.default_rng(9))
    assert len(policies) == len(actions) == 2
    for state, policy, action in zip(roots, policies, actions):
        assert action in state.legal_actions()
        assert abs(sum(policy) - 1.0) < 1e-6
        assert all(value >= 0 for value in policy)
