import pytest

from connect4.engine import ConnectFour
from connect4.engine import GameResult, Player


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


def test_batched_puct_terminal_value_uses_winner_perspective():
    torch = pytest.importorskip("torch")
    from connect4.selfplay import BatchedPUCT

    class ConstantNet(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros((len(inputs), 7)), torch.zeros(len(inputs))

    won = ConnectFour()
    for action in (0, 6, 1, 6, 2, 5, 3):
        won.step(action)
    draw = ConnectFour()
    draw._result = GameResult.DRAW

    assert won.current_player is Player.ONE  # engine retains the final mover
    assert BatchedPUCT._terminal_value(won) == 1.0
    assert BatchedPUCT._terminal_value(draw) == 0.0


def test_batched_puct_safety_filter_overrides_an_immediate_losing_move():
    torch = pytest.importorskip("torch")
    from connect4.selfplay import BatchedPUCT

    class UnsafePriorNet(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.as_tensor([10, 9, 8, 0, 7, 6, 5], dtype=torch.float32)
            return logits.repeat(len(inputs), 1), torch.zeros(len(inputs))

    state = ConnectFour()
    for action in (2, 1, 3, 5, 0, 0, 6, 4, 0, 2, 4, 0, 4, 1):
        state.step(action)
    net = UnsafePriorNet()
    raw = BatchedPUCT(net, torch.device("cpu"), 1, 1.5, 0.3, 0.0)
    filtered = BatchedPUCT(net, torch.device("cpu"), 1, 1.5, 0.3, 0.0,
                           safety_filter=True)

    assert raw.search([state], 0.0, __import__("numpy").random.default_rng(1))[1] == [0]
    assert filtered.search([state], 0.0, __import__("numpy").random.default_rng(1))[1] == [3]


def test_selfplay_smoke_saves_initial_and_iteration_snapshots(tmp_path):
    torch = pytest.importorskip("torch")
    from connect4.selfplay import train

    args = type("Args", (), {
        "dataset": "colab/data/solverlabels_20261120_seed20261120_v2",
        "output_root": str(tmp_path), "run_id": "az_snapshot_smoke",
        "seed": 7, "device": "cpu", "init_checkpoint": None,
            "resume": None, "max_minutes": 2.0, "learning_rate": 1e-3,
        "config": {
            "iterations": 1, "games_per_iteration": 2, "parallel_games": 2,
            "simulations": 2, "training_steps": 2, "batch_size": 4,
            "replay_capacity": 128, "learning_rate": 1e-3, "value_weight": 0.5,
            "c_puct": 1.5, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
            "temperature_moves": 12, "snapshot_interval": 1,
            "checkpoint_interval_seconds": 120, "smoke": True,
            "init_checkpoint_sha256": None,
        },
    })()

    manifest = train(args)

    assert manifest["status"] == "complete"
    assert manifest["torch_intraop_threads"] == 1
    snapshots = tmp_path / "az_snapshot_smoke" / "checkpoints"
    assert (snapshots / "checkpoint_iteration_0000.pt").is_file()
    assert (snapshots / "checkpoint_iteration_0001.pt").is_file()
    assert "checkpoints/checkpoint_iteration_0001.pt" in manifest["files_sha256"]
