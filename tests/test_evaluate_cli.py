import pytest

from evaluate import make_agents
from connect4.agents import PolicyAgent


def test_round_robin_agent_factory_can_include_two_fixed_alpha_beta_depths() -> None:
    agents = make_agents(alpha_beta_depth=4, extra_alpha_beta_depth=6)

    assert "AlphaBeta-depth4" in agents
    assert "AlphaBeta-depth6" in agents
    assert agents["AlphaBeta-depth4"].depth == 4
    assert agents["AlphaBeta-depth6"].depth == 6


def test_round_robin_agent_factory_rejects_duplicate_extra_depth() -> None:
    with pytest.raises(ValueError, match="differ from the main depth"):
        make_agents(alpha_beta_depth=4, extra_alpha_beta_depth=4)


def test_round_robin_agent_factory_labels_rl_policy(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(PolicyAgent, "from_checkpoint", lambda path, device: sentinel)

    agents = make_agents(rl_checkpoint="rl.pt", alpha_beta_depth=4)

    assert agents["RL-Policy"] is sentinel


def test_solver_evaluator_creates_raw_and_filtered_puct_variants(tmp_path, monkeypatch):
    from evaluate_solver import make_agents as make_solver_agents
    from connect4.agents import PUCTAgent

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"stub checkpoint")
    calls = []
    monkeypatch.setattr(
        PUCTAgent, "from_checkpoint",
        lambda path, simulations, device, safety_filter=False: (
            calls.append(safety_filter) or ("filtered" if safety_filter else "raw")
        ),
    )

    agents, hashes = make_solver_agents([], [f"level8={checkpoint}"], "cpu", 4, 100)

    assert agents["PUCT-level8-raw"] == "raw"
    assert agents["PUCT-level8-filtered"] == "filtered"
    assert calls == [False, True]
    assert "PUCT-level8" in hashes
