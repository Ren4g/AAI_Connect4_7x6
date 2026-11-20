import pytest


def test_reinforce_smoke_writes_a_resumable_artifact(tmp_path):
    pytest.importorskip("torch")
    from connect4.rl_train import train

    dataset = "colab/data/solverlabels_20261120_seed20261120_v2"
    if not __import__("pathlib").Path(dataset).exists():
        pytest.skip("prepared solver-labelled data package is not present")
    args = type("Args", (), {
        "seed": 123, "device": "cpu", "dataset": dataset,
        "run_id": "rl_test_smoke", "output_root": str(tmp_path),
        "resume": None, "init_checkpoint": None, "learning_rate": 1e-3,
        "max_minutes": 2.0,
        "config": {"episodes": 2, "learning_rate": 1e-3,
                   "checkpoint_interval_seconds": 120, "value_weight": 0.5,
                   "entropy_weight": 0.01, "smoke": True,
                   "init_checkpoint_sha256": None},
    })()
    manifest = train(args)
    assert manifest["status"] == "complete"
    assert manifest["completed_steps"] == 2
    assert manifest["test_overlap_count"] == 0
    assert (tmp_path / "rl_test_smoke" / "checkpoint_latest.pt").is_file()
    from scripts.ingest_colab_run import ingest
    with pytest.raises(ValueError, match="smoke artifacts"):
        ingest(tmp_path / "rl_test_smoke", __import__("pathlib").Path(dataset),
               tmp_path / "ingested", tmp_path / "TEST_LOG.md")
