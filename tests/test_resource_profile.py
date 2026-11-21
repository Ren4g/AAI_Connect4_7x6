import json

from scripts.measure_agent_resources import colab_memory, render_report


def test_colab_memory_preserves_unrecorded_vram_as_missing(tmp_path):
    manifest = {
        "run_id": "az_test",
        "phase": "alphazero_selfplay",
        "gpu_name": "Tesla T4",
        "ram_bytes": 1024 * 1024,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    [row] = colab_memory([path])

    assert row["ram_mib"] == 1
    assert row["peak_vram_bytes"] is None
    assert row["peak_vram_mib"] is None


def test_resource_report_labels_missing_vram_and_local_sampling():
    report = render_report({
        "timestamp_local": "2026-11-21",
        "source_commit": "abc123",
        "script_sha256": "deadbeef",
        "local_agents": [{
            "agent": "Random", "baseline_rss_bytes": 1024 * 1024,
            "peak_rss_mib": 2, "peak_increment_mib": 1,
            "measured_agent_decisions": 4, "game_seconds": 0.2,
        }],
        "colab_runs": [{
            "run_id": "az_test", "phase": "alphazero_selfplay", "gpu": "T4",
            "ram_mib": 1024, "peak_vram_mib": None,
        }],
    })

    assert "100 ms" in report
    assert "không đo/không ghi" in report
    assert "Working Set" in report
