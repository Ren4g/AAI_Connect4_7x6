from pathlib import Path

from scripts.build_phase_c_docs import main


def test_build_phase_c_docs_uses_completed_results_and_wdl_totals():
    main()
    root = Path(__file__).resolve().parents[1]
    phase = (root / "docs/PHASE_C_V2_RESULTS.md").read_text(encoding="utf-8")
    rr = (root / "docs/ROUND_ROBIN_PHASE_C_20261123.md").read_text(encoding="utf-8")
    assert "89.53%" in phase
    assert "-3.70 đpt" in phase
    assert "1.280" in phase
    assert "AlphaBeta-depth6 | PUCT" in rr
    assert "0.500" in rr
