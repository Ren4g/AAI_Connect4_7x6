import json
import ast
from pathlib import Path

import pytest

from connect4.train_cnn import verify_dataset


DATASET = Path("colab/data/solverlabels_20261120_seed20261120_v2")


def test_clean_colab_dataset_is_frozen_test_disjoint():
    if not DATASET.exists():
        pytest.skip("prepared local solver-label package is not present")
    manifest, arrays, records = verify_dataset(DATASET)
    assert manifest["frozen_state_overlap_count"] == 0
    assert len(records) == len(arrays["split"]) == 1500
    assert "legal_mask" in arrays.files


def test_v3_dataset_reconstructs_every_state_and_has_balanced_stages():
    dataset = Path("colab/data/solverlabels_20261121_seed20261121_v3_30k")
    if not dataset.exists():
        pytest.skip("V2 30k solver-labelled package is not present")
    manifest, arrays, records = verify_dataset(dataset)
    assert len(records) == len(arrays["split"]) == 30000
    assert manifest["frozen_state_overlap_count"] == 0
    assert {stage: sum(row["stage"] == stage for row in records if row["split"] == "train")
            + sum(row["stage"] == stage for row in records if row["split"] == "validation")
            for stage in ("opening", "middle", "endgame")} == {
                "opening": 10000, "middle": 10000, "endgame": 10000,
            }


def test_colab_notebook_is_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path

    notebooks = sorted(Path("colab").glob("*.ipynb"))
    assert {path.name for path in notebooks} == {
        "02_train_cnn.ipynb", "03_selfplay_az.ipynb", "04_rl_policy_gradient.ipynb",
        "05_train_cnn_v3_30k.ipynb", "06_selfplay_az_v2.(outdated).ipynb",
    }
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert len(notebook.cells) == 8
        assert all(cell.id for cell in notebook.cells)
        code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        parameter_cell = next(source for source in code_cells if "SOURCE_ZIP" in source)
        expected_bundle = "source_bundle_20261121_v12.zip" if path.name == "03_selfplay_az.ipynb" else (
            "source_bundle_20261121_v11.zip" if path.name in {
                "05_train_cnn_v3_30k.ipynb", "06_selfplay_az_v2.(outdated).ipynb"
            } else "source_bundle_20261121_v6.zip"
        )
        assert expected_bundle in parameter_cell
        assert "SOURCE_MANIFEST =" in parameter_cell
        assert "zip'SOURCE_MANIFEST" not in parameter_cell
        assert "NoneSOURCE_MANIFEST" not in parameter_cell
        if path.name == "03_selfplay_az.ipynb":
            all_source = "\n".join(code_cells)
            assert "SIMULATIONS = 400" in parameter_cell
            assert "MAX_ITERS = 40" in parameter_cell
            assert "SNAPSHOT_INTERVAL = 5" in parameter_cell
            assert "az_v2_warm_seed20261121" in parameter_cell
            assert "az_v2_scratch_seed20261122" in parameter_cell
            assert "PHASE_C_GPU_MINUTES = 120" in parameter_cell
            assert "MAX_SEGMENT_MINUTES = 30" in parameter_cell
            assert "PER_RUN_GPU_MINUTES = 60" in parameter_cell
            assert "az_v2_phase_c_budget.json" in all_source
            assert "run_dir.rglob('*')" in all_source
        syntax_errors = []
        for source in code_cells:
            python_lines = [line for line in source.splitlines()
                            if not line.lstrip().startswith(("%", "!"))]
            if python_lines:
                try:
                    ast.parse("\n".join(python_lines), filename=f"{path.name} code cell")
                except SyntaxError as error:
                    syntax_errors.append((error.lineno, error.msg, python_lines))
        assert not syntax_errors, f"invalid Python in {path.name}: {syntax_errors}"


def test_ingest_rejects_manifest_with_missing_required_metadata(tmp_path):
    from scripts.ingest_colab_run import ingest

    artifact = tmp_path / "bad"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"run_id": "bad"}))
    with pytest.raises(ValueError, match="missing fields"):
        ingest(artifact, DATASET, tmp_path / "out")


def test_source_bundle_roundtrip_hash_matches_runtime_tree(tmp_path):
    import zipfile

    from scripts.ingest_colab_run import source_tree_sha256
    from scripts.package_colab_source import package

    bundle = tmp_path / "source_bundle.zip"
    record = package(Path.cwd(), bundle)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extracted)
    assert source_tree_sha256(extracted) == record["source_tree_sha256"]
