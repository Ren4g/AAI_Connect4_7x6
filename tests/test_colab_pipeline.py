import json
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


def test_colab_notebook_is_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path

    notebooks = sorted(Path("colab").glob("*.ipynb"))
    assert {path.name for path in notebooks} == {
        "02_train_cnn.ipynb", "03_selfplay_az.ipynb", "04_rl_policy_gradient.ipynb"
    }
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert len(notebook.cells) == 8
        assert all(cell.id for cell in notebook.cells)


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
