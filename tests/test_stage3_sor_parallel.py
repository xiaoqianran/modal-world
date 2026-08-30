from pathlib import Path


def test_stage3_sor_uses_all_allocated_cpu_workers():
    patch = Path("modal_world/stage3_patch.py").read_text()
    assert "tree.query(points, k=nb_neighbors + 1, workers=-1)" in patch
    assert "expected pinned single-threaded SOR query not found" in patch
