from pathlib import Path

from modal_world.contracts import Operation, WorldRequest


def test_request_normalizes_paths(tmp_path: Path):
    req = WorldRequest(Operation.RECONSTRUCT, tmp_path, tmp_path / "out")
    normalized = req.normalized()
    assert normalized.input_path.is_absolute()
    assert normalized.output_dir.is_absolute()
