from pathlib import Path

import pytest

from modal_world.backend import BackendUnavailable, WorldBackendError
from modal_world.backends.hyworld2 import HYWorld2Backend
from modal_world.contracts import Operation, WorldRequest


def test_capability_reserves_both_operations():
    backend = HYWorld2Backend()
    assert backend.capability.supports(Operation.RECONSTRUCT)
    assert backend.capability.supports(Operation.GENERATE)


def test_generate_requires_pinned_worldgen_root(tmp_path: Path):
    backend = HYWorld2Backend()
    req = WorldRequest(Operation.GENERATE, tmp_path, tmp_path / "out")
    with pytest.raises(BackendUnavailable, match="worldgen_root"):
        backend.run(req)


def test_reconstruct_reports_missing_upstream_cleanly(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    backend = HYWorld2Backend()
    req = WorldRequest(
        Operation.RECONSTRUCT,
        source,
        tmp_path / "out",
        options={"module": "definitely_missing_hyworld2_module"},
    )
    with pytest.raises((BackendUnavailable, WorldBackendError)):
        backend.run(req)
