from pathlib import Path

import pytest

from modal_world.worldgen_job import (
    build_stage_manifest,
    fingerprint_files,
    manifest_matches,
    resolve_worldgen_job_root,
    write_stage_manifest,
)


def test_job_paths_preserve_case000_and_isolate_new_jobs():
    assert resolve_worldgen_job_root("case000") == Path("/worldgen/case000")
    assert resolve_worldgen_job_root("job-abc_123") == Path("/worldgen/jobs/job-abc_123")
    for bad in ("", "../escape", "/absolute", "a/b", " space"):
        with pytest.raises(ValueError):
            resolve_worldgen_job_root(bad)


def test_stage_manifest_invalidates_changed_inputs(tmp_path: Path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"first")
    fingerprint = fingerprint_files([source], root=tmp_path)
    manifest = build_stage_manifest(
        job_id="job1",
        stage="stage2",
        hyworld_revision="rev",
        input_fingerprint=fingerprint,
        config={"quality": 1},
    )
    write_stage_manifest(tmp_path, "stage2", manifest)
    assert manifest_matches(tmp_path, "stage2", manifest)

    source.write_bytes(b"second")
    changed = build_stage_manifest(
        job_id="job1",
        stage="stage2",
        hyworld_revision="rev",
        input_fingerprint=fingerprint_files([source], root=tmp_path),
        config={"quality": 1},
    )
    assert not manifest_matches(tmp_path, "stage2", changed)


def test_stage2_and_stage4_accept_job_id_and_use_manifest():
    app = Path("modal_world/app.py").read_text()
    stage2 = Path("modal_world/stage2_app.py").read_text()
    assert 'def worldgen_case000_stage2(job_id: str = "case000")' in app
    assert 'def worldgen_case000_stage4(job_id: str = "case000")' in app
    assert 'manifest_matches(target, "stage2"' in stage2
    assert 'write_stage_manifest(target, "stage2"' in stage2
    assert 'manifest_matches(target, "stage4"' in app
    assert 'write_stage_manifest(target, "stage4"' in app
