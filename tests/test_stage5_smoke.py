from pathlib import Path


def test_stage5_smoke_is_short_and_real():
    source = Path("modal_world/app.py").read_text()
    start = source.index('def worldgen_case000_stage5_smoke(job_id: str = "case000")')
    section = source[start:]
    assert "steps = 100" in section
    assert '"--disable_viewer"' in section
    assert '"--save_ply"' in section
    assert '"--convert_to_spz"' in section
    assert '"--depth_loss"' in section
    assert '"--normal_loss"' in section
    assert '"--export_mesh"' not in section


def test_stage5_smoke_uses_job_isolated_worldgen_root():
    source = Path("modal_world/app.py").read_text()
    start = source.index('def worldgen_case000_stage5_smoke(job_id: str = "case000")')
    section = source[start:]
    assert "target = resolve_worldgen_job_root(job_id)" in section
    assert 'Path("/worldgen/case000")' not in section
    assert 'result_dir = target / "gs_smoke_result"' in section
