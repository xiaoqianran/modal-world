from pathlib import Path


def test_stage4_uses_official_quality_flags_and_single_gpu():
    source = Path("modal_world/app.py").read_text()
    start = source.index("def worldgen_case000_stage4()")
    section = source[start:]
    assert '"--nproc_per_node=1"' in section
    assert '"--save_normal"' in section
    assert '"--split_sky"' in section
    assert '"--split_align"' not in section
    assert "local_files_only=True" in section
