from pathlib import Path


def test_stage5_smoke_is_short_and_real():
    source = Path("modal_world/app.py").read_text()
    start = source.index("def worldgen_case000_stage5_smoke()")
    section = source[start:]
    assert "steps = 100" in section
    assert '"--disable_viewer"' in section
    assert '"--save_ply"' in section
    assert '"--convert_to_spz"' in section
    assert '"--depth_loss"' in section
    assert '"--normal_loss"' in section
    assert '"--export_mesh"' not in section
