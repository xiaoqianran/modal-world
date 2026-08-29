from pathlib import Path


def test_stage5_preflight_uses_official_dataset_and_lpips_cache():
    source = Path("modal_world/app.py").read_text()
    start = source.index("def preflight_worldgen_case000_stage5()")
    section = source[start:]
    assert "downsample_pts_num=1_000_000" in section
    assert 'downsample_mode="geometry_aware"' in section
    assert 'LearnedPerceptualImagePatchSimilarity(net_type="vgg"' in section
    assert "TORCH_HOME" in section
