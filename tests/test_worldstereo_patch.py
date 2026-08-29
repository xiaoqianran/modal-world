from pathlib import Path

from modal_world.worldstereo_patch import patch_worldstereo_wrapper_source


def test_patch_matches_pinned_upstream_wrapper():
    upstream = Path("/tmp/hyworld2-src/hyworld2/worldgen/models/worldstereo_wrapper.py")
    if not upstream.exists():
        return
    patched = patch_worldstereo_wrapper_source(upstream.read_text())
    assert "AutoTokenizer" not in patched
    assert "T5TokenizerFast.from_pretrained" in patched
    assert "local_files_only=local_files_only" in patched
    assert patched.count("local_files_only=local_files_only") >= 8


def test_patch_rejects_unexpected_source():
    try:
        patch_worldstereo_wrapper_source("not the pinned wrapper")
    except RuntimeError as exc:
        assert "expected exactly one" in str(exc)
    else:
        raise AssertionError("unexpected source must fail closed")
