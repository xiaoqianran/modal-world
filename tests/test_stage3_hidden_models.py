from pathlib import Path


def test_stage3_preloads_dinov2_and_forces_offline_camera_selector():
    source = Path("modal_world/app.py").read_text()
    assert '("facebook/dinov2-base", None)' in source
    assert 'snapshot_download("facebook/dinov2-base", local_files_only=True)' in source
    assert "model_path, use_fast=True, local_files_only=True" in source
    assert "model_path, local_files_only=True" in source
