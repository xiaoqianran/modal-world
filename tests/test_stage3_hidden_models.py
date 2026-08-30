from pathlib import Path


def test_stage3_preloads_dinov2_and_forces_offline_camera_selector():
    app_source = Path("modal_world/app.py").read_text()
    patch_source = Path("modal_world/stage3_patch.py").read_text()
    runtime_source = Path("modal_world/hyworld2_runtime.py").read_text()
    assert '("facebook/dinov2-base", None)' in app_source
    assert 'snapshot_download("facebook/dinov2-base", local_files_only=True)' in app_source
    assert "model_path, use_fast=True, local_files_only=True" in patch_source
    assert "model_path, local_files_only=True" in patch_source
    assert ".run_function(patch_stage3_runtime" in runtime_source


def test_stage3_preloads_and_verifies_worldmirror_offline():
    app = Path("modal_world/app.py").read_text()
    preload = app[
        app.index("def preload_worldstereo_stage3_weights") : app.index(
            "@app.function", app.index("def preload_worldstereo_stage3_weights") + 10
        )
    ]
    verify = app[
        app.index("def verify_worldstereo_stage3_cache") : app.index(
            "@app.function", app.index("def verify_worldstereo_stage3_cache") + 10
        )
    ]
    assert '"tencent/HY-World-2.0", ["HY-WorldMirror-2.0/**"]' in preload
    assert 'worldmirror_repo = "tencent/HY-World-2.0"' in verify
    assert 'worldmirror_subfolder = "HY-WorldMirror-2.0"' in verify
    assert '"worldmirror_snapshot"' in verify
    assert 'worldmirror_dir / "model.safetensors"' in verify
    assert 'worldmirror_dir / "config.yaml"' in verify
    assert 'worldmirror_dir / "config.json"' in verify
    assert '"worldmirror": cache / "models--tencent--HY-World-2.0" / "blobs"' in verify
