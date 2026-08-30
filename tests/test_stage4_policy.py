from pathlib import Path


def test_stage4_uses_official_quality_flags_and_true_single_process():
    source = Path("modal_world/app.py").read_text()
    start = source.index('def worldgen_case000_stage4(job_id: str = "case000")')
    section = source[start:]
    assert '"torch.distributed.run"' not in section
    assert '"--nproc_per_node=1"' not in section
    assert '"gen_gs_data.py"' in section
    assert '"--save_normal"' in section
    assert '"--split_sky"' in section
    assert '"--split_align"' not in section
    assert 'env["WORLD_SIZE"] = "1"' in section


def test_stage4_runtime_patch_is_image_build_time_and_offline():
    runtime = Path("modal_world/hyworld2_runtime.py").read_text()
    patch = Path("modal_world/stage4_patch.py").read_text()
    app = Path("modal_world/app.py").read_text()
    assert ".run_function(patch_stage4_single_gpu" in runtime
    assert "local_files_only=True" in patch
    assert "if world_size == 1:" in patch
    assert "if world_size > 1:" in patch
    assert "dist.is_initialized()" in patch
    start = app.index('def worldgen_case000_stage4(job_id: str = "case000")')
    section = app[start:]
    assert "MoGeModel.from_pretrained" not in section


def test_stage4_gpu_subprocess_sampler_is_opt_in():
    source = Path("modal_world/app.py").read_text()
    start = source.index('def worldgen_case000_stage4(job_id: str = "case000")')
    section = source[start:]
    assert 'os.environ.get("MODAL_WORLD_DEBUG_GPU_SAMPLER") == "1"' in section
    assert '"gpu_sampler_enabled": monitor is not None' in section
