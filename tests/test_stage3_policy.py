from pathlib import Path


def test_stage3_is_single_gpu_without_fsdp():
    app = Path("modal_world/app.py").read_text()
    worker = Path("modal_world/stage3_app.py").read_text()
    start = app.index('def worldgen_case000_stage3(job_id: str = "case000")')
    proxy = app[start:]
    assert 'modal.Cls.from_name("modal-world-stage3", "WorldStereoWorker")' in proxy
    assert "_spawn_worker_call(worker_cls().generate" in proxy
    assert '"torch.distributed.run"' not in proxy
    assert "fsdp=False" in worker
    assert "sp_world_size=1" in worker
    assert "local_files_only=True" in worker


def test_stage3_requests_high_cpu_memory_and_scales_to_zero():
    source = Path("modal_world/stage3_app.py").read_text()
    start = source.index("@app.cls(")
    end = source.index("class WorldStereoWorker", start)
    decorator = source[start:end]
    assert "memory=131072" in decorator
    assert "cpu=16.0" in decorator
    assert "min_containers=0" in decorator
    assert "scaledown_window=5 * 60" in decorator
