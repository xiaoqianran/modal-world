from pathlib import Path


def test_stage3_is_single_gpu_without_fsdp():
    source = Path("modal_world/app.py").read_text()
    start = source.index("def worldgen_case000_stage3()")
    section = source[start:]
    assert '"--nproc_per_node=1"' in section
    assert '"--fsdp"' not in section
    assert '"--local_files_only"' in section
    assert '"--skip_exist"' in section


def test_stage3_requests_high_cpu_memory():
    source = Path("modal_world/app.py").read_text()
    marker = "def worldgen_case000_stage3()"
    start = source.index(marker)
    decorator = source[max(0, start - 500) : start]
    assert "memory=131072" in decorator
    assert "cpu=16.0" in decorator
