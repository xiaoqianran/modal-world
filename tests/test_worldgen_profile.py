from pathlib import Path

from modal_world.backends.hyworld2_worldgen import OfficialWorldgenProfile


def make_profile(tmp_path: Path, nproc: int = 8) -> OfficialWorldgenProfile:
    return OfficialWorldgenProfile(
        worldgen_root=tmp_path,
        target_path=tmp_path / "scene",
        result_dir=tmp_path / "result",
        llm_addr="127.0.0.1",
        llm_port=8000,
        llm_name="Qwen/Qwen3-VL-8B-Instruct",
        nproc=nproc,
    )


def test_official_profile_has_five_stages(tmp_path: Path):
    profile = make_profile(tmp_path)
    assert [stage.name for stage in profile.commands()] == [
        "trajectory_planning",
        "trajectory_rendering",
        "world_expansion",
        "gs_data_preparation",
        "gs_training",
    ]


def test_official_single_gpu_uses_documented_3dgs_steps(tmp_path: Path):
    assert make_profile(tmp_path, nproc=1).max_steps == 8000


def test_official_8gpu_uses_documented_3dgs_steps(tmp_path: Path):
    assert make_profile(tmp_path, nproc=8).max_steps == 1500


def test_single_gpu_scales_strategy_steps_with_max_steps(tmp_path: Path):
    profile = make_profile(tmp_path, nproc=1)
    train = next(stage for stage in profile.commands() if stage.name == "gs_training")
    argv = list(train.argv)

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value("--max_steps") == "8000"
    assert value("--strategy.refine-start-iter") == "800"
    assert value("--strategy.refine-stop-iter") == "4000"
    assert value("--strategy.refine-every") == "533"
    assert value("--strategy.refine-scale2d-stop-iter") == "4000"
