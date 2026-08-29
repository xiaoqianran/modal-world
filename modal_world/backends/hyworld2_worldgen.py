from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..backend import BackendUnavailable, WorldBackendError


@dataclass(frozen=True)
class StageCommand:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]


class OfficialWorldgenProfile:
    """Command builder for Tencent's upstream HY-World 2.0 worldgen pipeline.

    This profile mirrors the official five-stage pipeline. Process topology is
    configurable because upstream recommends >=4 GPUs and documents 8 GPUs.
    Single-GPU community patches belong in a separate profile.
    """

    name = "official"

    def __init__(
        self,
        *,
        worldgen_root: Path,
        target_path: Path,
        result_dir: Path,
        llm_addr: str,
        llm_port: int,
        llm_name: str,
        nproc: int = 8,
        python: str = sys.executable,
        cuda_visible_devices: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.worldgen_root = worldgen_root.resolve()
        self.target_path = target_path.resolve()
        self.result_dir = result_dir.resolve()
        self.llm_addr = llm_addr
        self.llm_port = int(llm_port)
        self.llm_name = llm_name
        self.nproc = int(nproc)
        self.python = python
        self.cuda_visible_devices = cuda_visible_devices
        if self.nproc < 1:
            raise ValueError("nproc must be >= 1")
        self.max_steps = max_steps or self._scaled_steps(self.nproc)

    @staticmethod
    def _scaled_steps(nproc: int) -> int:
        # Official README: x8=1500, x4=2000, x2=4000, x1=8000.
        return {1: 8000, 2: 4000, 4: 2000, 8: 1500}.get(nproc, 8000)

    @staticmethod
    def _scaled_strategy_steps(max_steps: int) -> tuple[int, int, int, int]:
        """Scale upstream 1500-step strategy timings proportionally to max_steps."""
        scale = max_steps / 1500.0
        refine_start = max(1, round(150 * scale))
        refine_stop = max(refine_start + 1, round(750 * scale))
        refine_every = max(1, round(100 * scale))
        scale2d_stop = max(refine_start + 1, round(750 * scale))
        return refine_start, refine_stop, refine_every, scale2d_stop

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return env

    def _torchrun(self, script: str, *args: str) -> tuple[str, ...]:
        return ("torchrun", "--nproc_per_node", str(self.nproc), script, *args)

    def commands(self) -> tuple[StageCommand, ...]:
        common_llm = (
            "--llm_addr",
            self.llm_addr,
            "--llm_port",
            str(self.llm_port),
            "--llm_name",
            self.llm_name,
        )
        target = str(self.target_path)
        env = self._env()
        steps = str(self.max_steps)
        refine_start, refine_stop, refine_every, scale2d_stop = self._scaled_strategy_steps(
            self.max_steps
        )
        return (
            StageCommand(
                "trajectory_planning",
                (
                    self.python,
                    "traj_generate.py",
                    "--target_path",
                    target,
                    *common_llm,
                    "--apply_nav_traj",
                    "--apply_up_route",
                    "--apply_recon_iteration",
                    "--force_vlm",
                ),
                self.worldgen_root,
                env,
            ),
            StageCommand(
                "trajectory_rendering",
                self._torchrun("traj_render.py", "--target_path", target, *common_llm),
                self.worldgen_root,
                env,
            ),
            StageCommand(
                "world_expansion",
                self._torchrun("video_gen.py", "--target_path", target, "--fsdp"),
                self.worldgen_root,
                env,
            ),
            StageCommand(
                "gs_data_preparation",
                self._torchrun(
                    "gen_gs_data.py",
                    "--root_path",
                    target,
                    "--save_normal",
                    "--split_sky",
                ),
                self.worldgen_root,
                env,
            ),
            StageCommand(
                "gs_training",
                (
                    self.python,
                    "-m",
                    "world_gs_trainer",
                    "default",
                    "--data_dir",
                    str(self.target_path / "gs_data"),
                    "--result_dir",
                    str(self.result_dir),
                    "--max_steps",
                    steps,
                    "--save_steps",
                    steps,
                    "--eval_steps",
                    steps,
                    "--ply_steps",
                    steps,
                    "--save_ply",
                    "--convert_to_spz",
                    "--disable_video",
                    "--use_scale_regularization",
                    "--antialiased",
                    "--depth_loss",
                    "--normal_loss",
                    "--sky_depth_from_pcd",
                    "--use_mask_gaussian",
                    "--mask_export_stochastic",
                    "--no-mask-export-anchor-protection",
                    "--use_anchor_protection",
                    "--export_mesh",
                    "--strategy.refine-start-iter",
                    str(refine_start),
                    "--strategy.refine-stop-iter",
                    str(refine_stop),
                    "--strategy.refine-every",
                    str(refine_every),
                    "--strategy.refine-scale2d-stop-iter",
                    str(scale2d_stop),
                    "--strategy.reset-every",
                    "99990",
                    "--strategy.grow-grad2d",
                    "0.0001",
                    "--strategy.prune-scale3d",
                    "0.1",
                ),
                self.worldgen_root,
                env,
            ),
        )

    def validate(self) -> None:
        if not self.worldgen_root.is_dir():
            raise BackendUnavailable(f"HYWorld2 worldgen root not found: {self.worldgen_root}")
        required = ("traj_generate.py", "traj_render.py", "video_gen.py", "gen_gs_data.py")
        missing = [name for name in required if not (self.worldgen_root / name).is_file()]
        if missing:
            raise BackendUnavailable(
                "HYWorld2 worldgen integration is incomplete; missing: " + ", ".join(missing)
            )

    def run(
        self,
        *,
        start_stage: str | None = None,
        stop_stage: str | None = None,
        timeout_s: float = 6 * 60 * 60,
    ) -> list[dict[str, object]]:
        self.validate()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        commands = list(self.commands())
        names = [stage.name for stage in commands]
        start = names.index(start_stage) if start_stage else 0
        stop = names.index(stop_stage) + 1 if stop_stage else len(commands)
        if stop < start:
            raise ValueError("stop_stage must not precede start_stage")

        results: list[dict[str, object]] = []
        for stage in commands[start:stop]:
            try:
                completed = subprocess.run(
                    stage.argv,
                    cwd=stage.cwd,
                    env=dict(stage.env),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout_s,
                )
            except FileNotFoundError as exc:
                raise BackendUnavailable(
                    f"HYWorld2 stage {stage.name} executable unavailable: {stage.argv[0]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise WorldBackendError(f"HYWorld2 stage {stage.name} timed out") from exc

            record = {
                "stage": stage.name,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            results.append(record)
            if completed.returncode != 0:
                raise WorldBackendError(
                    f"HYWorld2 stage {stage.name} failed with exit {completed.returncode}: "
                    f"{completed.stderr[-4000:].strip()}"
                )
        return results
