from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..backend import BackendUnavailable, UnsupportedOperation, WorldBackend, WorldBackendError
from ..contracts import Artifact, Capability, Operation, WorldRequest, WorldResult
from .hyworld2_worldgen import OfficialWorldgenProfile


class HYWorld2Backend(WorldBackend):
    """Pure-Python/process adapter for Tencent HY-World 2.0.

    WorldMirror reconstruction is wired through the upstream module entrypoint instead
    of ComfyUI. Full world generation intentionally remains a separate operation so it
    can be connected once the WorldNav/MemoryBank/WorldStereo/3DGS chain is validated.
    """

    name = "hyworld2"
    _capability = Capability(
        backend=name,
        operations=frozenset({Operation.RECONSTRUCT, Operation.GENERATE}),
        inputs=frozenset({"image", "images", "panorama", "video"}),
        outputs=frozenset(
            {
                "gaussian_splat",
                "point_cloud",
                "depth",
                "normal",
                "camera",
                "world",
            }
        ),
        notes=(
            "reconstruct is wired to hyworld2.worldrecon.pipeline",
            "generate uses the official five-stage worldgen profile when configured",
            "official worldgen recommends >=4 GPUs; single-GPU patched profile is separate",
            "ComfyUI is not part of the runtime contract",
        ),
    )

    @property
    def capability(self) -> Capability:
        return self._capability

    def run(self, request: WorldRequest) -> WorldResult:
        req = request.normalized()
        if req.operation is Operation.RECONSTRUCT:
            return self.reconstruct(req)
        if req.operation is Operation.GENERATE:
            return self.generate(req)
        raise UnsupportedOperation(f"unsupported HYWorld2 operation: {req.operation}")

    def reconstruct(self, request: WorldRequest) -> WorldResult:
        if not request.input_path.exists():
            raise FileNotFoundError(request.input_path)
        request.output_dir.mkdir(parents=True, exist_ok=True)

        module = str(request.options.get("module", "hyworld2.worldrecon.pipeline"))
        python = str(request.options.get("python", sys.executable))
        command = [python, "-m", module, "--input_path", str(request.input_path)]

        # Upstream output flags have changed across revisions. Only add an explicit
        # output flag when the pinned integration declares one.
        output_flag = request.options.get("output_flag")
        if output_flag:
            command.extend([str(output_flag), str(request.output_dir)])

        extra_args = request.options.get("extra_args", ())
        if not isinstance(extra_args, (list, tuple)) or not all(
            isinstance(item, str) for item in extra_args
        ):
            raise ValueError("extra_args must be a list/tuple of strings")
        command.extend(extra_args)

        env = os.environ.copy()
        extra_env = request.options.get("env", {})
        if not isinstance(extra_env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra_env.items()
        ):
            raise ValueError("env must be a string-to-string mapping")
        env.update(extra_env)

        try:
            completed = subprocess.run(
                command,
                cwd=str(request.output_dir),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(request.options.get("timeout_s", 60 * 60)),
            )
        except FileNotFoundError as exc:
            raise BackendUnavailable(f"HYWorld2 python executable unavailable: {python}") from exc
        except subprocess.TimeoutExpired as exc:
            raise WorldBackendError(f"HYWorld2 reconstruction timed out: {exc}") from exc

        if completed.returncode != 0:
            stderr = completed.stderr[-4000:].strip()
            if "No module named" in stderr and "hyworld2" in stderr:
                raise BackendUnavailable(
                    "HYWorld2 is not installed in this runtime image; build/pin upstream first"
                )
            raise WorldBackendError(
                f"HYWorld2 reconstruction failed with exit {completed.returncode}: {stderr}"
            )

        artifacts = self._discover_artifacts(request.output_dir)
        metadata: dict[str, Any] = {
            "command": [Path(command[0]).name, *command[1:]],
            "stdout_tail": completed.stdout[-4000:],
            "artifact_count": len(artifacts),
        }
        return WorldResult(
            backend=self.name,
            operation=Operation.RECONSTRUCT,
            artifacts=artifacts,
            metadata=metadata,
        )

    def generate(self, request: WorldRequest) -> WorldResult:
        options = dict(request.options)
        profile_name = str(options.get("profile", "official"))
        if profile_name != "official":
            raise BackendUnavailable(
                f"HYWorld2 worldgen profile {profile_name!r} is not installed; "
                "single-GPU community patches must be integrated as a separate profile"
            )

        worldgen_root_value = options.get("worldgen_root")
        if not worldgen_root_value:
            raise BackendUnavailable(
                "HYWorld2 official worldgen requires options.worldgen_root pointing to "
                "hyworld2/worldgen in a pinned upstream checkout"
            )
        llm_addr = str(options.get("llm_addr", "")).strip()
        llm_name = str(options.get("llm_name", "")).strip()
        if not llm_addr or not llm_name:
            raise BackendUnavailable(
                "HYWorld2 official worldgen requires llm_addr and llm_name for WorldNav"
            )

        profile = OfficialWorldgenProfile(
            worldgen_root=Path(str(worldgen_root_value)),
            target_path=request.input_path,
            result_dir=request.output_dir,
            llm_addr=llm_addr,
            llm_port=int(options.get("llm_port", 8000)),
            llm_name=llm_name,
            nproc=int(options.get("nproc", 8)),
            python=str(options.get("python", sys.executable)),
            cuda_visible_devices=options.get("cuda_visible_devices"),
            max_steps=(int(options["max_steps"]) if "max_steps" in options else None),
        )
        stage_results = profile.run(
            start_stage=options.get("start_stage"),
            stop_stage=options.get("stop_stage"),
            timeout_s=float(options.get("stage_timeout_s", 6 * 60 * 60)),
        )
        artifacts = self._discover_artifacts(request.output_dir)
        return WorldResult(
            backend=self.name,
            operation=Operation.GENERATE,
            artifacts=artifacts,
            metadata={
                "profile": profile.name,
                "nproc": profile.nproc,
                "max_steps": profile.max_steps,
                "stages": stage_results,
                "artifact_count": len(artifacts),
            },
        )

    @staticmethod
    def _discover_artifacts(root: Path) -> tuple[Artifact, ...]:
        kinds = {
            ".ply": "point_cloud",
            ".splat": "gaussian_splat",
            ".spz": "gaussian_splat",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".json": "metadata",
            ".npz": "tensor_data",
            ".npy": "tensor_data",
        }
        found: list[Artifact] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            kind = kinds.get(path.suffix.lower())
            if kind:
                found.append(Artifact(kind=kind, path=path))
        return tuple(found)

    @staticmethod
    def integration_manifest() -> str:
        return json.dumps(
            {
                "backend": "hyworld2",
                "reconstruct_module": "hyworld2.worldrecon.pipeline",
                "full_worldgen": "official-five-stage-profile",
            },
            sort_keys=True,
        )
