from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Any

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_REVISION,
    HYWORLD2_SOURCE,
    hyworld2_worldgen_stage1_image,
)
from .worldgen_job import (
    build_stage_manifest,
    fingerprint_files,
    manifest_matches,
    resolve_worldgen_job_root,
    stage_manifest_path,
    write_stage_manifest,
)

app = modal.App("modal-world-stage2")
model_cache = modal.Volume.from_name("hyworld2-models", create_if_missing=True)
runtime_cache = modal.Volume.from_name(
    "hyworld2-runtime-cache-v2", create_if_missing=True, version=2
)
worldgen_outputs = modal.Volume.from_name("hyworld2-worldgen-output", create_if_missing=True)
hf_secret = modal.Secret.from_name(os.environ.get("MODAL_WORLD_HF_SECRET", "hyworld2-hf"))

_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


@app.cls(
    image=hyworld2_worldgen_stage1_image,
    gpu=GPU,
    cpu=8.0,
    memory=65536,
    volumes={
        "/models": model_cache.with_mount_options(read_only=True),
        "/runtime-cache": runtime_cache,
        "/worldgen": worldgen_outputs,
    },
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
    startup_timeout=20 * 60,
    min_containers=0,
    max_containers=1,
    scaledown_window=5 * 60,
)
class WorldNavRenderer:
    """Keep Qwen and PyTorch3D alive across worlds to amortize renderer warm-up."""

    @modal.enter()
    def load_runtime(self) -> None:
        os.environ["HF_HOME"] = "/models/huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_CACHE_PATH"] = "/runtime-cache/cuda-cache"
        os.environ["CUDA_CACHE_MAXSIZE"] = str(4 * 1024**3)
        os.environ["TORCH_EXTENSIONS_DIR"] = "/runtime-cache/torch-extensions"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/runtime-cache/torchinductor"
        os.environ["TRITON_CACHE_DIR"] = "/runtime-cache/triton"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        import sys

        worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
        for path in (str(worldgen_root), HYWORLD2_SOURCE):
            if path not in sys.path:
                sys.path.insert(0, path)

        import torch
        from src.general_utils import set_seed
        from src.pointcloud import multi_gpu_point_rendering, point_rendering
        from src.vlm_utils import get_traj_caption

        from modal_world.qwen_vlm_server import Qwen3VLEngine, start_openai_server

        self.torch = torch
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        self.set_seed = set_seed
        self.multi_gpu_point_rendering = multi_gpu_point_rendering
        self.get_traj_caption = get_traj_caption

        started = time.perf_counter()
        self.engine = Qwen3VLEngine(_MODEL_ID, max_batch_size=3, batch_window_s=0.03)
        self.server, self.server_thread = start_openai_server(self.engine, port=8000)
        torch.cuda.synchronize()
        self.model_load_s = time.perf_counter() - started

        warmup_started = time.perf_counter()
        k = torch.tensor([[[500.0, 0.0, 416.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]])
        w2c = torch.eye(4).unsqueeze(0)
        points = torch.tensor(
            [[-0.1, -0.1, 2.0], [0.1, -0.1, 2.0], [-0.1, 0.1, 2.0], [0.1, 0.1, 2.0]],
            dtype=torch.float32,
        )
        colors = torch.tensor(
            [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        point_rendering(
            K=k,
            w2cs=w2c,
            points=points,
            colors=colors,
            device=self.device,
            h=480,
            w=832,
            render_radius=0.008,
            points_per_pixel=20,
        )
        torch.cuda.synchronize()
        self.renderer_warmup_s = time.perf_counter() - warmup_started
        runtime_cache.commit()
        self.call_count = 0

    @modal.exit()
    def close_runtime(self) -> None:
        server = getattr(self, "server", None)
        if server is not None:
            server.shutdown()
            server.server_close()

    @modal.method()
    def generate_nav(self, job_id: str = "case000", force: bool = False) -> dict[str, Any]:
        """Run WorldNav Stage 1 while reusing the persistent Qwen server."""
        worldgen_outputs.reload()
        import subprocess
        import urllib.request

        import torch

        target = resolve_worldgen_job_root(job_id)
        if job_id == "case000":
            source_case = Path(HYWORLD2_SOURCE) / "examples/worldgen/case000"
            if not target.exists():
                shutil.copytree(source_case, target)
            elif not (target / "panorama.png").is_file():
                shutil.copy2(source_case / "panorama.png", target / "panorama.png")
        elif not (target / "panorama.png").is_file():
            raise RuntimeError(
                f"Stage 1 panorama is missing for job {job_id!r}: {target / 'panorama.png'}"
            )

        panorama = target / "panorama.png"
        manifest = build_stage_manifest(
            job_id=job_id,
            stage="stage1",
            hyworld_revision=HYWORLD2_REVISION,
            input_fingerprint=fingerprint_files([panorama], root=target),
            config={
                "profile": "persistent-worldnav-v1",
                "llm": _MODEL_ID,
                "mesh_resolution": [480, 960],
                "apply_nav_traj": True,
                "apply_up_route": True,
                "apply_recon_iteration": True,
            },
        )
        required = [
            target / "meta_info.json",
            target / "objects.json",
            target / "camera_trajectory/target_camera.json",
            target / "render_results/global_pcd.ply",
            target / "navmesh/metadata.json",
        ]
        if not force and all(path.exists() for path in required):
            manifest_ok = manifest_matches(target, "stage1", manifest)
            legacy_adopted = (
                job_id == "case000" and not stage_manifest_path(target, "stage1").exists()
            )
            if manifest_ok or legacy_adopted:
                if legacy_adopted:
                    write_stage_manifest(target, "stage1", manifest)
                    worldgen_outputs.commit()
                return {
                    "resumed": True,
                    "manifest_adopted": legacy_adopted,
                    "worker_call_index": self.call_count,
                    "model_load_s": round(self.model_load_s, 3),
                    "required_outputs": len(required),
                }

        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"persistent Qwen3-VL server unhealthy: {response.status}")

        call_index = self.call_count
        self.call_count += 1
        torch.cuda.reset_peak_memory_stats()
        worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
        log_path = target / "stage1.log"
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-u",
            "traj_generate.py",
            "--target_path",
            str(target),
            "--llm_addr",
            "127.0.0.1",
            "--llm_port",
            "8000",
            "--llm_name",
            _MODEL_ID,
            "--apply_nav_traj",
            "--apply_up_route",
            "--apply_recon_iteration",
            "--force_vlm",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
        env["PYTHONFAULTHANDLER"] = "1"
        env["MODAL_WORLD_MESH_DEBUG_DIR"] = str(target / "mesh_debug")
        started = time.perf_counter()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=worldgen_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=90 * 60,
            )
        stage1_s = time.perf_counter() - started
        log_text = log_path.read_text(errors="replace")
        if completed.returncode != 0:
            worldgen_outputs.commit()
            raise RuntimeError(
                f"WorldGen Stage 1 failed with exit {completed.returncode}:\n{log_text[-20000:]}"
            )
        fatal_navmesh_markers = (
            "Navmesh Error:",
            "Path planning failed:",
            "Artifact saving failed:",
            "NavMesh build failed.",
        )
        found_navmesh_errors = [marker for marker in fatal_navmesh_markers if marker in log_text]
        if found_navmesh_errors:
            worldgen_outputs.commit()
            raise RuntimeError(
                "WorldGen Stage 1 navmesh failed despite a zero subprocess exit: "
                f"{found_navmesh_errors}\n{log_text[-20000:]}"
            )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            worldgen_outputs.commit()
            raise RuntimeError(f"Stage 1 completed but outputs are missing: {missing}")

        write_stage_manifest(target, "stage1", manifest)
        runtime_cache.commit()
        worldgen_outputs.commit()
        return {
            "stage1_s": round(stage1_s, 3),
            "model_load_s": round(self.model_load_s, 3),
            "worker_call_index": call_index,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "target": str(target),
            "stage1_log_tail": log_text[-8000:],
        }

    @staticmethod
    def _manifest(job_id: str, target: Path) -> tuple[dict[str, Any], list[Path]]:
        camera_files = sorted(target.glob("render_results/*/traj*/camera.json"))
        inputs = [
            target / "camera_trajectory/target_camera.json",
            target / "render_results/global_pcd.ply",
            *camera_files,
        ]
        manifest = build_stage_manifest(
            job_id=job_id,
            stage="stage2",
            hyworld_revision=HYWORLD2_REVISION,
            input_fingerprint=fingerprint_files(inputs, root=target),
            config={
                "profile": "persistent-single-gpu-v3",
                "llm": _MODEL_ID,
                "render_radius": 0.008,
                "points_per_pixel": 20,
                "slice_size": 4,
            },
        )
        return manifest, camera_files

    @modal.method()
    def render(self, job_id: str = "case000", force: bool = False) -> dict[str, Any]:
        worldgen_outputs.reload()
        import numpy as np
        import torch
        import trimesh
        from diffusers.utils import export_to_video
        from PIL import Image
        from torchvision import transforms

        target = resolve_worldgen_job_root(job_id)
        if not (target / "camera_trajectory/target_camera.json").is_file():
            raise RuntimeError("Stage 1 camera trajectory is missing")
        if not (target / "render_results/global_pcd.ply").is_file():
            raise RuntimeError("Stage 1 global point cloud is missing")

        manifest, camera_files = self._manifest(job_id, target)
        renders = sorted(target.glob("render_results/*/traj*/render.mp4"))
        masks = sorted(target.glob("render_results/*/traj*/render_mask.mp4"))
        captions = sorted(target.glob("render_results/*/traj*/traj_caption.json"))
        if (
            not force
            and camera_files
            and len(camera_files) == len(renders) == len(masks) == len(captions)
        ):
            valid_captions = True
            for caption in captions:
                payload = json.loads(caption.read_text())
                if not str(payload.get("prompt", "")).strip():
                    valid_captions = False
                    break
            if valid_captions:
                manifest_ok = manifest_matches(target, "stage2", manifest)
                legacy_adopted = (
                    job_id == "case000" and not stage_manifest_path(target, "stage2").exists()
                )
                if manifest_ok or legacy_adopted:
                    if legacy_adopted:
                        write_stage_manifest(target, "stage2", manifest)
                        worldgen_outputs.commit()
                    return {
                        "resumed": True,
                        "manifest_adopted": legacy_adopted,
                        "worker_call_index": self.call_count,
                        "model_load_s": round(self.model_load_s, 3),
                        "renderer_warmup_s": round(self.renderer_warmup_s, 3),
                        "render_count": len(renders),
                        "mask_count": len(masks),
                        "caption_count": len(captions),
                    }

        call_index = self.call_count
        self.call_count += 1
        self.set_seed(1024)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        scene = str(target)
        traj_list = (
            glob(f"{scene}/render_results/view*/traj*")
            + glob(f"{scene}/render_results/target*/traj*")
            + glob(f"{scene}/render_results/wonder*/traj*")
            + glob(f"{scene}/render_results/reconstruct*/traj*")
        )
        traj_list.sort()
        global_pcd = trimesh.load(target / "render_results/global_pcd.ply")
        to_pil = transforms.ToPILImage()
        render_timings: list[dict[str, Any]] = []

        for traj_path_str in traj_list:
            traj_path = Path(traj_path_str)
            camera_path = traj_path / "camera.json"
            if not camera_path.is_file():
                continue
            trajectory_started = time.perf_counter()
            camera_info = json.loads(camera_path.read_text())
            view_id = traj_path.parent.name
            traj_id = traj_path.name
            image_path = target / "render_results" / view_id / "start_frame.png"
            image = Image.open(image_path)
            image_w, image_h = image.size
            ks = torch.tensor(np.array(camera_info["intrinsic"]), dtype=torch.float32)
            w2cs = torch.tensor(np.array(camera_info["extrinsic"]), dtype=torch.float32)
            replace_first_frame = not (view_id.startswith("reconstruct_") and traj_id == "traj1")
            pcd_renders, pcd_mask = self.multi_gpu_point_rendering(
                image=image,
                Ks=ks,
                w2cs=w2cs,
                render_points=global_pcd.vertices,
                render_colors=global_pcd.colors[:, :3] / 255 * 2 - 1,
                image_h=image_h,
                image_w=image_w,
                device=self.device,
                device_num=1,
                render_radius=0.008,
                points_per_pixel=20,
                slice_size=4,
                local_rank=0,
                replace_first_frame=replace_first_frame,
            )
            pcd_renders = pcd_renders.to(torch.float32)
            render_video = [to_pil((frame + 1) / 2) for frame in pcd_renders]
            mask_video = [to_pil(mask) for mask in pcd_mask]
            export_to_video(render_video, str(traj_path / "render.mp4"), fps=16)
            export_to_video(mask_video, str(traj_path / "render_mask.mp4"), fps=16)
            render_timings.append(
                {
                    "trajectory": str(traj_path.relative_to(target)),
                    "elapsed_s": round(time.perf_counter() - trajectory_started, 3),
                }
            )

        total_render_list = sorted(glob(f"{scene}/render_results/*/traj*/render.mp4"))
        caption_inputs = [
            path
            for path in total_render_list
            if not (
                Path(path).parent.parent.name.startswith("reconstruct_")
                and Path(path).parent.name == "traj1"
            )
        ]

        def caption_one(render_path: str) -> tuple[str, bool, str | None]:
            output_path = Path(render_path).with_name("traj_caption.json")
            try:
                caption = self.get_traj_caption("127.0.0.1", 8000, _MODEL_ID, render_path)
                output_path.write_text(json.dumps({"prompt": caption}, indent=2) + "\n")
                return render_path, True, None
            except Exception as exc:  # noqa: BLE001 - preserve which trajectory failed
                return render_path, False, str(exc)

        caption_started = time.perf_counter()
        if caption_inputs:
            with ThreadPoolExecutor(max_workers=min(len(caption_inputs), 32)) as executor:
                futures = [executor.submit(caption_one, path) for path in caption_inputs]
                for future in as_completed(futures):
                    render_path, success, error = future.result()
                    if not success:
                        raise RuntimeError(f"Stage 2 caption failed for {render_path}: {error}")

        for render_path in glob(f"{scene}/render_results/reconstruct_*/traj1/render.mp4"):
            render = Path(render_path)
            source_caption = render.parent.parent / "traj0/traj_caption.json"
            shutil.copy2(source_caption, render.with_name("traj_caption.json"))
        caption_s = time.perf_counter() - caption_started
        elapsed = time.perf_counter() - started

        renders = sorted(target.glob("render_results/*/traj*/render.mp4"))
        masks = sorted(target.glob("render_results/*/traj*/render_mask.mp4"))
        captions = sorted(target.glob("render_results/*/traj*/traj_caption.json"))
        if (
            len(renders) != len(camera_files)
            or len(masks) != len(camera_files)
            or len(captions) != len(camera_files)
        ):
            raise RuntimeError(
                "Stage 2 output count mismatch: "
                f"cameras={len(camera_files)} renders={len(renders)} masks={len(masks)} captions={len(captions)}"
            )

        timing = {
            "stage2_s": round(elapsed, 3),
            "caption_s": round(caption_s, 3),
            "model_load_s": round(self.model_load_s, 3),
            "renderer_warmup_s": round(self.renderer_warmup_s, 3),
            "worker_call_index": call_index,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "trajectory_timings": render_timings,
        }
        (target / "stage2_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        write_stage_manifest(target, "stage2", manifest)
        runtime_cache.commit()
        worldgen_outputs.commit()
        return {
            **timing,
            "render_count": len(renders),
            "mask_count": len(masks),
            "caption_count": len(captions),
            "render_bytes": sum(path.stat().st_size for path in renders),
        }
