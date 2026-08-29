"""Modal deployment boundary for modal-world."""

from __future__ import annotations

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_SOURCE,
    hyworld2_artifact_image,
    hyworld2_worldgen_stage1_image,
    hyworld2_worldmirror_image,
)
from .service import capabilities as local_capabilities

app = modal.App("modal-world")

base_image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=base_image)
def capabilities() -> list[dict]:
    return local_capabilities()


@app.function(image=hyworld2_artifact_image, gpu=GPU, timeout=10 * 60)
def hyworld2_artifact_smoke() -> dict:
    """Prove modal-world can consume modal-build artifacts on the target Blackwell GPU."""
    import importlib.metadata
    import inspect

    import torch
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    from pytorch3d.ops import knn_points

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            f"expected sm_120, got {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}"
        )
    required = {"distloss", "gauss_masks"}
    missing = required - set(inspect.signature(rasterization).parameters)
    if missing:
        raise RuntimeError(f"HY custom gsplat missing arguments: {sorted(missing)}")

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.tensor([[0.1, 0.1, 0.1]], device="cuda")
    opacities = torch.tensor([0.9], device="cuda")
    colors = torch.tensor([[[0.8, 0.2, 0.1]]], device="cuda")
    viewmats = torch.eye(4, device="cuda")[None]
    ks = torch.tensor([[[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]], device="cuda")
    _, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=ks,
        width=64,
        height=64,
        sh_degree=0,
        packed=False,
        distloss=True,
        gauss_masks=torch.ones((1,), device="cuda"),
    )
    if alpha.max().item() <= 0:
        raise RuntimeError("modal-world HY gsplat smoke rendered empty alpha")

    points = torch.rand((1, 16, 3), device="cuda")
    if knn_points(points, points, K=1).dists.numel() != 16:
        raise RuntimeError("modal-world PyTorch3D KNN smoke failed")
    image = torch.rand((1, 3, 32, 32), device="cuda")
    if float(fused_ssim(image, image)) < 0.99:
        raise RuntimeError("modal-world fused-ssim smoke failed")

    import recast
    import spz  # noqa: F401

    if not hasattr(recast, "RecastNavMesh"):
        raise RuntimeError("modal-world recast binding missing RecastNavMesh")

    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "gsplat": importlib.metadata.version("gsplat"),
        "pytorch3d": importlib.metadata.version("pytorch3d"),
        "fused_ssim": importlib.metadata.version("fused-ssim"),
        "spz": importlib.metadata.version("spz"),
        "recast": importlib.metadata.version("recast"),
        "moge": importlib.metadata.version("moge"),
        "nerfview": importlib.metadata.version("nerfview"),
    }


model_cache = modal.Volume.from_name("hyworld2-models", create_if_missing=True)
inference_outputs = modal.Volume.from_name("hyworld2-inference-output", create_if_missing=True)


@app.function(
    image=hyworld2_worldmirror_image,
    gpu=GPU,
    volumes={"/models": model_cache, "/outputs": inference_outputs},
    timeout=60 * 60,
)
def worldmirror_office_inference() -> dict:
    """Run a real single-image WorldMirror 2.0 inference on the official Office sample."""
    import gc
    import os
    import shutil
    import sys
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, HYWORLD2_SOURCE)

    import torch
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    sample = Path(HYWORLD2_SOURCE) / "examples/worldrecon/realistic/Office/Office.jpg"
    if not sample.is_file():
        raise RuntimeError(f"official sample missing: {sample}")

    output_dir = Path("/outputs/worldmirror-office-smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipeline = WorldMirrorPipeline.from_pretrained(
        "tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",
        enable_bf16=True,
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started

    infer_started = time.perf_counter()
    pipeline(
        str(sample),
        strict_output_path=str(output_dir),
        target_size=952,
        save_depth=True,
        save_normal=True,
        save_gs=True,
        save_camera=True,
        save_points=True,
        apply_sky_mask=False,
        apply_edge_mask=True,
        log_time=True,
    )
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - infer_started
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    files = []
    total_bytes = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            files.append({"path": str(path.relative_to(output_dir)), "bytes": size})

    if not files:
        raise RuntimeError("WorldMirror inference completed without output files")
    suffixes = {Path(item["path"]).suffix.lower() for item in files}
    if ".ply" not in suffixes:
        raise RuntimeError(f"WorldMirror output contains no PLY artifact: {sorted(suffixes)}")

    model_cache.commit()
    inference_outputs.commit()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "sample": str(sample.relative_to(HYWORLD2_SOURCE)),
        "load_s": round(load_s, 3),
        "inference_s": round(infer_s, 3),
        "peak_allocated_gb": round(peak_gb, 3),
        "output_dir": str(output_dir),
        "total_output_bytes": total_bytes,
        "files": files,
    }


worldgen_outputs = modal.Volume.from_name("hyworld2-worldgen-output", create_if_missing=True)
hf_secret = modal.Secret.from_name("hyworld2-hf")


@app.function(
    image=hyworld2_worldgen_stage1_image,
    gpu=GPU,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def worldgen_case000_stage1() -> dict:
    """Run official HYWorld2 WorldNav Stage 1 on the official case000 panorama."""
    import os
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    sys.path.insert(0, HYWORLD2_SOURCE)
    from modal_world.qwen_vlm_server import Qwen3VLEngine, start_openai_server

    source_case = Path(HYWORLD2_SOURCE) / "examples/worldgen/case000"
    target = Path("/worldgen/case000")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_case, target)

    torch.cuda.reset_peak_memory_stats()
    vlm_started = time.perf_counter()
    engine = Qwen3VLEngine("Qwen/Qwen3-VL-8B-Instruct")
    server, _thread = start_openai_server(engine, port=8000)
    vlm_load_s = time.perf_counter() - vlm_started
    model_cache.commit()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"local Qwen3-VL server unhealthy: {response.status}")

        worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
        panorama_utils = worldgen_root / "src/panorama_utils.py"
        panorama_source = panorama_utils.read_text()
        old_camera_code = """def get_panorama_cameras_v2(subdivisions=0):
    vertices = subdivide_icosahedron(subdivisions=subdivisions)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.numpy.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)
"""
        new_camera_code = """def get_panorama_cameras_v2(subdivisions=0):
    vertices = subdivide_icosahedron(subdivisions=subdivisions)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    eye = np.zeros_like(vertices, dtype=np.float32)
    up = np.broadcast_to(np.array([0, 0, 1], dtype=np.float32), vertices.shape).copy()
    view = vertices / np.linalg.norm(vertices, axis=-1, keepdims=True)
    pole_mask = np.abs(view[:, 2]) > 0.999
    up[pole_mask] = np.array([0, 1, 0], dtype=np.float32)
    extrinsics = utils3d.numpy.extrinsics_look_at(eye, vertices, up).astype(np.float32)
    if not np.isfinite(extrinsics).all():
        raise RuntimeError("non-finite panorama camera extrinsics after pole-safe up selection")
    return extrinsics, [intrinsics] * len(vertices)
"""
        if old_camera_code not in panorama_source:
            raise RuntimeError("expected upstream get_panorama_cameras_v2 block not found")
        panorama_utils.write_text(panorama_source.replace(old_camera_code, new_camera_code, 1))

        log_path = target / "stage1.log"
        command = [
            sys.executable,
            "traj_generate.py",
            "--target_path",
            str(target),
            "--llm_addr",
            "127.0.0.1",
            "--llm_port",
            "8000",
            "--llm_name",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--apply_nav_traj",
            "--apply_up_route",
            "--apply_recon_iteration",
            "--force_vlm",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
        stage_started = time.perf_counter()
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
        stage1_s = time.perf_counter() - stage_started
        if completed.returncode != 0:
            model_cache.commit()
            worldgen_outputs.commit()
            tail = log_path.read_text(errors="replace")[-12000:]
            raise RuntimeError(f"WorldGen Stage 1 failed with exit {completed.returncode}:\n{tail}")
    finally:
        server.shutdown()
        server.server_close()

    required = [
        target / "meta_info.json",
        target / "objects.json",
        target / "camera_trajectory",
        target / "navmesh",
        target / "render_results",
    ]
    missing = [str(path.relative_to(target)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Stage 1 completed but required outputs are missing: {missing}")

    files = []
    total_bytes = 0
    for path in sorted(target.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            files.append({"path": str(path.relative_to(target)), "bytes": size})

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    model_cache.commit()
    worldgen_outputs.commit()
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "vlm_load_s": round(vlm_load_s, 3),
        "stage1_s": round(stage1_s, 3),
        "peak_allocated_gb": round(peak_gb, 3),
        "target": str(target),
        "total_output_bytes": total_bytes,
        "file_count": len(files),
        "files": files,
        "stage1_log_tail": (target / "stage1.log").read_text(errors="replace")[-8000:],
    }
