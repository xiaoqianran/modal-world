"""Modal deployment boundary for modal-world."""

from __future__ import annotations

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_SOURCE,
    hyworld2_artifact_image,
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
