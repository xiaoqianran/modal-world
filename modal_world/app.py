"""Modal deployment boundary for modal-world."""

from __future__ import annotations

import modal

from .hyworld2_runtime import GPU, hyworld2_artifact_image
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
        "torch": torch.__version__,
        "gsplat": importlib.metadata.version("gsplat"),
        "pytorch3d": importlib.metadata.version("pytorch3d"),
        "fused_ssim": importlib.metadata.version("fused-ssim"),
        "spz": importlib.metadata.version("spz"),
        "recast": importlib.metadata.version("recast"),
        "moge": importlib.metadata.version("moge"),
        "nerfview": importlib.metadata.version("nerfview"),
    }
