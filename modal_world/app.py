"""Modal deployment boundary for modal-world."""

from __future__ import annotations

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_SOURCE,
    hyworld2_artifact_image,
    hyworld2_worldgen_stage1_image,
    hyworld2_worldgen_stage3_image,
    hyworld2_worldgen_stage5_image,
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
    if not target.exists():
        shutil.copytree(source_case, target)
    else:
        source_panorama = source_case / "panorama.png"
        target_panorama = target / "panorama.png"
        if not target_panorama.exists():
            shutil.copy2(source_panorama, target_panorama)

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
        panorama_source = panorama_source.replace(old_camera_code, new_camera_code, 1)

        old_mesh_cleanup = """    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()

    # ========== 6. Boundary handling. ==========
    if connect_boundary_max_dist is not None and connect_boundary_max_dist > 0:
        mesh = _fill_small_boundary_spikes(mesh, connect_boundary_max_dist, connect_boundary_repeat_times)
        # Recompute normals after potential modification, if mesh still valid
        if mesh.has_triangles() and mesh.has_vertices():
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()  # Also computes triangle normals if vertex normals are computed

    return mesh
"""
        new_mesh_cleanup = """    # modal-world single-GPU profile: faces come from a structured panorama grid.
    # Open3D 0.18 native cleanup segfaults on the ~1.8M-vertex case000 mesh on Blackwell runtime.
    # Preserve the structured vertices/faces and skip boundary repair; later WorldNav code only
    # consumes vertices/triangles for rendering and navmesh construction.
    if not np.isfinite(vertices_np).all():
        raise RuntimeError("non-finite panorama mesh vertices")
    if faces_np.size and (faces_np.min() < 0 or faces_np.max() >= len(vertices_np)):
        raise RuntimeError("panorama mesh face index out of range")
    print(
        f"[modal-world] safe panorama mesh: vertices={len(vertices_np)} faces={len(faces_np)}; "
        "skipping Open3D native cleanup/boundary repair",
        flush=True,
    )
    return mesh
"""
        if old_mesh_cleanup not in panorama_source:
            raise RuntimeError("expected upstream Open3D mesh cleanup block not found")
        panorama_source = panorama_source.replace(old_mesh_cleanup, new_mesh_cleanup, 1)

        mesh_assign_old = """    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
"""
        mesh_assign_new = """    vertices_np = np.ascontiguousarray(vertices_np, dtype=np.float64)
    faces_np = np.ascontiguousarray(faces_np, dtype=np.int32)
    colors_np = np.ascontiguousarray(colors_np, dtype=np.float64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise RuntimeError(f"invalid panorama mesh vertex shape: {vertices_np.shape}")
    if not np.isfinite(vertices_np).all():
        bad = np.argwhere(~np.isfinite(vertices_np))[:20]
        raise RuntimeError(f"non-finite panorama mesh vertices before Open3D: {bad.tolist()}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise RuntimeError(f"invalid panorama mesh face shape: {faces_np.shape}")
    if faces_np.size and (faces_np.min() < 0 or faces_np.max() >= len(vertices_np)):
        raise RuntimeError(
            f"panorama mesh face index out of range: min={faces_np.min()} max={faces_np.max()} vertices={len(vertices_np)}"
        )
    if colors_np.shape != vertices_np.shape:
        raise RuntimeError(f"panorama mesh color shape mismatch: {colors_np.shape} vs {vertices_np.shape}")
    print(
        f"[modal-world] mesh precheck: vertices={vertices_np.shape} dtype={vertices_np.dtype} "
        f"contiguous={vertices_np.flags.c_contiguous} min={vertices_np.min(axis=0).tolist()} "
        f"max={vertices_np.max(axis=0).tolist()} faces={faces_np.shape} "
        f"face_min={int(faces_np.min()) if faces_np.size else -1} "
        f"face_max={int(faces_np.max()) if faces_np.size else -1}",
        flush=True,
    )
    _os = __import__("os")
    _json = __import__("json")
    debug_dir = _os.environ.get("MODAL_WORLD_MESH_DEBUG_DIR")
    if debug_dir:
        _os.makedirs(debug_dir, exist_ok=True)
        np.save(_os.path.join(debug_dir, "vertices_head.npy"), vertices_np[:10000])
        np.save(_os.path.join(debug_dir, "faces_head.npy"), faces_np[:20000])
        with open(_os.path.join(debug_dir, "mesh_stats.json"), "w") as fh:
            _json.dump(
                {
                    "vertices_shape": list(vertices_np.shape),
                    "vertices_dtype": str(vertices_np.dtype),
                    "vertices_min": vertices_np.min(axis=0).tolist(),
                    "vertices_max": vertices_np.max(axis=0).tolist(),
                    "faces_shape": list(faces_np.shape),
                    "faces_dtype": str(faces_np.dtype),
                    "faces_min": int(faces_np.min()) if faces_np.size else None,
                    "faces_max": int(faces_np.max()) if faces_np.size else None,
                },
                fh,
                indent=2,
            )
    print("[modal-world] Open3D Vector3dVector begin", flush=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    print("[modal-world] Open3D Vector3dVector ok", flush=True)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    print("[modal-world] Open3D Vector3iVector ok", flush=True)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
    print("[modal-world] Open3D vertex colors ok", flush=True)
"""
        if mesh_assign_old not in panorama_source:
            raise RuntimeError("expected upstream Open3D mesh assignment block not found")
        panorama_source = panorama_source.replace(mesh_assign_old, mesh_assign_new, 1)
        mesh_resolution_old = "mesh_h, mesh_w = 960, 1920"
        mesh_resolution_new = "mesh_h, mesh_w = 480, 960  # modal-world single-GPU WorldNav mesh"
        if (
            mesh_resolution_old not in panorama_source
            and mesh_resolution_old not in (worldgen_root / "traj_generate.py").read_text()
        ):
            raise RuntimeError("expected upstream WorldNav mesh resolution not found")
        panorama_utils.write_text(panorama_source)

        navi_utils_path = worldgen_root / "src/navi_utils.py"
        navi_source = navi_utils_path.read_text()
        old_rotation = """        R_to_yup = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        mesh.rotate(R_to_yup, center=(0, 0, 0))

        verts = [(float(x), float(y), float(z)) for x, y, z in np.asarray(mesh.vertices)]
        faces = [(int(a), int(b), int(c)) for a, b, c in np.asarray(mesh.triangles)]
"""
        new_rotation = """        # modal-world single-GPU profile: avoid Open3D 0.18 rotation helper segfault.
        # Equivalent to get_rotation_matrix_from_xyz((-pi/2, 0, 0)).
        R_to_yup = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        verts_np = np.asarray(mesh.vertices, dtype=np.float64).copy()
        verts_np = np.ascontiguousarray(verts_np @ R_to_yup.T, dtype=np.float64)
        mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        print(f"[modal-world] NumPy Z-up -> Y-up rotation ok: {verts_np.shape}", flush=True)

        verts = [(float(x), float(y), float(z)) for x, y, z in verts_np]
        faces = [(int(a), int(b), int(c)) for a, b, c in np.asarray(mesh.triangles)]
"""
        if old_rotation not in navi_source:
            raise RuntimeError("expected upstream Open3D rotation block not found")
        navi_utils_path.write_text(navi_source.replace(old_rotation, new_rotation, 1))

        traj_source = (worldgen_root / "traj_generate.py").read_text()
        if mesh_resolution_old not in traj_source:
            raise RuntimeError(
                "expected upstream WorldNav mesh resolution not found in traj_generate.py"
            )
        (worldgen_root / "traj_generate.py").write_text(
            traj_source.replace(mesh_resolution_old, mesh_resolution_new, 1)
        )

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
            "Qwen/Qwen3-VL-8B-Instruct",
            "--apply_nav_traj",
            "--apply_up_route",
            "--apply_recon_iteration",
            "--force_vlm",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
        env["PYTHONFAULTHANDLER"] = "1"
        env["MODAL_WORLD_MESH_DEBUG_DIR"] = str(target / "mesh_debug")
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
        timing = {
            "vlm_load_s": round(vlm_load_s, 3),
            "stage1_s": round(stage1_s, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "returncode": completed.returncode,
        }
        (target / "wrapper_timing.json").write_text(
            __import__("json").dumps(timing, indent=2) + "\n"
        )
        if completed.returncode != 0:
            model_cache.commit()
            worldgen_outputs.commit()
            tail = log_path.read_text(errors="replace")[-20000:]
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


@app.function(
    image=hyworld2_worldgen_stage1_image,
    gpu=GPU,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def worldgen_case000_stage2() -> dict:
    """Render Stage 1 trajectories and caption them with local Qwen3-VL."""
    import json
    import os
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

    target = Path("/worldgen/case000")
    if not (target / "camera_trajectory/target_camera.json").is_file():
        raise RuntimeError("Stage 1 camera trajectory is missing")
    if not (target / "render_results/global_pcd.ply").is_file():
        raise RuntimeError("Stage 1 global point cloud is missing")

    camera_files = sorted(target.glob("render_results/view*/traj*/camera.json"))
    renders = sorted(target.glob("render_results/view*/traj*/render.mp4"))
    masks = sorted(target.glob("render_results/view*/traj*/render_mask.mp4"))
    captions = sorted(target.glob("render_results/view*/traj*/traj_caption.json"))
    if camera_files and len(camera_files) == len(renders) == len(masks) == len(captions):
        for caption in captions:
            payload = json.loads(caption.read_text())
            if not str(payload.get("prompt", "")).strip():
                raise RuntimeError(f"empty Stage 2 caption: {caption}")
        return {
            "resumed": True,
            "stage2_s": 0.0,
            "render_count": len(renders),
            "mask_count": len(masks),
            "caption_count": len(captions),
            "render_bytes": sum(path.stat().st_size for path in renders),
        }

    torch.cuda.reset_peak_memory_stats()
    vlm_started = time.perf_counter()
    engine = Qwen3VLEngine("Qwen/Qwen3-VL-8B-Instruct")
    server, _thread = start_openai_server(engine, port=8000)
    vlm_load_s = time.perf_counter() - vlm_started
    model_cache.commit()

    log_path = target / "stage2.log"
    timing_path = target / "stage2_timing.json"
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"local Qwen3-VL server unhealthy: {response.status}")

        worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "traj_render.py",
            "--target_path",
            str(target),
            "--llm_addr",
            "127.0.0.1",
            "--llm_port",
            "8000",
            "--llm_name",
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
        env["PYTHONFAULTHANDLER"] = "1"
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
        stage2_s = time.perf_counter() - started
        timing = {
            "vlm_load_s": round(vlm_load_s, 3),
            "stage2_s": round(stage2_s, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "returncode": completed.returncode,
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n")
        worldgen_outputs.commit()
        if completed.returncode != 0:
            tail = log_path.read_text(errors="replace")[-24000:]
            raise RuntimeError(f"WorldGen Stage 2 failed with exit {completed.returncode}:\n{tail}")
    finally:
        server.shutdown()
        server.server_close()

    renders = sorted(target.glob("render_results/*/traj*/render.mp4"))
    masks = sorted(target.glob("render_results/*/traj*/render_mask.mp4"))
    captions = sorted(target.glob("render_results/*/traj*/traj_caption.json"))
    if not renders:
        raise RuntimeError("Stage 2 completed without rendered trajectory videos")
    if len(masks) != len(renders):
        raise RuntimeError(f"Stage 2 render/mask count mismatch: {len(renders)} vs {len(masks)}")
    if not captions:
        raise RuntimeError("Stage 2 completed without trajectory captions")

    worldgen_outputs.commit()
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": str(torch.__version__),
        "vlm_load_s": round(vlm_load_s, 3),
        "stage2_s": round(stage2_s, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "render_count": len(renders),
        "mask_count": len(masks),
        "caption_count": len(captions),
        "render_bytes": sum(path.stat().st_size for path in renders),
        "stage2_log_tail": log_path.read_text(errors="replace")[-8000:],
    }


@app.function(
    image=hyworld2_worldgen_stage3_image,
    cpu=8.0,
    memory=16384,
    volumes={"/models": model_cache},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def preload_worldstereo_stage3_weights() -> dict:
    """Populate all Stage 3 HF assets on CPU so the GPU worker can run fully offline."""
    import os
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    snapshots = {}
    specs = [
        (
            "hanshanxue/WorldStereo",
            [
                "worldstereo-memory-dmd/config.json",
                "worldstereo-memory-dmd/model.safetensors",
            ],
        ),
        (
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            [
                "model_index.json",
                "transformer/**",
                "text_encoder/**",
                "image_encoder/**",
                "image_processor/**",
                "tokenizer/**",
                "vae/**",
                "scheduler/**",
            ],
        ),
        ("Ruicheng/moge-2-vitl-normal", None),
        ("facebook/sam3", None),
        ("facebook/dinov2-base", None),
    ]
    for repo_id, allow_patterns in specs:
        repo_started = time.perf_counter()
        path = snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        snapshots[repo_id] = {
            "path": path,
            "elapsed_s": round(time.perf_counter() - repo_started, 3),
        }
        model_cache.commit()

    elapsed = time.perf_counter() - started
    cache_root = Path("/models/huggingface/hub")
    file_count = (
        sum(1 for path in cache_root.rglob("*") if path.is_file()) if cache_root.exists() else 0
    )
    return {
        "elapsed_s": round(elapsed, 3),
        "repos": snapshots,
        "cache_file_count": file_count,
    }


@app.function(
    image=hyworld2_worldgen_stage3_image,
    cpu=4.0,
    memory=16384,
    volumes={"/models": model_cache},
    secrets=[hf_secret],
    timeout=30 * 60,
)
def verify_worldstereo_stage3_cache() -> dict:
    """Verify Stage 3 assets are complete and resolvable with networking disabled."""
    import json
    import os
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"

    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.schedulers import UniPCMultistepScheduler
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoConfig, AutoImageProcessor, CLIPImageProcessor, T5TokenizerFast

    worldstereo_repo = "hanshanxue/WorldStereo"
    wan_repo = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    report_path = Path("/models/stage3_cache_verify.json")
    report = {"offline": True, "steps": {}, "required_paths": {}}
    started = time.perf_counter()

    def checked(name, func):
        step_started = time.perf_counter()
        value = func()
        report["steps"][name] = {"elapsed_s": round(time.perf_counter() - step_started, 3)}
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        model_cache.commit()
        return value

    wan_allow_patterns = [
        "model_index.json",
        "transformer/**",
        "text_encoder/**",
        "image_encoder/**",
        "image_processor/**",
        "tokenizer/**",
        "vae/**",
        "scheduler/**",
    ]

    report["required_paths"]["worldstereo_config"] = checked(
        "worldstereo_config",
        lambda: hf_hub_download(
            worldstereo_repo,
            "config.json",
            subfolder="worldstereo-memory-dmd",
            local_files_only=True,
        ),
    )
    report["required_paths"]["worldstereo_weights"] = checked(
        "worldstereo_weights",
        lambda: hf_hub_download(
            worldstereo_repo,
            "model.safetensors",
            subfolder="worldstereo-memory-dmd",
            local_files_only=True,
        ),
    )
    report["required_paths"]["wan_snapshot"] = checked(
        "wan_snapshot",
        lambda: snapshot_download(
            wan_repo,
            allow_patterns=wan_allow_patterns,
            local_files_only=True,
        ),
    )
    report["required_paths"]["moge_snapshot"] = checked(
        "moge_snapshot",
        lambda: snapshot_download("Ruicheng/moge-2-vitl-normal", local_files_only=True),
    )
    report["required_paths"]["sam3_snapshot"] = checked(
        "sam3_snapshot",
        lambda: snapshot_download("facebook/sam3", local_files_only=True),
    )
    report["required_paths"]["dinov2_snapshot"] = checked(
        "dinov2_snapshot",
        lambda: snapshot_download("facebook/dinov2-base", local_files_only=True),
    )

    checked(
        "dinov2_config",
        lambda: AutoConfig.from_pretrained("facebook/dinov2-base", local_files_only=True),
    )
    checked(
        "dinov2_image_processor",
        lambda: AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", use_fast=True, local_files_only=True
        ),
    )

    checked(
        "wan_transformer_config",
        lambda: WanTransformer3DModel.load_config(
            wan_repo, subfolder="transformer", local_files_only=True
        ),
    )
    checked(
        "wan_vae_config",
        lambda: AutoencoderKLWan.load_config(wan_repo, subfolder="vae", local_files_only=True),
    )
    checked(
        "wan_scheduler_config",
        lambda: UniPCMultistepScheduler.load_config(
            wan_repo, subfolder="scheduler", local_files_only=True
        ),
    )
    checked(
        "wan_text_encoder_config",
        lambda: AutoConfig.from_pretrained(
            wan_repo, subfolder="text_encoder", local_files_only=True
        ),
    )
    checked(
        "wan_image_encoder_config",
        lambda: AutoConfig.from_pretrained(
            wan_repo, subfolder="image_encoder", local_files_only=True
        ),
    )
    checked(
        "wan_tokenizer",
        lambda: T5TokenizerFast.from_pretrained(
            wan_repo, subfolder="tokenizer", local_files_only=True
        ),
    )
    checked(
        "wan_image_processor",
        lambda: CLIPImageProcessor.from_pretrained(
            wan_repo, subfolder="image_processor", local_files_only=True
        ),
    )

    cache = Path("/models/huggingface/hub")
    repo_dirs = {
        "worldstereo": cache / "models--hanshanxue--WorldStereo" / "blobs",
        "wan": cache / "models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers" / "blobs",
        "moge": cache / "models--Ruicheng--moge-2-vitl-normal" / "blobs",
        "sam3": cache / "models--facebook--sam3" / "blobs",
        "dinov2": cache / "models--facebook--dinov2-base" / "blobs",
    }
    blob_bytes = {}
    blob_files = {}
    for name, root in repo_dirs.items():
        files = [
            path for path in root.iterdir() if path.is_file() and not path.name.endswith(".lock")
        ]
        if not files:
            raise RuntimeError(f"empty Hugging Face blob cache for {name}: {root}")
        blob_files[name] = len(files)
        blob_bytes[name] = sum(path.stat().st_size for path in files)

    report.update(
        {
            "success": True,
            "blob_bytes": blob_bytes,
            "blob_files": blob_files,
            "total_blob_bytes": sum(blob_bytes.values()),
            "elapsed_s": round(time.perf_counter() - started, 3),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    model_cache.commit()
    return report


@app.function(
    image=hyworld2_worldgen_stage3_image,
    gpu=GPU,
    cpu=16.0,
    memory=131072,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def worldgen_case000_stage3() -> dict:
    """Run single-GPU WorldStereo-2 DMD expansion with fully preloaded weights."""
    import json
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/models/torchinductor"
    os.environ["TRITON_CACHE_DIR"] = "/models/triton"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    target = Path("/worldgen/case000")
    camera_files = sorted(target.glob("render_results/view*/traj*/camera.json"))
    renders = sorted(target.glob("render_results/view*/traj*/render.mp4"))
    masks = sorted(target.glob("render_results/view*/traj*/render_mask.mp4"))
    captions = sorted(target.glob("render_results/view*/traj*/traj_caption.json"))
    if not camera_files or not (len(camera_files) == len(renders) == len(masks) == len(captions)):
        raise RuntimeError(
            f"Stage 2 incomplete: cameras={len(camera_files)} renders={len(renders)} "
            f"masks={len(masks)} captions={len(captions)}"
        )
    for caption in captions:
        payload = json.loads(caption.read_text())
        if not str(payload.get("prompt", "")).strip():
            raise RuntimeError(f"empty Stage 2 caption: {caption}")

    worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
    from modal_world.worldstereo_patch import patch_worldstereo_wrapper

    patch_worldstereo_wrapper(worldgen_root / "models/worldstereo_wrapper.py")

    retrieval_path = worldgen_root / "src/retrieval_wm.py"
    retrieval_source = retrieval_path.read_text()
    dino_processor_old = "            self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)\n"
    dino_processor_new = (
        "            self.processor = AutoImageProcessor.from_pretrained(\n"
        "                model_path, use_fast=True, local_files_only=True\n"
        "            )\n"
    )
    dino_model_old = (
        "            self.model = AutoModel.from_pretrained(model_path).to(self.device)\n"
    )
    dino_model_new = (
        "            self.model = AutoModel.from_pretrained(\n"
        "                model_path, local_files_only=True\n"
        "            ).to(self.device)\n"
    )
    if retrieval_source.count(dino_processor_old) != 1:
        raise RuntimeError("expected pinned DINO processor loader not found")
    if retrieval_source.count(dino_model_old) != 1:
        raise RuntimeError("expected pinned DINO model loader not found")
    retrieval_source = retrieval_source.replace(dino_processor_old, dino_processor_new, 1)
    retrieval_source = retrieval_source.replace(dino_model_old, dino_model_new, 1)
    retrieval_path.write_text(retrieval_source)

    log_path = target / "stage3.log"
    timing_path = target / "stage3_timing.json"
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "video_gen.py",
        "--target_path",
        str(target),
        "--model_type",
        "worldstereo-memory-dmd",
        "--local_files_only",
        "--skip_exist",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
    env["PYTHONFAULTHANDLER"] = "1"

    stop_monitor = threading.Event()
    gpu_peak_mib = 0

    def monitor_gpu() -> None:
        nonlocal gpu_peak_mib
        while not stop_monitor.wait(1.0):
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).splitlines()[0]
                gpu_peak_mib = max(gpu_peak_mib, int(raw.strip()))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=worldgen_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=3 * 60 * 60,
            )
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    stage3_s = time.perf_counter() - started
    results = sorted(target.glob("render_results/*/traj*/worldstereo-memory-dmd_result.mp4"))
    aligned_pcd = target / "render_results/generation_bank_worldstereo-memory-dmd/aligned_pcd.ply"
    timing = {
        "stage3_s": round(stage3_s, 3),
        "gpu_peak_used_mib": gpu_peak_mib,
        "returncode": completed.returncode,
        "result_count": len(results),
        "aligned_pcd_exists": aligned_pcd.is_file(),
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    model_cache.commit()
    worldgen_outputs.commit()
    if completed.returncode != 0:
        tail = log_path.read_text(errors="replace")[-30000:]
        raise RuntimeError(f"WorldGen Stage 3 failed with exit {completed.returncode}:\n{tail}")
    if len(results) != len(camera_files):
        raise RuntimeError(f"Stage 3 result count mismatch: {len(results)} vs {len(camera_files)}")
    if not aligned_pcd.is_file():
        raise RuntimeError("Stage 3 completed without aligned memory-bank point cloud")
    return {
        **timing,
        "result_bytes": sum(path.stat().st_size for path in results),
        "aligned_pcd_bytes": aligned_pcd.stat().st_size,
        "stage3_log_tail": log_path.read_text(errors="replace")[-8000:],
    }


@app.function(
    image=hyworld2_worldgen_stage1_image,
    gpu=GPU,
    cpu=8.0,
    memory=32768,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def worldgen_case000_stage4() -> dict:
    """Prepare official HYWorld2 3DGS training data on one RTX PRO 6000."""
    import json
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    target = Path("/worldgen/case000")
    generation_bank = target / "render_results/generation_bank_worldstereo-memory-dmd"
    required_stage3 = [generation_bank / "global_pcd.ply", generation_bank / "aligned_pcd.ply"]
    missing_stage3 = [
        str(path.relative_to(target)) for path in required_stage3 if not path.is_file()
    ]
    if missing_stage3:
        raise RuntimeError(f"Stage 3 incomplete: missing {missing_stage3}")

    gs_data = target / "gs_data"
    cameras_path = gs_data / "cameras.json"
    points_path = gs_data / "points.ply"
    sky_points_path = gs_data / "sky_points.ply"
    if cameras_path.is_file() and points_path.is_file():
        payload = json.loads(cameras_path.read_text())
        camera_count = len([key for key in payload if key not in {"width", "height"}])
        images = sorted((gs_data / "images").glob("*.png"))
        depths = sorted((gs_data / "depths").glob("*.png"))
        normals = sorted((gs_data / "normals").glob("*.png"))
        if camera_count and len(images) == camera_count and len(normals) == camera_count:
            return {
                "resumed": True,
                "stage4_s": 0.0,
                "camera_count": camera_count,
                "image_count": len(images),
                "depth_count": len(depths),
                "normal_count": len(normals),
                "points_bytes": points_path.stat().st_size,
                "sky_points_bytes": sky_points_path.stat().st_size,
            }

    worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
    script_path = worldgen_root / "gen_gs_data.py"
    script_source = script_path.read_text()
    moge_old = 'MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)'
    moge_new = (
        'MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal", local_files_only=True).to(device)'
    )
    if script_source.count(moge_old) != 1:
        raise RuntimeError("expected pinned Stage 4 MoGe loader not found")
    script_path.write_text(script_source.replace(moge_old, moge_new, 1))

    log_path = target / "stage4.log"
    timing_path = target / "stage4_timing.json"
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "gen_gs_data.py",
        "--root_path",
        str(target),
        "--save_normal",
        "--split_sky",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
    env["PYTHONFAULTHANDLER"] = "1"

    stop_monitor = threading.Event()
    gpu_peak_mib = 0

    def monitor_gpu() -> None:
        nonlocal gpu_peak_mib
        while not stop_monitor.wait(1.0):
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).splitlines()[0]
                gpu_peak_mib = max(gpu_peak_mib, int(raw.strip()))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=worldgen_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=50 * 60,
            )
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    stage4_s = time.perf_counter() - started
    camera_count = 0
    if cameras_path.is_file():
        payload = json.loads(cameras_path.read_text())
        camera_count = len([key for key in payload if key not in {"width", "height"}])
    images = sorted((gs_data / "images").glob("*.png"))
    depths = sorted((gs_data / "depths").glob("*.png"))
    normals = sorted((gs_data / "normals").glob("*.png"))
    timing = {
        "stage4_s": round(stage4_s, 3),
        "gpu_peak_used_mib": gpu_peak_mib,
        "returncode": completed.returncode,
        "camera_count": camera_count,
        "image_count": len(images),
        "depth_count": len(depths),
        "normal_count": len(normals),
        "points_exists": points_path.is_file(),
        "sky_points_exists": sky_points_path.is_file(),
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    worldgen_outputs.commit()

    if completed.returncode != 0:
        tail = log_path.read_text(errors="replace")[-30000:]
        raise RuntimeError(f"WorldGen Stage 4 failed with exit {completed.returncode}:\n{tail}")
    if not cameras_path.is_file() or not points_path.is_file():
        raise RuntimeError("Stage 4 completed without required GS dataset files")
    if not camera_count or len(images) != camera_count or len(normals) != camera_count:
        raise RuntimeError(
            f"Stage 4 dataset count mismatch: cameras={camera_count} images={len(images)} "
            f"normals={len(normals)} depths={len(depths)}"
        )

    return {
        **timing,
        "points_bytes": points_path.stat().st_size,
        "sky_points_bytes": sky_points_path.stat().st_size,
        "stage4_log_tail": log_path.read_text(errors="replace")[-8000:],
    }


@app.function(
    image=hyworld2_worldgen_stage5_image,
    cpu=8.0,
    memory=32768,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    timeout=45 * 60,
)
def preflight_worldgen_case000_stage5() -> dict:
    """CPU-only Stage 5 preflight: cache LPIPS and parse the real GS dataset."""
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["XDG_CACHE_HOME"] = "/models/cache"
    os.environ["PYTHONPATH"] = f"{HYWORLD2_SOURCE}/hyworld2/worldgen:{HYWORLD2_SOURCE}"

    data_dir = Path("/worldgen/case000/gs_data")
    required = [
        data_dir / "cameras.json",
        data_dir / "points.ply",
        data_dir / "meta_info.json",
        data_dir / "images",
        data_dir / "normals",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Stage 5 preflight missing GS data: {missing}")

    started = time.perf_counter()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_started = time.perf_counter()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
    del metric
    lpips_s = time.perf_counter() - lpips_started
    model_cache.commit()

    worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
    help_started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "world_gs_trainer", "default", "--help"],
        cwd=worldgen_root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=120,
    )
    help_s = time.perf_counter() - help_started
    if completed.returncode != 0:
        raise RuntimeError(f"Stage 5 trainer CLI failed:\n{completed.stdout[-12000:]}")

    sys.path.insert(0, str(worldgen_root))
    from gs.opencv import Dataset, Parser

    parser_started = time.perf_counter()
    parser = Parser(
        data_dir=str(data_dir),
        factor=1,
        normalize=True,
        test_every=32,
        downsample_pts_num=1_000_000,
        downsample_mode="geometry_aware",
        detect_anchor_candidates=False,
        world_rank=0,
        world_size=1,
        local_rank=0,
    )
    parser_s = time.perf_counter() - parser_started
    trainset = Dataset(
        parser,
        split="train",
        load_depths=True,
        load_normals=True,
    )
    valset = Dataset(parser, split="val")

    depth_available = sum(path is not None for path in parser.depth_dict.values())
    normal_available = sum(path is not None for path in parser.normal_dict.values())
    vgg_cache = Path("/models/torch/hub/checkpoints/vgg16-397923af.pth")
    report = {
        "success": True,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "lpips_cache_s": round(lpips_s, 3),
        "trainer_help_s": round(help_s, 3),
        "parser_s": round(parser_s, 3),
        "image_count": len(parser.image_names),
        "train_count": len(trainset),
        "val_count": len(valset),
        "initial_point_count": len(parser.points),
        "sky_point_count": int(parser.sky_mask.sum()),
        "depth_available": depth_available,
        "normal_available": normal_available,
        "vgg_cache_exists": vgg_cache.is_file(),
        "vgg_cache_bytes": vgg_cache.stat().st_size if vgg_cache.is_file() else 0,
    }
    Path("/models/stage5_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    model_cache.commit()
    return report


@app.function(
    image=hyworld2_worldgen_stage5_image,
    gpu=GPU,
    cpu=16.0,
    memory=65536,
    volumes={"/models": model_cache, "/worldgen": worldgen_outputs},
    timeout=20 * 60,
)
def worldgen_case000_stage5_smoke() -> dict:
    """Run a short real 3DGS optimization to validate the final world-generation stage."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    os.environ["TORCH_HOME"] = "/models/torch"
    os.environ["XDG_CACHE_HOME"] = "/models/cache"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTHONFAULTHANDLER"] = "1"

    target = Path("/worldgen/case000")
    data_dir = target / "gs_data"
    result_dir = target / "gs_smoke_result"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    required = [
        data_dir / "cameras.json",
        data_dir / "points.ply",
        data_dir / "images",
        data_dir / "normals",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Stage 5 smoke missing GS data: {missing}")

    worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
    log_path = target / "stage5_smoke.log"
    timing_path = target / "stage5_smoke_timing.json"
    steps = 100
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-u",
        "-m",
        "world_gs_trainer",
        "default",
        "--data_dir",
        str(data_dir),
        "--result_dir",
        str(result_dir),
        "--max_steps",
        str(steps),
        "--save_steps",
        str(steps),
        "--ply_steps",
        str(steps),
        "--save_ply",
        "--convert_to_spz",
        "--disable_video",
        "--disable_viewer",
        "--use_scale_regularization",
        "--antialiased",
        "--depth_loss",
        "--normal_loss",
        "--sky_depth_from_pcd",
        "--use_mask_gaussian",
        "--mask_export_stochastic",
        "--no-mask-export-anchor-protection",
        "--use_anchor_protection",
        "--strategy.refine-start-iter",
        "10",
        "--strategy.refine-stop-iter",
        "50",
        "--strategy.refine-every",
        "7",
        "--strategy.refine-scale2d-stop-iter",
        "50",
        "--strategy.reset-every",
        "99990",
        "--strategy.grow-grad2d",
        "0.0001",
        "--strategy.prune-scale3d",
        "0.1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"

    stop_monitor = threading.Event()
    gpu_peak_mib = 0

    def monitor_gpu() -> None:
        nonlocal gpu_peak_mib
        while not stop_monitor.wait(1.0):
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                ).splitlines()[0]
                gpu_peak_mib = max(gpu_peak_mib, int(raw.strip()))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=worldgen_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=15 * 60,
            )
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    elapsed = time.perf_counter() - started
    checkpoints = sorted(result_dir.rglob("*.pt"))
    plys = sorted(result_dir.rglob("*.ply"))
    spzs = sorted(result_dir.rglob("*.spz"))
    timing = {
        "steps": steps,
        "stage5_smoke_s": round(elapsed, 3),
        "gpu_peak_used_mib": gpu_peak_mib,
        "returncode": completed.returncode,
        "checkpoint_count": len(checkpoints),
        "ply_count": len(plys),
        "spz_count": len(spzs),
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    worldgen_outputs.commit()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage 5 smoke failed with exit {completed.returncode}:\n"
            f"{log_path.read_text(errors='replace')[-30000:]}"
        )
    if not checkpoints or not plys:
        raise RuntimeError(f"Stage 5 smoke completed without checkpoint/PLY: {timing}")
    return {
        **timing,
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        "ply_bytes": sum(path.stat().st_size for path in plys),
        "spz_bytes": sum(path.stat().st_size for path in spzs),
        "log_tail": log_path.read_text(errors="replace")[-8000:],
    }
