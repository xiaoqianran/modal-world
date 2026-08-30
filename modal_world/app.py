"""Modal deployment boundary for modal-world."""

from __future__ import annotations

import os

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_REVISION,
    HYWORLD2_SOURCE,
    hyworld2_artifact_image,
    hyworld2_worldgen_stage1_image,
    hyworld2_worldgen_stage3_image,
    hyworld2_worldgen_stage5_image,
    hyworld2_worldmirror_image,
)
from .service import capabilities as local_capabilities
from .worldgen_job import (
    build_stage_manifest,
    fingerprint_files,
    manifest_matches,
    resolve_worldgen_job_root,
    stage_manifest_path,
    write_stage_manifest,
)

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
runtime_cache = modal.Volume.from_name(
    "hyworld2-runtime-cache-v2", create_if_missing=True, version=2
)
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
hf_secret = modal.Secret.from_name(os.environ.get("MODAL_WORLD_HF_SECRET", "hyworld2-hf"))


def _spawn_worker_call(method, *, job_id: str, wait_timeout_s: float) -> dict:
    """Spawn a deployed worker call with a hard timeout and cost-safe cancellation."""
    call = method.spawn(job_id=job_id, force=False)
    try:
        result = call.get(timeout=wait_timeout_s)
    except TimeoutError as exc:
        try:
            call.cancel(terminate_containers=True)
        finally:
            raise TimeoutError(
                f"worker call timed out after {wait_timeout_s}s and was cancelled: {call.object_id}"
            ) from exc
    if isinstance(result, dict):
        return {**result, "function_call_id": call.object_id}
    return {"result": result, "function_call_id": call.object_id}


@app.function(image=base_image, timeout=2 * 60 * 60)
def worldgen_case000_stage1(job_id: str = "case000") -> dict:
    """Dispatch Stage 1 to the persistent WorldNav/Qwen worker."""
    worker_cls = modal.Cls.from_name("modal-world-stage2", "WorldNavRenderer")
    return _spawn_worker_call(worker_cls().generate_nav, job_id=job_id, wait_timeout_s=30 * 60)


@app.function(
    image=hyworld2_worldgen_stage1_image,
    cpu=4.0,
    memory=16384,
    volumes={"/models": model_cache},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def preload_worldnav_stage1_weights() -> dict:
    """Populate hidden Stage1 HF assets so WorldNav can run fully offline."""
    import os
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    from huggingface_hub import snapshot_download

    specs = [
        ("Qwen/Qwen3-VL-8B-Instruct", None),
        (
            "naver-iv/zim-anything-vitl",
            ["zim_vit_l_2092/**"],
        ),
        ("IDEA-Research/grounding-dino-tiny", None),
        ("Ruicheng/moge-2-vitl-normal", None),
        ("facebook/sam3", None),
    ]
    started = time.perf_counter()
    repos = {}
    for repo_id, allow_patterns in specs:
        repo_started = time.perf_counter()
        path = snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        repos[repo_id] = {
            "path": path,
            "elapsed_s": round(time.perf_counter() - repo_started, 3),
        }
        model_cache.commit()

    zim = Path(repos["naver-iv/zim-anything-vitl"]["path"]) / "zim_vit_l_2092"
    required = [zim / "encoder.onnx", zim / "decoder.onnx"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage1 ZIM preload is incomplete: {missing}")
    return {
        "elapsed_s": round(time.perf_counter() - started, 3),
        "repos": repos,
        "zim_required_files": [str(path) for path in required],
    }


@app.function(
    image=hyworld2_worldgen_stage1_image,
    cpu=4.0,
    memory=16384,
    volumes={"/models": model_cache.with_mount_options(read_only=True)},
    secrets=[hf_secret],
    timeout=20 * 60,
)
def verify_worldnav_stage1_cache() -> dict:
    """Verify Stage 1 assets resolve fully offline before allocating a GPU worker."""
    import json
    import os
    import time
    from pathlib import Path

    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor

    started = time.perf_counter()
    qwen_id = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_snapshot = Path(snapshot_download(qwen_id, local_files_only=True))
    AutoConfig.from_pretrained(qwen_id, local_files_only=True)
    AutoProcessor.from_pretrained(qwen_id, local_files_only=True)

    index_path = qwen_snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted({str(name) for name in index.get("weight_map", {}).values()})
    else:
        shards = ["model.safetensors"]
    missing_shards = [name for name in shards if not (qwen_snapshot / name).is_file()]
    if not shards or missing_shards:
        raise RuntimeError(f"Qwen Stage1 cache incomplete: missing={missing_shards}")

    other_specs = [
        ("naver-iv/zim-anything-vitl", ["zim_vit_l_2092/**"]),
        ("IDEA-Research/grounding-dino-tiny", None),
        ("Ruicheng/moge-2-vitl-normal", None),
        ("facebook/sam3", None),
    ]
    snapshots = {}
    for repo_id, allow_patterns in other_specs:
        snapshots[repo_id] = snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )

    zim = Path(snapshots["naver-iv/zim-anything-vitl"]) / "zim_vit_l_2092"
    zim_required = [zim / "encoder.onnx", zim / "decoder.onnx"]
    missing_zim = [str(path) for path in zim_required if not path.is_file()]
    if missing_zim:
        raise RuntimeError(f"Stage1 ZIM cache incomplete: {missing_zim}")

    return {
        "success": True,
        "offline": True,
        "qwen_snapshot": str(qwen_snapshot),
        "qwen_weight_shards": len(shards),
        "qwen_weight_bytes": sum((qwen_snapshot / name).stat().st_size for name in shards),
        "other_snapshots": snapshots,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


@app.function(image=base_image, timeout=2 * 60 * 60)
def worldgen_case000_stage2(job_id: str = "case000") -> dict:
    """Dispatch Stage 2 to the deployed persistent WorldNav renderer worker."""
    worker_cls = modal.Cls.from_name("modal-world-stage2", "WorldNavRenderer")
    return _spawn_worker_call(worker_cls().render, job_id=job_id, wait_timeout_s=30 * 60)


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
        ("tencent/HY-World-2.0", ["HY-WorldMirror-2.0/**"]),
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
    worldmirror_repo = "tencent/HY-World-2.0"
    worldmirror_subfolder = "HY-WorldMirror-2.0"
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
    worldmirror_snapshot = Path(
        checked(
            "worldmirror_snapshot",
            lambda: snapshot_download(
                worldmirror_repo,
                allow_patterns=[f"{worldmirror_subfolder}/**"],
                local_files_only=True,
            ),
        )
    )
    worldmirror_dir = worldmirror_snapshot / worldmirror_subfolder
    worldmirror_weights = worldmirror_dir / "model.safetensors"
    worldmirror_configs = [
        worldmirror_dir / "config.yaml",
        worldmirror_dir / "config.json",
    ]
    if not worldmirror_weights.is_file() or not any(path.is_file() for path in worldmirror_configs):
        raise RuntimeError(
            "WorldMirror cache incomplete: "
            f"weights={worldmirror_weights.is_file()} "
            f"config={any(path.is_file() for path in worldmirror_configs)} "
            f"dir={worldmirror_dir}"
        )
    report["required_paths"]["worldmirror_snapshot"] = str(worldmirror_snapshot)
    report["required_paths"]["worldmirror_weights"] = str(worldmirror_weights)
    report["required_paths"]["worldmirror_config"] = str(
        next(path for path in worldmirror_configs if path.is_file())
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
        "worldmirror": cache / "models--tencent--HY-World-2.0" / "blobs",
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


@app.function(image=base_image, timeout=4 * 60 * 60)
def worldgen_case000_stage3(job_id: str = "case000") -> dict:
    """Dispatch Stage 3 to the deployed persistent WorldStereo worker."""
    worker_cls = modal.Cls.from_name("modal-world-stage3", "WorldStereoWorker")
    return _spawn_worker_call(worker_cls().generate, job_id=job_id, wait_timeout_s=45 * 60)


@app.function(
    image=hyworld2_worldgen_stage1_image,
    gpu=GPU,
    cpu=8.0,
    memory=32768,
    volumes={
        "/models": model_cache.with_mount_options(read_only=True),
        "/runtime-cache": runtime_cache,
        "/worldgen": worldgen_outputs,
    },
    secrets=[hf_secret],
    timeout=60 * 60,
)
def worldgen_case000_stage4(job_id: str = "case000") -> dict:
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
    os.environ["CUDA_CACHE_PATH"] = "/runtime-cache/cuda-cache"
    os.environ["CUDA_CACHE_MAXSIZE"] = str(4 * 1024**3)
    os.environ["TORCH_EXTENSIONS_DIR"] = "/runtime-cache/torch-extensions"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/runtime-cache/torchinductor"
    os.environ["TRITON_CACHE_DIR"] = "/runtime-cache/triton"

    target = resolve_worldgen_job_root(job_id)
    generation_bank = target / "render_results/generation_bank_worldstereo-memory-dmd"
    required_stage3 = [generation_bank / "global_pcd.ply", generation_bank / "aligned_pcd.ply"]
    missing_stage3 = [
        str(path.relative_to(target)) for path in required_stage3 if not path.is_file()
    ]
    if missing_stage3:
        raise RuntimeError(f"Stage 3 incomplete: missing {missing_stage3}")

    stage4_inputs = [*required_stage3]
    pcd_info = generation_bank / "pcd_info.json"
    if pcd_info.is_file():
        stage4_inputs.append(pcd_info)
    stage4_manifest = build_stage_manifest(
        job_id=job_id,
        stage="stage4",
        hyworld_revision=HYWORLD2_REVISION,
        input_fingerprint=fingerprint_files(stage4_inputs, root=target),
        config={"save_normal": True, "split_sky": True, "split_align": False},
    )

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
            manifest_ok = manifest_matches(target, "stage4", stage4_manifest)
            legacy_adopted = (
                job_id == "case000" and not stage_manifest_path(target, "stage4").exists()
            )
            if manifest_ok or legacy_adopted:
                if legacy_adopted:
                    write_stage_manifest(target, "stage4", stage4_manifest)
                    worldgen_outputs.commit()
                return {
                    "resumed": True,
                    "manifest_adopted": legacy_adopted,
                    "stage4_s": 0.0,
                    "camera_count": camera_count,
                    "image_count": len(images),
                    "depth_count": len(depths),
                    "normal_count": len(normals),
                    "points_bytes": points_path.stat().st_size,
                    "sky_points_bytes": (
                        sky_points_path.stat().st_size if sky_points_path.is_file() else 0
                    ),
                }

    worldgen_root = Path(HYWORLD2_SOURCE) / "hyworld2/worldgen"
    log_path = target / "stage4.log"
    timing_path = target / "stage4_timing.json"
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-u",
        "gen_gs_data.py",
        "--root_path",
        str(target),
        "--save_normal",
        "--split_sky",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{worldgen_root}:{HYWORLD2_SOURCE}"
    env["PYTHONFAULTHANDLER"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"

    stop_monitor = threading.Event()
    gpu_peak_mib = None

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
                sample_mib = int(raw.strip())
                gpu_peak_mib = max(gpu_peak_mib or 0, sample_mib)
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass

    monitor = None
    if os.environ.get("MODAL_WORLD_DEBUG_GPU_SAMPLER") == "1":
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
        if monitor is not None:
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
        "gpu_sampler_enabled": monitor is not None,
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

    write_stage_manifest(target, "stage4", stage4_manifest)
    worldgen_outputs.commit()
    return {
        **timing,
        "points_bytes": points_path.stat().st_size,
        "sky_points_bytes": (sky_points_path.stat().st_size if sky_points_path.is_file() else 0),
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
def worldgen_case000_stage5_smoke(job_id: str = "case000") -> dict:
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

    target = resolve_worldgen_job_root(job_id)
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
