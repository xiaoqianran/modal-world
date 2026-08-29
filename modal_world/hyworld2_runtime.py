from __future__ import annotations

from typing import Any

import modal

ARTIFACT_VOLUME_NAME = "modal-build-artifacts"
GPU = "RTX-PRO-6000"
PYTHON = "3.11"
CUDA = "12.8.1"
TORCH = "2.7.1"
TORCHVISION = "0.22.1"

# Exact bundle hashes are produced and smoke-tested by xiaoqianran/modal-build.
# They are kwargs to Image.run_function, so changing a hash invalidates Modal's image cache.
HYWORLD2_ARTIFACT_BUNDLES: tuple[dict[str, Any], ...] = (
    {
        "tag": "hyworld2-hy-native-py311-cu128-torch271-sm120-v1",
        "archive_sha256": "094e611679e02135e7f4e746d63554145d960aa52c3392ab5db8e1a6bc69f87a",
        "public_release": False,
    },
    {
        "tag": "hyworld2-oss-native-py311-cu128-torch271-sm120-v1",
        "archive_sha256": "2c6b787925dbbbd7df389d77d548db2639f18113705686586bf85ca63902a746",
        "public_release": True,
    },
    {
        "tag": "hyworld2-oss-source-py311-v1",
        "archive_sha256": "c294c84b2645a5105fe911e519927f016956c73058c5e8a97acea375a4ac94b6",
        "public_release": False,
    },
    {
        "tag": "hyworld2-flash-attn-py311-cu128-torch271-sm120-v1",
        "archive_sha256": "7653177eb13c6056066f72cd27c1e3f540ada13d9d6dbf65e5657930e7522952",
        "public_release": True,
    },
)

artifacts_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=False)


def install_artifact_bundles(bundles: tuple[dict[str, Any], ...]) -> None:
    """Install prebuilt HYWorld2 wheels from modal-build into a captured Image layer."""
    import hashlib
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    import zipfile
    from pathlib import Path

    root = Path("/build-artifacts")
    for spec in bundles:
        tag = str(spec["tag"])
        archive = root / f"{tag}.wheels.zip"
        manifest_path = root / f"{tag}.manifest.json"
        if not archive.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"modal-build artifact missing for {tag}")

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != spec["archive_sha256"]:
            raise RuntimeError(f"archive checksum mismatch for {tag}: {digest}")

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("tag") != tag:
            raise RuntimeError(f"manifest tag mismatch for {tag}")
        if manifest.get("archive_sha256") != digest:
            raise RuntimeError(f"manifest archive checksum mismatch for {tag}")
        if bool(manifest.get("public_release")) != bool(spec["public_release"]):
            raise RuntimeError(f"manifest distribution policy mismatch for {tag}")

        with tempfile.TemporaryDirectory(prefix=f"{tag}-") as temp:
            temp_path = Path(temp)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(temp_path)
            wheels = sorted((temp_path / "wheels").glob("*.whl"))
            if not wheels:
                raise RuntimeError(f"no wheels found in {tag}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", *map(str, wheels)],
                check=True,
            )

    # Avoid capturing extracted temporary build state.
    shutil.rmtree("/root/.cache/pip", ignore_errors=True)


hyworld2_artifact_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-runtime-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        f"python -m pip install torch=={TORCH} torchvision=={TORCHVISION} --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install numpy==1.26.4 'rich>=12,<14' 'jaxtyping>=0.2,<0.3'",
    )
    .run_function(
        install_artifact_bundles,
        volumes={"/build-artifacts": artifacts_volume},
        kwargs={"bundles": HYWORLD2_ARTIFACT_BUNDLES},
        timeout=30 * 60,
    )
)

HYWORLD2_REVISION = "df9988efb87bfc0f4947eb3889411cf957478b06"
HYWORLD2_SOURCE = "/opt/HY-World-2.0"

hyworld2_worldmirror_image = (
    hyworld2_artifact_image.pip_install(
        "huggingface_hub>=0.36,<1",
        "omegaconf>=2.3,<3",
        "einops>=0.8,<1",
        "safetensors>=0.5,<1",
        "scipy==1.14.1",
        "timm==1.0.11",
        "opencv-python-headless==4.10.0.84",
        "Pillow>=10,<12",
        "imageio[ffmpeg]>=2.37,<3",
        "trimesh>=4,<5",
        "plyfile>=1,<2",
        "pycolmap==3.10.0",
        "matplotlib==3.10.3",
        "tqdm>=4.66,<5",
        "requests>=2.32,<3",
    )
    .run_commands(
        f"git clone --filter=blob:none https://github.com/Tencent-Hunyuan/HY-World-2.0.git {HYWORLD2_SOURCE}",
        f"cd {HYWORLD2_SOURCE} && git checkout --detach {HYWORLD2_REVISION}",
    )
    .env({"PYTHONPATH": HYWORLD2_SOURCE})
)

hyworld2_worldgen_stage1_image = (
    hyworld2_worldmirror_image.apt_install("ffmpeg", "libgomp1")
    .pip_install(
        "transformers==5.2.0",
        "accelerate>=1.10,<2",
        "peft==0.18.1",
        "diffusers==0.36.0",
        "openai>=1.55,<3",
        "kornia>=0.8,<1",
        "easydict>=1.13,<2",
        "scikit-image==0.25.2",
        "open3d==0.18.0",
        "loguru==0.7.3",
        "decord>=0.6,<1",
        "ftfy>=6.3,<7",
        "regex>=2024.11",
        "zim_anything==0.1",
        "onnx>=1.17,<2",
        "onnxruntime-gpu>=1.20,<2",
        "pycocotools>=2.0.8,<3",
        "cupy-cuda12x==13.6.0",
    )
    .run_commands(
        "python -m pip install --no-deps 'git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38'",
    )
)


hyworld2_worldgen_stage3_image = hyworld2_worldgen_stage1_image.apt_install(
    "build-essential", "ninja-build"
).pip_install(
    "imagesize==1.4.1",
)

hyworld2_worldgen_stage5_image = hyworld2_worldgen_stage3_image.pip_install(
    "tensorboard>=2.19,<3",
    "torchmetrics==1.7.2",
    "viser==0.2.23",
    "tyro==1.0.8",
    "PyYAML>=6,<7",
    "splines>=0.3,<1",
).pip_install(
    "numpy==1.26.4",
    "plyfile==1.1.3",
    "ml-dtypes==0.5.4",
    "ninja>=1.11,<2",
)
