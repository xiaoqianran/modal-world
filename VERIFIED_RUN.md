# Verified HY-World 2.0 run

This file records the reproducible state of the validated single-GPU HY-World 2.0 path. It intentionally contains no credentials.

## Canonical runtime

- GPU: `RTX-PRO-6000` (96 GB)
- HY-World 2.0 revision: `df9988efb87bfc0f4947eb3889411cf957478b06`
- CUDA runtime: 12.8
- PyTorch: 2.7.1 + cu128
- Python: 3.11
- Modal Secret name: `hyworld2-hf`

Persistent Modal Volumes:

- `modal-build-artifacts`: pinned native/source runtime artifacts.
- `hyworld2-models`: Hugging Face, Torch/LPIPS and compile caches.
- `hyworld2-worldgen-output`: resumable generation intermediates and final artifacts.
- `hyworld2-inference-output`: WorldMirror reconstruction output.

Do not delete these Volumes for normal reruns. Model/checkpoint blobs are intentionally not stored in Git.

## Persisted cache verification

`hyworld2-models/stage3_cache_verify.json` was verified with `success=true` and `offline=true`.
The verified HF blob cache was `133872963782` bytes and included:

- `hanshanxue/WorldStereo`
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- `Ruicheng/moge-2-vitl-normal`
- `facebook/sam3`
- `facebook/dinov2-base`

LPIPS VGG16 is persisted at `torch/hub/checkpoints/vgg16-397923af.pth` in `hyworld2-models`.
`hyworld2-models/stage5_preflight.json` was also verified with `success=true`.

## Verified case000 pipeline

The complete path was validated without ComfyUI:

```text
image/panorama
  -> WorldMirror / WorldNav
  -> trajectory rendering
  -> WorldStereo memory expansion
  -> WorldMirror alignment
  -> GS dataset
  -> 3DGS optimization smoke
  -> checkpoint + PLY + SPZ
```

Measured successful checkpoints:

| Stage | Result | Measured time | GPU peak |
| --- | --- | ---: | ---: |
| WorldNav Stage 1 | success | 56.170 s (+13.405 s VLM load) | 16.973 GiB allocated |
| WorldStereo Stage 3 | success, 9/9 results | 701.884 s wrapper | 80309 MiB |
| GS data Stage 4 | success, 283 cameras/images/normals | 75.773 s wrapper | 3777 MiB |
| Stage 5 smoke | success, 100 steps | 49.553 s | 3173 MiB |

Stage 5 preflight parsed 283 images, 277 train views, 6 validation views, 166 depth views, 283 normal views and downsampled 2.18M source points to 1,000,000 initial Gaussian points.

## Persisted successful outputs

Important paths in `hyworld2-worldgen-output`:

```text
case000/camera_trajectory/target_camera.json
case000/render_results/generation_bank_worldstereo-memory-dmd/aligned_pcd.ply
case000/render_results/generation_bank_worldstereo-memory-dmd/global_pcd.ply
case000/gs_data/cameras.json
case000/gs_data/points.ply
case000/gs_data/images/
case000/gs_data/depths/
case000/gs_data/normals/
case000/gs_smoke_result/ckpts/ckpt_99_rank0.pt
case000/gs_smoke_result/ply/point_cloud_99.ply
case000/gs_smoke_result/ply/point_cloud_99.spz
case000/stage1.log
case000/stage3.log
case000/stage4.log
case000/stage5_smoke.log
case000/wrapper_timing.json
case000/stage3_timing.json
case000/stage4_timing.json
case000/stage5_smoke_timing.json
```

`sky_points.ply` is optional. `case000` legitimately has no sky point cloud; the official trainer handles this case.

## Rerun entry points

From the package root with Modal credentials configured:

```bash
uv run modal run -m modal_world.app::verify_worldstereo_stage3_cache
uv run modal run -m modal_world.app::preflight_worldgen_case000_stage5
```

The validated stages are exposed as:

```text
worldmirror_office_inference
worldgen_case000_stage1
worldgen_case000_stage2
preload_worldstereo_stage3_weights
verify_worldstereo_stage3_cache
worldgen_case000_stage3
worldgen_case000_stage4
preflight_worldgen_case000_stage5
worldgen_case000_stage5_smoke
```

For a short final-chain validation, use `worldgen_case000_stage5_smoke`; it performs a real 100-step Gaussian optimization and exports checkpoint/PLY/SPZ without the expensive final mesh path.

## Repository ownership

`modal-provider/modal-world` is the sole source of truth. Do not sync changes back to the former standalone repository.
