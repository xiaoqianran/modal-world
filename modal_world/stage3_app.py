from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

from .hyworld2_runtime import (
    GPU,
    HYWORLD2_REVISION,
    HYWORLD2_SOURCE,
    hyworld2_worldgen_stage3_image,
)
from .worldgen_job import (
    build_stage_manifest,
    fingerprint_files,
    manifest_matches,
    resolve_worldgen_job_root,
    stage_manifest_path,
    write_stage_manifest,
)

app = modal.App("modal-world-stage3")
model_cache = modal.Volume.from_name("hyworld2-models", create_if_missing=True)
runtime_cache = modal.Volume.from_name(
    "hyworld2-runtime-cache-v2", create_if_missing=True, version=2
)
worldgen_outputs = modal.Volume.from_name("hyworld2-worldgen-output", create_if_missing=True)
hf_secret = modal.Secret.from_name(os.environ.get("MODAL_WORLD_HF_SECRET", "hyworld2-hf"))

_MODEL_TYPE = "worldstereo-memory-dmd"


@app.function(
    image=hyworld2_worldgen_stage3_image,
    cpu=1.0,
    memory=2048,
    timeout=5 * 60,
)
def verify_stage3_module_paths() -> dict[str, Any]:
    """CPU-only import-path preflight for nested WorldMirror subprocesses."""
    import importlib.util
    import os
    import sys

    hyworld2_root = f"{HYWORLD2_SOURCE}/hyworld2"
    worldgen_root = f"{hyworld2_root}/worldgen"
    for path in (hyworld2_root, worldgen_root, HYWORLD2_SOURCE):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(worldgen_root)
    spec = importlib.util.find_spec("worldrecon.pipeline")
    return {
        "success": spec is not None,
        "cwd": os.getcwd(),
        "hyworld2_root": hyworld2_root,
        "worldrecon_origin": getattr(spec, "origin", None),
    }


@app.cls(
    image=hyworld2_worldgen_stage3_image,
    gpu=GPU,
    cpu=16.0,
    memory=131072,
    volumes={
        "/models": model_cache.with_mount_options(read_only=True),
        "/runtime-cache": runtime_cache,
        "/worldgen": worldgen_outputs,
    },
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
    startup_timeout=30 * 60,
    min_containers=0,
    max_containers=1,
    scaledown_window=5 * 60,
)
class WorldStereoWorker:
    @modal.enter()
    def load_models(self) -> None:
        os.environ["HF_HOME"] = "/models/huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/huggingface/hub"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/runtime-cache/torchinductor"
        os.environ["TRITON_CACHE_DIR"] = "/runtime-cache/triton"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        hyworld2_root = f"{HYWORLD2_SOURCE}/hyworld2"
        worldgen_root = f"{hyworld2_root}/worldgen"
        os.environ["PYTHONPATH"] = f"{hyworld2_root}:{worldgen_root}:{HYWORLD2_SOURCE}"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        import sys

        for path in (hyworld2_root, worldgen_root, HYWORLD2_SOURCE):
            if path not in sys.path:
                sys.path.insert(0, path)
        # Preserve upstream video_gen.py cwd semantics. PanoramaMemoryBank invokes
        # ``torchrun -m worldrecon.pipeline`` with cwd="..", which must resolve
        # from hyworld2/worldgen to hyworld2 so the sibling package is importable.
        os.chdir(worldgen_root)

        import importlib.util

        if importlib.util.find_spec("worldrecon.pipeline") is None:
            raise RuntimeError("Stage 3 worker cannot import worldrecon.pipeline")

        import torch
        import torch.distributed as dist
        from models.worldstereo_wrapper import WorldStereo
        from moge.model.v2 import MoGeModel
        from src.sp_utils.parallel_states import initialize_parallel_state
        from torch.distributed.device_mesh import init_device_mesh
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        started = time.perf_counter()
        self.torch = torch
        self.dist = dist
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        if not dist.is_initialized():
            rendezvous = tempfile.mktemp(prefix="modal-world-stage3-pg-")
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                init_method=f"file://{rendezvous}",
                rank=0,
                world_size=1,
            )

        self.device_mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=("rep", "shard"))
        self.parallel_dims = initialize_parallel_state(sp=1)

        self.moge_model = MoGeModel.from_pretrained(
            "Ruicheng/moge-2-vitl-normal", local_files_only=True
        ).to(self.device)
        self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3", local_files_only=True).to(
            self.device, dtype=torch.bfloat16
        )
        self.sam3_processor = Sam3VideoProcessor.from_pretrained(
            "facebook/sam3", local_files_only=True
        )
        self.worldstereo = WorldStereo.from_pretrained(
            "hanshanxue/WorldStereo",
            subfolder=_MODEL_TYPE,
            local_files_only=True,
            sp_world_size=1,
            fsdp=False,
            device_mesh=self.device_mesh,
            device=self.device,
        )
        torch.set_default_dtype(torch.float)
        if torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
        elif torch.cuda.get_device_capability(self.device)[0] >= 7:
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = None
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - started
        self.call_count = 0

    @modal.method()
    def probe(self) -> dict[str, Any]:
        """Validate full Stage 3 GPU model loading without running world generation."""
        torch = self.torch
        return {
            "load_s": round(self.load_s, 3),
            "device_name": torch.cuda.get_device_name(self.device),
            "device_capability": list(torch.cuda.get_device_capability(self.device)),
            "allocated_gib": round(torch.cuda.memory_allocated(self.device) / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(self.device) / 1024**3, 3),
            "autocast_dtype": str(self.autocast_dtype),
            "worldstereo_loaded": self.worldstereo is not None,
            "moge_loaded": self.moge_model is not None,
            "sam3_loaded": self.sam3_model is not None,
        }

    def _stage3_manifest(self, *, job_id: str, target: Path) -> dict[str, Any]:
        camera_files = sorted(target.glob("render_results/*/traj*/camera.json"))
        renders = sorted(target.glob("render_results/*/traj*/render.mp4"))
        masks = sorted(target.glob("render_results/*/traj*/render_mask.mp4"))
        captions = sorted(target.glob("render_results/*/traj*/traj_caption.json"))
        inputs = [*camera_files, *renders, *masks, *captions]
        if not camera_files or not (
            len(camera_files) == len(renders) == len(masks) == len(captions)
        ):
            raise RuntimeError(
                f"Stage 2 incomplete: cameras={len(camera_files)} renders={len(renders)} "
                f"masks={len(masks)} captions={len(captions)}"
            )
        for caption in captions:
            payload = json.loads(caption.read_text())
            if not str(payload.get("prompt", "")).strip():
                raise RuntimeError(f"empty Stage 2 caption: {caption}")
        return build_stage_manifest(
            job_id=job_id,
            stage="stage3",
            hyworld_revision=HYWORLD2_REVISION,
            input_fingerprint=fingerprint_files(inputs, root=target),
            config={
                "model_type": _MODEL_TYPE,
                "align_nframe": 8,
                "max_reference": 8,
                "downsampled_pts": 2_000_000,
                "kb_anomaly_percentile": 90.0,
                "pcd_nb_neighbors": 10,
                "pcd_std_ratio": 2.0,
                "seed": 1024,
            },
        )

    @modal.method()
    def generate(self, job_id: str = "case000", force: bool = False) -> dict[str, Any]:
        worldgen_outputs.reload()

        import imagesize
        import numpy as np
        from diffusers.utils import export_to_video
        from src.data_utils import load_mutli_traj_dataset, sort_trajs
        from src.general_utils import Timer, load_video, set_seed
        from src.retrieval_wm import PanoramaMemoryBank

        torch = self.torch
        target = resolve_worldgen_job_root(job_id)
        manifest = self._stage3_manifest(job_id=job_id, target=target)
        aligned_pcd = target / f"render_results/generation_bank_{_MODEL_TYPE}/aligned_pcd.ply"
        render_list = sort_trajs(str(target / "render_results"))
        results = sorted(target.glob(f"render_results/*/traj*/{_MODEL_TYPE}_result.mp4"))
        expected_count = len(render_list)
        if not force and aligned_pcd.is_file() and len(results) == expected_count:
            manifest_ok = manifest_matches(target, "stage3", manifest)
            legacy_adopted = (
                job_id == "case000" and not stage_manifest_path(target, "stage3").exists()
            )
            if manifest_ok or legacy_adopted:
                if legacy_adopted:
                    write_stage_manifest(target, "stage3", manifest)
                    worldgen_outputs.commit()
                return {
                    "resumed": True,
                    "manifest_adopted": legacy_adopted,
                    "worker_call_index": self.call_count,
                    "worker_load_s": round(self.load_s, 3),
                    "result_count": len(results),
                    "aligned_pcd_exists": True,
                }

        timer = Timer()
        call_index = self.call_count
        self.call_count += 1
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        set_seed(1024)
        generator = torch.Generator(device=self.device).manual_seed(1024)

        if not render_list:
            raise RuntimeError(f"no Stage 2 renderings found under {target}")
        width, height = imagesize.get(f"{'/'.join(render_list[0].split('/')[:-2])}/start_frame.png")
        scene_type = json.loads((target / "meta_info.json").read_text())["scene_type"]
        del scene_type  # Parsed exactly as upstream; retained as an input validity check.

        with timer.track("[IO] Memory Bank Initialization"):
            memory_bank = PanoramaMemoryBank(
                root_path=str(target),
                image_width=width,
                image_height=height,
                device=self.device,
                nframe=self.worldstereo.cfg.nframe,
                max_reference=8,
                align_nframe=8,
                rank=0,
                world_size=1,
                moge_model=self.moge_model,
                sam3_model=self.sam3_model,
                sam3_processor=self.sam3_processor,
                results_name=_MODEL_TYPE,
                valid_threshold=0.15,
                pts_num=2_000_000,
                kb_anomaly_percentile=90.0,
                pcd_nb_neighbors=10,
                pcd_std_ratio=2.0,
            )

        try:
            for render_path in render_list:
                view_id, traj_id = render_path.split("/")[-3:-1]
                camera_path = target / f"render_results/{view_id}/{traj_id}/camera.json"
                target_cameras = json.loads(camera_path.read_text())
                tar_w2cs = torch.from_numpy(np.array(target_cameras["extrinsic"])).to(
                    dtype=torch.float32, device=self.device
                )
                tar_ks = torch.from_numpy(np.array(target_cameras["intrinsic"])).to(
                    dtype=torch.float32, device=self.device
                )
                result_path = (
                    target / f"render_results/{view_id}/{traj_id}/{_MODEL_TYPE}_result.mp4"
                )
                if not force and result_path.is_file():
                    with timer.track("[IO] Reload existing result for memory update"):
                        gen_frames = load_video(str(result_path))
                    memory_bank.update_memory(
                        gen_frames=gen_frames,
                        tar_w2cs_full=tar_w2cs,
                        tar_Ks_full=tar_ks,
                        view_id=view_id,
                        traj_id=traj_id,
                    )
                    continue

                with timer.track("Memory Retrieval"):
                    retrieved_frames, ref_index, ref_index_dict, ref_w2cs, _ = (
                        memory_bank.retrieval(tar_w2cs, tar_ks, view_id=view_id, traj_id=traj_id)
                    )
                    combined_frames = retrieved_frames / 255

                memory_inputs = target / f"render_results/{view_id}/{traj_id}/memory_inputs"
                memory_inputs.mkdir(parents=True, exist_ok=True)
                with timer.track("[IO] Save Memory retrieval results"):
                    export_to_video(
                        combined_frames,
                        str(memory_inputs / f"{_MODEL_TYPE}.mp4"),
                        fps=16,
                    )
                    if ref_index_dict is not None:
                        (memory_inputs / f"{_MODEL_TYPE}_ref_index.json").write_text(
                            json.dumps(ref_index_dict, indent=2)
                        )
                    if ref_w2cs is not None:
                        (memory_inputs / f"{_MODEL_TYPE}_ref_w2cs.json").write_text(
                            json.dumps(ref_w2cs.cpu().numpy().tolist(), indent=2)
                        )

                with timer.track("[IO] Loading meta inputs"):
                    meta_data = load_mutli_traj_dataset(
                        cfg=self.worldstereo.cfg,
                        input_path=str(target / "render_results"),
                        output_path=str(target / "render_results"),
                        view_id=view_id,
                        traj_id=traj_id,
                        device=self.device,
                        ref_index=ref_index,
                        model_type=_MODEL_TYPE,
                        task_type="panorama",
                    )
                pipeline_kwargs = {
                    key: value for key, value in meta_data.items() if value is not None
                }
                pipeline_kwargs.update(
                    negative_prompt=self.worldstereo.cfg.get("negative_prompt", ""),
                    generator=generator,
                    output_type="pt",
                    latent_cond_mode=self.worldstereo.cfg.latent_cond_mode,
                    mode="test",
                )
                with (
                    timer.track("Video Model Inference"),
                    torch.autocast(
                        "cuda",
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_dtype is not None,
                    ),
                ):
                    output = self.worldstereo.pipeline(**pipeline_kwargs).frames[0].float()

                gc.collect()
                torch.cuda.empty_cache()
                with timer.track("[IO] Save Results"):
                    output_np = output.permute(0, 2, 3, 1).cpu().numpy()
                    export_to_video(output_np, str(result_path), fps=16)

                with timer.track("[IO] Reload results for memory update"):
                    gen_frames = load_video(str(result_path))
                memory_bank.update_memory(
                    gen_frames=gen_frames,
                    tar_w2cs_full=tar_w2cs,
                    tar_Ks_full=tar_ks,
                    view_id=view_id,
                    traj_id=traj_id,
                )

            with timer.track("Run World Mirror"):
                memory_bank.apply_worldmirror(skip_exist=True)
            with timer.track("Memory bank Alignment"):
                memory_bank.alignment(debug_mode=False)
            alignment_profile = {
                name: round(value, 4)
                for name, value in getattr(memory_bank, "alignment_profile", {}).items()
            }
            alignment_phase2_profile = {
                name: round(value, 4)
                for name, value in getattr(memory_bank, "alignment_phase2_profile", {}).items()
            }
            alignment_phase2_detail = {
                name: round(value, 4)
                for name, value in getattr(memory_bank, "alignment_phase2_detail", {}).items()
            }
            alignment_phase2_detail["unattributed"] = round(
                max(
                    0.0,
                    alignment_phase2_profile.get("frame_align_total", 0.0)
                    - sum(alignment_phase2_detail.values()),
                ),
                4,
            )
            with timer.track("[IO] Save final aligned pointcloud (update memory)"):
                memory_bank.export_pcd(
                    str(target / f"render_results/generation_bank_{_MODEL_TYPE}"),
                    N_points=2_000_000,
                )
        finally:
            del memory_bank
            gc.collect()
            torch.cuda.empty_cache()

        stage3_s = time.perf_counter() - started
        results = sorted(target.glob(f"render_results/*/traj*/{_MODEL_TYPE}_result.mp4"))
        timing = {
            "stage3_s": round(stage3_s, 3),
            "worker_load_s": round(self.load_s, 3),
            "worker_call_index": call_index,
            "gpu_peak_used_mib": int(torch.cuda.max_memory_allocated() / (1024**2)),
            "result_count": len(results),
            "aligned_pcd_exists": aligned_pcd.is_file(),
            "alignment_profile": alignment_profile,
            "alignment_phase2_profile": alignment_phase2_profile,
            "alignment_phase2_detail": alignment_phase2_detail,
            "timer_records": {
                name: [round(value, 4) for value in values]
                for name, values in timer.records.items()
            },
        }
        (target / "stage3_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        if len(results) != expected_count:
            raise RuntimeError(f"Stage 3 result count mismatch: {len(results)} vs {expected_count}")
        if not aligned_pcd.is_file():
            raise RuntimeError("Stage 3 completed without aligned_pcd.ply")
        write_stage_manifest(target, "stage3", manifest)
        runtime_cache.commit()
        worldgen_outputs.commit()
        return timing
