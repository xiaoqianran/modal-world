from pathlib import Path

from modal_world.stage1_patch import patch_stage1_worldnav


def test_stage1_dispatches_to_persistent_worldnav_worker():
    app = Path("modal_world/app.py").read_text()
    start = app.index('def worldgen_case000_stage1(job_id: str = "case000")')
    end = app.index("\n\n@app.function", start)
    proxy = app[start:end]
    assert 'modal.Cls.from_name("modal-world-stage2", "WorldNavRenderer")' in proxy
    assert "_spawn_worker_call(worker_cls().generate_nav" in proxy
    assert "Qwen3VLEngine(" not in proxy
    assert "panorama_utils.write_text" not in proxy

    worker = Path("modal_world/stage2_app.py").read_text()
    assert "def generate_nav" in worker
    assert 'urlopen("http://127.0.0.1:8000/v1/models"' in worker
    assert 'stage="stage1"' in worker
    assert '"mesh_resolution": [480, 960]' in worker
    assert "model_load_s" in worker
    assert "runtime_cache.commit()" in worker


def test_stage1_patch_is_image_build_time():
    runtime = Path("modal_world/hyworld2_runtime.py").read_text()
    assert ".run_function(patch_stage1_worldnav" in runtime
    patch = Path("modal_world/stage1_patch.py").read_text()
    assert "get_panorama_cameras_v2" in patch
    assert "Open3D 0.18 native cleanup segfaults" in patch
    assert "NumPy Z-up -> Y-up rotation" in patch
    assert "mesh_h, mesh_w = 480, 960" in patch


def test_stage1_patch_matches_pinned_upstream(tmp_path: Path):
    src = Path("/tmp/hyworld2-src")
    required = (
        "hyworld2/worldgen/src/panorama_utils.py",
        "hyworld2/worldgen/src/navi_utils.py",
        "hyworld2/worldgen/traj_generate.py",
    )
    if not all((src / rel).is_file() for rel in required):
        return
    root = tmp_path / "source"
    for rel in required:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((src / rel).read_text())
    patch_stage1_worldnav(root)
    panorama = (root / required[0]).read_text()
    navi = (root / required[1]).read_text()
    traj = (root / required[2]).read_text()
    assert "pole-safe up selection" in panorama
    assert "skipping Open3D native cleanup/boundary repair" in panorama
    assert "NumPy Z-up -> Y-up rotation" in navi
    assert "mesh_h, mesh_w = 480, 960" in traj


def test_stage1_uses_persistent_hf_cache_and_preloads_hidden_models():
    patch = Path("modal_world/stage1_patch.py").read_text()
    app = Path("modal_world/app.py").read_text()
    assert 'os.environ.get("HUGGINGFACE_HUB_CACHE"' in patch
    assert "def preload_worldnav_stage1_weights" in app
    assert '"naver-iv/zim-anything-vitl"' in app
    assert '"IDEA-Research/grounding-dino-tiny"' in app
    assert '"zim_vit_l_2092/**"' in app
    assert 'zim / "encoder.onnx"' in app
    assert 'zim / "decoder.onnx"' in app


def test_stage1_rejects_hidden_navmesh_failures():
    worker = Path("modal_world/stage2_app.py").read_text()
    assert 'target / "navmesh/metadata.json"' in worker
    for marker in (
        "Navmesh Error:",
        "Path planning failed:",
        "Artifact saving failed:",
        "NavMesh build failed.",
    ):
        assert marker in worker
    assert "failed despite a zero subprocess exit" in worker


def test_stage1_skips_unused_open3d_reconstruction_debug_mesh():
    patch = Path("modal_world/stage1_patch.py").read_text()
    assert "candidate sphere/torus geometry below is debug-only" in patch
    assert "if False and len(vis_all_candidates) > 0:" in patch
    assert "sphere.translate()" in patch


def test_stage1_save_artifacts_avoids_open3d_rotation():
    patch = Path("modal_world/stage1_patch.py").read_text()
    assert "avoid Open3D native rotation in artifact export" in patch
    assert "mesh_verts_rotated = np.ascontiguousarray" in patch
    assert "mesh_min_bound = mesh_verts_rotated.min(axis=0)" in patch
    assert "mesh_max_bound = mesh_verts_rotated.max(axis=0)" in patch


def test_worker_dispatch_timeout_cancels_container():
    app = Path("modal_world/app.py").read_text()
    helper = app[
        app.index("def _spawn_worker_call") : app.index(
            "@app.function", app.index("def _spawn_worker_call")
        )
    ]
    assert ".spawn(job_id=job_id, force=False)" in helper
    assert "call.get(timeout=wait_timeout_s)" in helper
    assert "call.cancel(terminate_containers=True)" in helper
    assert "function_call_id" in helper


def test_stage1_preload_includes_qwen_and_worker_is_offline():
    app = Path("modal_world/app.py").read_text()
    qwen = Path("modal_world/qwen_vlm_server.py").read_text()
    preload = app[
        app.index("def preload_worldnav_stage1_weights") : app.index(
            "@app.function", app.index("def preload_worldnav_stage1_weights") + 10
        )
    ]
    assert '"Qwen/Qwen3-VL-8B-Instruct"' in preload
    assert "AutoProcessor.from_pretrained(model_id, local_files_only=True)" in qwen
    assert (
        "local_files_only=True"
        in qwen[
            qwen.index("Qwen3VLForConditionalGeneration.from_pretrained") : qwen.index(
                ").eval()", qwen.index("Qwen3VLForConditionalGeneration.from_pretrained")
            )
        ]
    )


def test_stage1_cache_verify_is_cpu_only_and_offline():
    app = Path("modal_world/app.py").read_text()
    start = app.index("def verify_worldnav_stage1_cache")
    decorator = app[app.rfind("@app.function(", 0, start) : start]
    body = app[start : app.index("@app.function", start)]
    assert "gpu=" not in decorator
    assert "read_only=True" in decorator
    assert "HF_HUB_OFFLINE" in body
    assert "TRANSFORMERS_OFFLINE" in body
    assert "AutoProcessor.from_pretrained(qwen_id, local_files_only=True)" in body
    assert "missing_shards" in body


def test_stage1_worker_reloads_worldgen_volume_before_job_read():
    worker = Path("modal_world/stage2_app.py").read_text()
    start = worker.index('def generate_nav(self, job_id: str = "case000", force: bool = False)')
    body = worker[start : worker.index("@modal.method()", start)]
    assert "worldgen_outputs.reload()" in body
