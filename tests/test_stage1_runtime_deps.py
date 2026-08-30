from pathlib import Path


def test_stage1_image_includes_rtree_for_navmesh_queries():
    runtime = Path("modal_world/hyworld2_runtime.py").read_text()
    start = runtime.index("hyworld2_worldgen_stage1_image =")
    end = runtime.index("hyworld2_worldgen_stage3_image =", start)
    stage1 = runtime[start:end]
    assert '"rtree==1.4.1"' in stage1
