from __future__ import annotations

from pathlib import Path


def patch_stage1_worldnav(source_root: str | Path) -> None:
    """Apply pinned single-GPU WorldNav safety patches at image build time."""
    root = Path(source_root)
    worldgen_root = root / "hyworld2/worldgen"

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
    if panorama_source.count(old_camera_code) != 1:
        raise RuntimeError("expected pinned get_panorama_cameras_v2 block not found")
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
    if panorama_source.count(old_mesh_cleanup) != 1:
        raise RuntimeError("expected pinned Open3D mesh cleanup block not found")
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
    if panorama_source.count(mesh_assign_old) != 1:
        raise RuntimeError("expected pinned Open3D mesh assignment block not found")
    panorama_source = panorama_source.replace(mesh_assign_old, mesh_assign_new, 1)
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
    if navi_source.count(old_rotation) != 1:
        raise RuntimeError("expected pinned Open3D rotation block not found")
    navi_source = navi_source.replace(old_rotation, new_rotation, 1)

    save_rback_old = "    R_back = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))\n"
    save_rback_new = (
        "    # modal-world: avoid Open3D native rotation in artifact export; it segfaults on this runtime.\n"
        "    # This is exactly get_rotation_matrix_from_xyz((+pi/2, 0, 0)).\n"
        "    R_back = np.array(\n"
        "        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],\n"
        "        dtype=np.float64,\n"
        "    )\n"
    )
    if navi_source.count(save_rback_old) != 1:
        raise RuntimeError("expected pinned save_artifacts rotation-matrix line not found")
    navi_source = navi_source.replace(save_rback_old, save_rback_new, 1)

    save_rotate_old = (
        "    mesh.rotate(R_back, center=(0, 0, 0))\n"
        "    mesh_min_bound = mesh.get_min_bound()\n"
        "    mesh_max_bound = mesh.get_max_bound()\n"
        "    mesh_verts_rotated = np.asarray(mesh.vertices)\n"
        "    mesh_faces_rotated = np.asarray(mesh.triangles)\n"
    )
    save_rotate_new = (
        "    mesh_verts_rotated = np.ascontiguousarray(\n"
        "        np.asarray(mesh.vertices, dtype=np.float64) @ R_back.T,\n"
        "        dtype=np.float64,\n"
        "    )\n"
        "    mesh_min_bound = mesh_verts_rotated.min(axis=0)\n"
        "    mesh_max_bound = mesh_verts_rotated.max(axis=0)\n"
        "    mesh_faces_rotated = np.asarray(mesh.triangles)\n"
    )
    if navi_source.count(save_rotate_old) != 1:
        raise RuntimeError("expected pinned save_artifacts mesh.rotate block not found")
    navi_source = navi_source.replace(save_rotate_old, save_rotate_new, 1)

    debug_mesh_old = "    if len(vis_all_candidates) > 0:\n"
    debug_mesh_new = (
        "    # modal-world: candidate sphere/torus geometry below is debug-only; it is neither saved nor returned.\n"
        "    # Open3D 0.18 segfaults in sphere.translate() on the Blackwell runtime, so skip this no-op visualization.\n"
        "    if False and len(vis_all_candidates) > 0:\n"
    )
    if navi_source.count(debug_mesh_old) != 1:
        raise RuntimeError("expected pinned reconstruction-candidate debug mesh block not found")
    navi_source = navi_source.replace(debug_mesh_old, debug_mesh_new, 1)
    navi_utils_path.write_text(navi_source)

    traj_path = worldgen_root / "traj_generate.py"
    traj_source = traj_path.read_text()
    cache_old = 'HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")'
    cache_new = 'HF_CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))'
    if traj_source.count(cache_old) != 1:
        raise RuntimeError("expected pinned Stage1 HF cache directory assignment not found")
    traj_source = traj_source.replace(cache_old, cache_new, 1)
    mesh_resolution_old = "mesh_h, mesh_w = 960, 1920"
    mesh_resolution_new = "mesh_h, mesh_w = 480, 960  # modal-world single-GPU WorldNav mesh"
    if traj_source.count(mesh_resolution_old) != 1:
        raise RuntimeError("expected pinned WorldNav mesh resolution not found")
    traj_source = traj_source.replace(mesh_resolution_old, mesh_resolution_new, 1)
    traj_path.write_text(traj_source)
