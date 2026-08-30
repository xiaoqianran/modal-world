from __future__ import annotations

from pathlib import Path


def patch_stage2_single_gpu(source_root: str | Path) -> None:
    """Patch pinned HY-World Stage 2 to avoid distributed overhead on one GPU."""
    root = Path(source_root)

    traj_render = root / "hyworld2/worldgen/traj_render.py"
    source = traj_render.read_text()
    dist_init = (
        "    dist.init_process_group(\n"
        '        backend="cpu:gloo,cuda:nccl",\n'
        "        rank=rank,\n"
        "        world_size=world_size,\n"
        "    )\n"
    )
    dist_init_single = (
        "    if world_size > 1:\n"
        "        dist.init_process_group(\n"
        '            backend="cpu:gloo,cuda:nccl",\n'
        "            rank=rank,\n"
        "            world_size=world_size,\n"
        "        )\n"
    )
    if source.count(dist_init) != 1:
        raise RuntimeError("expected pinned Stage 2 distributed init block not found")
    source = source.replace(dist_init, dist_init_single, 1)

    barrier_count = 0
    patched_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        if line.strip() == "dist.barrier()":
            indent = line[: len(line) - len(line.lstrip())]
            patched_lines.append(f"{indent}if world_size > 1:\n")
            patched_lines.append(f"{indent}    dist.barrier()\n")
            barrier_count += 1
        else:
            patched_lines.append(line)
    if barrier_count != 4:
        raise RuntimeError(f"expected 4 Stage 2 barriers, found {barrier_count}")
    traj_render.write_text("".join(patched_lines))

    pointcloud = root / "hyworld2/worldgen/src/pointcloud.py"
    source = pointcloud.read_text()
    gather_marker = (
        "    pcd_mask = torch.cat(pcd_mask, dim=0).to(torch.float32)  # [f,1,h,w]\n"
        "\n"
        "    dist.barrier()\n"
    )
    single_gpu_return = (
        "    pcd_mask = torch.cat(pcd_mask, dim=0).to(torch.float32)  # [f,1,h,w]\n"
        "\n"
        "    if device_num == 1:\n"
        "        if replace_first_frame:\n"
        "            pcd_renders[0:1] = image_tensor.to(device)\n"
        "            pcd_mask[0:1] = 0\n"
        "        return pcd_renders, pcd_mask\n"
        "\n"
        "    dist.barrier()\n"
    )
    if source.count(gather_marker) != 1:
        raise RuntimeError("expected pinned Stage 2 gather marker not found")
    pointcloud.write_text(source.replace(gather_marker, single_gpu_return, 1))
