from __future__ import annotations

from pathlib import Path


def patch_stage4_single_gpu(source_root: str | Path) -> None:
    """Patch pinned GS-data preparation for offline, single-GPU execution."""
    script = Path(source_root) / "hyworld2/worldgen/gen_gs_data.py"
    source = script.read_text()

    model_old = (
        '    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)\n'
    )
    model_new = (
        "    moge_model = MoGeModel.from_pretrained(\n"
        '        "Ruicheng/moge-2-vitl-normal", local_files_only=True\n'
        "    ).to(device)\n"
    )
    if source.count(model_old) != 1:
        raise RuntimeError("expected pinned Stage 4 MoGe loader not found")
    source = source.replace(model_old, model_new, 1)

    init_old = (
        "    dist.init_process_group(\n"
        '        backend="cpu:gloo,cuda:nccl",\n'
        "        rank=rank,\n"
        "        world_size=world_size,\n"
        "    )\n"
    )
    init_new = (
        "    if world_size > 1:\n"
        "        dist.init_process_group(\n"
        '            backend="cpu:gloo,cuda:nccl",\n'
        "            rank=rank,\n"
        "            world_size=world_size,\n"
        "        )\n"
    )
    if source.count(init_old) != 1:
        raise RuntimeError("expected pinned Stage 4 distributed init block not found")
    source = source.replace(init_old, init_new, 1)

    gather_old = (
        "    gather_list = [None] * world_size\n"
        "    dist.all_gather_object(gather_list, local_cameras)\n\n"
        "    if rank == 0:\n"
    )
    gather_new = (
        "    if world_size == 1:\n"
        "        return dict(local_cameras) if rank == 0 else {}\n\n"
        "    gather_list = [None] * world_size\n"
        "    dist.all_gather_object(gather_list, local_cameras)\n\n"
        "    if rank == 0:\n"
    )
    if source.count(gather_old) != 1:
        raise RuntimeError("expected pinned Stage 4 camera gather block not found")
    source = source.replace(gather_old, gather_new, 1)

    sizes_old = (
        "            all_sizes = [None] * world_size\n"
        "            dist.all_gather_object(all_sizes, (img_width, img_height))\n"
    )
    sizes_new = (
        "            if world_size == 1:\n"
        "                all_sizes = [(img_width, img_height)]\n"
        "            else:\n"
        "                all_sizes = [None] * world_size\n"
        "                dist.all_gather_object(all_sizes, (img_width, img_height))\n"
    )
    if source.count(sizes_old) != 1:
        raise RuntimeError("expected pinned Stage 4 size gather block not found")
    source = source.replace(sizes_old, sizes_new, 1)

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
        raise RuntimeError(f"expected 4 Stage 4 barriers, found {barrier_count}")
    source = "".join(patched_lines)

    destroy_old = "    dist.destroy_process_group()\n"
    destroy_new = "    if dist.is_initialized():\n        dist.destroy_process_group()\n"
    if source.count(destroy_old) != 1:
        raise RuntimeError("expected pinned Stage 4 process-group cleanup not found")
    script.write_text(source.replace(destroy_old, destroy_new, 1))
