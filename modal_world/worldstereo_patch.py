from __future__ import annotations

from pathlib import Path


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} block, found {count}")
    return source.replace(old, new, 1)


def patch_worldstereo_wrapper_source(source: str) -> str:
    """Patch pinned HYWorld2 WorldStereo wrapper for Transformers 5.2 offline single-GPU use."""
    source = _replace_once(
        source,
        "from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel",
        "from transformers import CLIPImageProcessor, CLIPVisionModel, T5TokenizerFast, UMT5EncoderModel",
        "WorldStereo tokenizer import",
    )
    source = _replace_once(
        source,
        """        transformer = cls._load_transformer(\n            cfg,\n            model_type,\n            model_weights_path,\n            sp_world_size=sp_world_size,\n            fsdp=fsdp,\n            device_mesh=device_mesh,\n            device=device,\n        )\n""",
        """        transformer = cls._load_transformer(\n            cfg,\n            model_type,\n            model_weights_path,\n            sp_world_size=sp_world_size,\n            fsdp=fsdp,\n            device_mesh=device_mesh,\n            device=device,\n            local_files_only=local_files_only,\n        )\n""",
        "WorldStereo transformer call",
    )
    source = _replace_once(
        source,
        """        device_mesh,\n        device,\n    ):\n\n        half_dtype = _get_half_dtype()\n""",
        """        device_mesh,\n        device,\n        local_files_only: bool = False,\n    ):\n\n        half_dtype = _get_half_dtype()\n""",
        "WorldStereo transformer signature",
    )
    source = _replace_once(
        source,
        """            transformer = WorldStereoModel.from_pretrained(\n                cfg.base_model,\n                subfolder="transformer",\n                controlnet_cfg=cfg.controlnet_cfg,\n                torch_dtype=half_dtype,\n            )\n""",
        """            transformer = WorldStereoModel.from_pretrained(\n                cfg.base_model,\n                subfolder="transformer",\n                controlnet_cfg=cfg.controlnet_cfg,\n                torch_dtype=half_dtype,\n                local_files_only=local_files_only,\n            )\n""",
        "WorldStereoModel loader",
    )
    source = _replace_once(
        source,
        """            transformer = WorldStereoRefSModel.from_pretrained(\n                cfg.base_model,\n                subfolder="transformer",\n                controlnet_cfg=cfg.controlnet_cfg,\n                torch_dtype=half_dtype,\n            )\n""",
        """            transformer = WorldStereoRefSModel.from_pretrained(\n                cfg.base_model,\n                subfolder="transformer",\n                controlnet_cfg=cfg.controlnet_cfg,\n                torch_dtype=half_dtype,\n                local_files_only=local_files_only,\n            )\n""",
        "WorldStereoRefSModel loader",
    )
    source = _replace_once(
        source,
        '        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)\n',
        '        tokenizer = T5TokenizerFast.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)\n',
        "WorldStereo tokenizer loader",
    )
    return source


def patch_worldstereo_wrapper(path: str | Path) -> None:
    wrapper = Path(path)
    original = wrapper.read_text()
    patched = patch_worldstereo_wrapper_source(original)
    wrapper.write_text(patched)
