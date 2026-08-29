# modal-world

`modal-world` is an extensible Modal provider boundary for world-generation and world-reconstruction models.
It starts with **HY-World 2.0 (HYWorld2)** and keeps the public contract model-agnostic so additional world
models can be added as isolated backends.

## Architecture

```text
caller
  |
  v
modal_world.service
  |
  +--> registry -------------------------+
  |                                     |
  v                                     v
WorldBackend                       future backends
  |
  v
HYWorld2Backend
  |-- reconstruct -> hyworld2.worldrecon.pipeline
  `-- generate    -> reserved until full chain is validated
```

The core rule is **provider contract != model implementation**. ComfyUI is not a runtime dependency.

## Stable provider contract

- `Operation.RECONSTRUCT`: source media -> geometry/reconstruction artifacts.
- `Operation.GENERATE`: source media -> expanded/explorable world.
- `WorldBackend`: model adapter interface.
- `registry`: backend discovery/selection.
- `Capability`: truthful declaration of operation/input/output support.
- `WorldResult`: provider-neutral artifact manifest.

Adding another world model should only require a new `WorldBackend` implementation and one registry entry.

## HYWorld2 status

| Area | Status |
| --- | --- |
| WorldMirror reconstruction contract | wired |
| Pure Python/process invocation | wired |
| ComfyUI runtime | intentionally excluded |
| WorldNav / trajectory | official profile wired |
| Memory-guided WorldStereo expansion | official profile wired |
| GS data preparation | official profile wired |
| 3DGS training | official profile wired |
| Single-GPU community patch profile | reserved, not yet wired |
| End-to-end benchmark on target GPU | **not claimed yet** |

The reconstruction adapter invokes the upstream module entrypoint:

```bash
python -m hyworld2.worldrecon.pipeline --input_path <path>
```

Full generation follows Tencent's official five-stage `hyworld2/worldgen` pipeline. The official profile recommends >=4 GPUs and documents 8-GPU commands; its 3DGS step scaling is x8=1500, x4=2000, x2=4000, x1=8000. A separate single-GPU patched profile will be added rather than silently mutating the official profile.

A pinned HYWorld2 runtime image is the next integration layer. Do not install upstream dynamically at request time.

## Modal packaging direction

Keep HYWorld2/CUDA/Torch/gsplat/model weights in a HYWorld2-specific Modal image/volume. The generic provider
package should remain lightweight. Once this standalone package is validated, move it into `modal-provider/modal-world/`
without changing the external provider contract.

## Development

```bash
python -m pytest -q
```
